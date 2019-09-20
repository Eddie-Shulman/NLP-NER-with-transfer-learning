import numpy as np
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report


class TrainingAnalyzer(object):

    def _predict(self, tf_batch_size, tokenized_padded_sentences, model):
        """
        Prepares the data for input to TF predict and runs the predicts on a given model
        :param tf_batch_size:
        :param tokenized_padded_sentences:
        :param model:
        :return:
        """
        X_te = tokenized_padded_sentences[:len(tokenized_padded_sentences) - len(tokenized_padded_sentences) % tf_batch_size]
        results = model.predict(np.array(X_te), batch_size=tf_batch_size, verbose=1)
        return results

    def _analyze_training(self, tf_batch_size, tokenized_padded_tag2idx, tags, results):
        """
        Analyzing training data, calculating the F1 score, for both CEG (Complete entity group) and
        -OG(CEG without the O entity label)
        :param tf_batch_size:
        :param tokenized_padded_tag2idx:
        :param tags:
        :param results:
        :return:
        """
        pred = np.argmax(results, axis=-1)
        y_true = tokenized_padded_tag2idx [:len(tokenized_padded_tag2idx) - len(tokenized_padded_tag2idx) % tf_batch_size]
        idx2tag = {i: t for i, t in enumerate(tags)}
        pred_tag = [[idx2tag[i] for i in row] for row in pred]
        y_te_true_tag = [[idx2tag[i] for i in row] for row in y_true]

        eval_tags = list(tags)
        eval_tags.remove('__PAD__')
        f1_score = flat_f1_score(y_true=y_te_true_tag, y_pred=pred_tag, average='weighted', labels=eval_tags)
        eval_tags.remove('O')
        fq1_score_without_o = flat_f1_score(y_true=y_te_true_tag, y_pred=pred_tag, average='weighted', labels=eval_tags)

        return f1_score, fq1_score_without_o

