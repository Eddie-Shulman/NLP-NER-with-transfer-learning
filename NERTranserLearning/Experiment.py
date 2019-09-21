import os
from abc import abstractmethod

from keras.preprocessing.text import one_hot

from NERTranserLearning.Datasets import TAGS
from NERTranserLearning.TensorFlowUtils import create_tf_session
from NERTranserLearning.TrainingAnalyzer import TrainingAnalyzer

from NERTranserLearning import Datasets, Models
import keras
import numpy as np
from keras.utils import to_categorical


class Experiment(TrainingAnalyzer):
    """
    This is an abstract class providing platform for conducting experiments. It handle an experiment configuration
    consisting of the model to compile, train data-set and test data-set, and previous model weights (checkpoints)
    to load then runs the training and the prediction.
    """

    MAX_SENTENCE_LEN = 50
    MODEL_BATCH_SIZE = 128

    @property
    @abstractmethod
    def EXPERIMENT_PLAN(self):
        """
        :return: An array of training configurations (currently supports only one configuration!)
        """
        pass

    def __init__(self) -> None:
        super().__init__()
        self.tf_session = create_tf_session()

    def _simple_embedding_sentence_adapter(self, sentences):
        """
        Using keras naive approach for word embedding
        :param sentences:
        :return:
        """
        updated_sentences = [[-1 if len(one_hot(w, 100*1000)) == 0 else one_hot(w, 100*1000)[0] for w in s] for s in sentences]
        return updated_sentences

    def _do_run_train(self, tags_len, tokenized_padded_tag2idx, tokenized_padded_sentences, checkpoint_path_output,
                          model):
        """
        prepares the training input for TF and runs the fit on the given model. Saves the training weights via
        TF checkpoints for future use.
        :param tags_len:
        :param tokenized_padded_tag2idx:
        :param tokenized_padded_sentences:
        :param checkpoint_path_output:
        :param model:
        :return:
        """
        tokenized_padded_tag2idx = [to_categorical(i, num_classes=tags_len) for i in tokenized_padded_tag2idx]

        y_tr = np.array(tokenized_padded_tag2idx)
        X_tr = np.array(tokenized_padded_sentences)

        X_tr = X_tr[:len(X_tr) - len(X_tr) % self.MODEL_BATCH_SIZE]
        y_tr = y_tr[:len(y_tr) - len(y_tr) % self.MODEL_BATCH_SIZE]

        checkpoint_path = checkpoint_path_output
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=2)

        history = model.fit(X_tr, y_tr,
                            batch_size=self.MODEL_BATCH_SIZE, epochs=5, verbose=1, callbacks=[cp_callback])
        print(history.history)
        return history

    def _run_train(self, tf_session, train):
        """
        If there is not TF checkpoint with model weights - dynamically compile a model according to give
        configuration and run the training
        :param tf_session:
        :param train: model configuration
        :return:
        """
        if not os.path.isfile(train['output_checkpoint']):

            # load data
            tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = getattr(Datasets, 'get_%s' % train['dataset'])(self.MAX_SENTENCE_LEN)

            # if simple - transform words to idx
            if 'simple' in train['model']:
                tokenized_padded_sentences = self._simple_embedding_sentence_adapter(tokenized_padded_sentences)

            # create output folder is such doesn't exist
            if not os.path.isdir('/'.join(train['output_checkpoint'].split('/')[:-1])):
                os.mkdir('/'.join(train['output_checkpoint'].split('/')[:-1]))

            # create & train model
            if 'tdd_' in train['model']:
                model = getattr(Models, 'get_%s' % train['model'])(
                    tf_session, TAGS, sentences, freeze_bi_lstm=train['freeze_bi_lstm'],
                    freeze_output_layer=train['freeze_output_layer'], tdd_layer_name='tdd_layer', tf_batch_size=128,
                    max_sentence_len=50, model_checkpoint_path=train['input_checkpoint'])
            else:
                model = getattr(Models, 'get_%s' % train['model'])(
                    tf_session, TAGS, sentences, freeze_bi_lstm=train['freeze_bi_lstm'],
                    freeze_output_layer=train['freeze_output_layer'], crf_layer_name='crf_layer', tf_batch_size=128,
                    max_sentence_len=50, model_checkpoint_path=train['input_checkpoint'])

            self._do_run_train(len(TAGS), tokenized_padded_tag2idx, tokenized_padded_sentences,
                                          checkpoint_path_output=train['output_checkpoint'], model=model)

        else:
            print('_run_train:: found a checkpoint for model training - skipping training')

    def _run_test(self, tf_session, train, test_dataset):
        """
        Compiles a model according to given configuration and runs TF predict on the test data, prints the F1 scores.
        :param tf_session:
        :param train: model configuration
        :param test_dataset:
        :return:
        """
        # load data
        tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = getattr(Datasets, 'get_%s' % test_dataset)(
            self.MAX_SENTENCE_LEN)

        # if simple - transform words to idx
        if 'simple' in train['model']:
            tokenized_padded_sentences = self._simple_embedding_sentence_adapter(tokenized_padded_sentences)

        if 'tdd_' in train['model']:
            model = getattr(Models, 'get_%s' % train['model'])(
                tf_session, TAGS, sentences, tdd_layer_name='tdd_layer', tf_batch_size=128,
                max_sentence_len=50, model_checkpoint_path=train['output_checkpoint'])
        else:
            model = getattr(Models, 'get_%s' % train['model'])(
                tf_session, TAGS, sentences, crf_layer_name='crf_layer', tf_batch_size=128,
                max_sentence_len=50, model_checkpoint_path=train['output_checkpoint'])

        results = self._predict(tf_batch_size=128, tokenized_padded_sentences=tokenized_padded_sentences, model=model)
        f1_score, fq1_score_without_o = self._analyze_training(tf_batch_size=128,
                                                               tokenized_padded_tag2idx=tokenized_padded_tag2idx,
                                                               tags=TAGS, results=results)
        print(
            'test results:\ntrain model:%s\ntrain model input: %s\ntrain dataset: %s\ndataset: %s\nf1: %s | f1(-O): %s'
            % (train['model'], train['input_checkpoint'], train['dataset'], test_dataset, f1_score,
               fq1_score_without_o))

    def run_experiment(self):
        """
        Runs an experiment out of the EXPERIMENT_PLAN (currently only support running one experiment at a time)
        :return:
        """
        for experiment in self.EXPERIMENT_PLAN:
            tf_session = create_tf_session()
            train = experiment['train']
            self._run_train(tf_session, train)

            # test model
            for test_dataset in experiment['test']:
                self._run_test(tf_session, train, test_dataset)
            tf_session.close()
            break
