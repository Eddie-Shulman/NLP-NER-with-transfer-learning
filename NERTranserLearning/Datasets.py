import json

from keras.preprocessing.sequence import pad_sequences
import pandas as pd

from data.juand_r_entity_recognition_datasets.src import utils

import numpy as np

PAD_TAG = '__PAD__'

# global group of tags combining GMB, BTC and Ritter tags
TAGS =          [PAD_TAG,'B-GPE','I-EVE','I-TIM','I-GEO-LOC','I-ORG','B-PERSON','B-NAT','B-TIM','I-ART','B-GEO-LOC', 'I-PERSON','O','I-NAT','I-GPE','B-ART','B-EVE','B-ORG', 'I-FACILITY', 'B-FACILITY', 'I-COMPANY', 'B-COMPANY', 'I-SPORTSTEAM', 'B-SPORTSTEAM', 'I-MUSICARTIST', 'B-MUSICARTIST', 'I-PRODUCT', 'B-PRODUCT', 'I-TVSHOW', 'B-TVSHOW', 'I-MOVIE', 'B-MOVIE', 'I-OTHER', 'B-OTHER']

# tags unique to GMB dataset
GMB_TAGS =      [PAD_TAG,'B-GPE','I-EVE','I-TIM','I-GEO',    'I-ORG','B-PER',   'B-NAT','B-TIM','I-ART','B-GEO',     'I-PER',   'O','I-NAT','I-GPE','B-ART','B-EVE','B-ORG']
GMB_TAGS_2_TAGS = {tag: i for i, tag in enumerate(GMB_TAGS)}

# tags unique to BTC data-set
BTC_TAGS =      [PAD_TAG,'B-GPE','I-EVE','I-TIM','I-LOC',    'I-ORG','B-PER',   'B-NAT','B-TIM','I-ART','B-LOC',     'I-PER',   'O','I-NAT','I-GPE','B-ART','B-EVE','B-ORG']
BTC_TAGS_2_TAGS = {tag: i for i, tag in enumerate(BTC_TAGS)}

# tags unique to ritter data-set
RITTER_TAGS =   [PAD_TAG,'B-GPE','I-EVE','I-TIM','I-GEO-LOC','I-ORG','B-PERSON','B-NAT','B-TIM','I-ART','B-GEO-LOC', 'I-PERSON','O','I-NAT','I-GPE','B-ART','B-EVE','B-ORG', 'I-FACILITY', 'B-FACILITY', 'I-COMPANY', 'B-COMPANY', 'I-SPORTSTEAM', 'B-SPORTSTEAM', 'I-MUSICARTIST', 'B-MUSICARTIST', 'I-PRODUCT', 'B-PRODUCT', 'I-TVSHOW', 'B-TVSHOW', 'I-MOVIE', 'B-MOVIE', 'I-OTHER', 'B-OTHER']
RITTER_TAGS_2_TAGS = {tag: i for i, tag in enumerate(RITTER_TAGS)}

# the ratio between the test and the train portions of data-sets.
TRAIN_TEST_SPLIT = 0.7
# used only for the WSJ data-set to extract a 10% portion of the data for training
MICRO_TRAIN_TEST_SPLIT = 0.1


"""
Helper methods to return complete, train or test data-sets
"""

def get_gmb_dataset(max_sentence_len):

    DATASET_FILE = 'data/entity-annotated-corpus/ner_dataset.csv'

    class SentenceGetter(object):

        def __init__(self, data):
            self.n_sent = 1
            self.data = data
            self.empty = False
            agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                               s["POS"].values.tolist(),
                                                               s["Tag"].values.tolist())]
            self.grouped = self.data.groupby("Sentence #").apply(agg_func)
            self.sentences = [s for s in self.grouped]

        def get_next(self):
            try:
                s = self.grouped["Sentence: {}".format(self.n_sent)]
                self.n_sent += 1
                return s
            except:
                return None

    print('start loading GMB data-set')
    data = pd.read_csv(DATASET_FILE, encoding="latin1")
    data = data.fillna(method="ffill")

    getter = SentenceGetter(data)
    sentences = getter.sentences

    print('start tokenizing GMB data-set')
    tokenized_tag2idx = [[GMB_TAGS_2_TAGS[w[2].upper()] for w in s] for s in sentences]
    tokenized_padded_tag2idx = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_tag2idx, padding="post", value=TAGS.index(PAD_TAG))

    tokenized_sentences = [[w[0] for w in s] for s in sentences]
    tokenized_padded_sentences = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_sentences, padding="post", value=PAD_TAG, dtype=object)

    print('GMB data-set loaded and tokenized')
    return tokenized_padded_tag2idx, tokenized_padded_sentences, sentences


def get_gmb_dataset_train(max_sentence_len):
    """
    Returns the train portion of the gmb data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_gmb_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[:int(len(tokenized_padded_tag2idx)*TRAIN_TEST_SPLIT)], \
           tokenized_padded_sentences[:int(len(tokenized_padded_sentences)*TRAIN_TEST_SPLIT)], \
           sentences[:int(len(sentences)*TRAIN_TEST_SPLIT)]


def get_gmb_dataset_test(max_sentence_len):
    """
    Returns the test portion of the gmb data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_gmb_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[int(len(tokenized_padded_tag2idx)*TRAIN_TEST_SPLIT):], \
           tokenized_padded_sentences[int(len(tokenized_padded_sentences)*TRAIN_TEST_SPLIT):], \
           sentences[int(len(sentences)*TRAIN_TEST_SPLIT):]


def get_btc_dataset(max_sentence_len):
    """
    A twitter dataset annotated at 2016, part of the NER datasets projects
    (https://github.com/juand-r/entity-recognition-datasets)
    :param max_sentence_len:
    :return: tokenized_padded_tag2idx, tokenized_padded_sentences
    """
    sentences = list(utils.read_conll('BTC'))
    # tags = sentence_utils.get_tagset(sentences, True)

    tokenized_tag2idx = [[BTC_TAGS_2_TAGS[w[1].upper()] for w in s] for s in sentences]
    tokenized_padded_tag2idx = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_tag2idx, padding="post", value=TAGS.index(PAD_TAG))

    tokenized_sentences = [[w[0][0] for w in s] for s in sentences]
    tokenized_padded_sentences = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_sentences, padding="post", value=PAD_TAG, dtype=object)

    return tokenized_padded_tag2idx, tokenized_padded_sentences, sentences


def get_btc_dataset_train(max_sentence_len):
    """
    Returns the train portion of the btc data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_btc_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[:int(len(tokenized_padded_tag2idx)*TRAIN_TEST_SPLIT)], \
           tokenized_padded_sentences[:int(len(tokenized_padded_sentences)*TRAIN_TEST_SPLIT)], \
           sentences[:int(len(sentences)*TRAIN_TEST_SPLIT)]


def get_btc_dataset_test(max_sentence_len):
    """
    Returns the test portion of the btc data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_btc_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[int(len(tokenized_padded_tag2idx)*TRAIN_TEST_SPLIT):], \
           tokenized_padded_sentences[int(len(tokenized_padded_sentences)*TRAIN_TEST_SPLIT):], \
           sentences[int(len(sentences)*TRAIN_TEST_SPLIT):]


def get_ritter_dataset(max_sentence_len):
    """
    A twitter dataset annotated at 2011, part of the NER datasets projects
    (https://github.com/juand-r/entity-recognition-datasets)
    :param max_sentence_len:
    :return: tokenized_padded_tag2idx, tokenized_padded_sentences
    """
    sentences = list(utils.read_conll('Ritter'))
    # tags = sentence_utils.get_tagset(sentences, True)

    tokenized_tag2idx = [[RITTER_TAGS_2_TAGS[w[1].upper()] for w in s] for s in sentences]
    tokenized_padded_tag2idx = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_tag2idx, padding="post", value=TAGS.index(PAD_TAG))

    tokenized_sentences = [[w[0][0] for w in s] for s in sentences]
    tokenized_padded_sentences = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_sentences, padding="post", value=PAD_TAG, dtype=object)

    return tokenized_padded_tag2idx, tokenized_padded_sentences, sentences


def get_ritter_dataset_train(max_sentence_len):
    """
    Returns the train portion of the ritter data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """

    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_ritter_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[:int(len(tokenized_padded_tag2idx)*TRAIN_TEST_SPLIT)], \
           tokenized_padded_sentences[:int(len(tokenized_padded_sentences)*TRAIN_TEST_SPLIT)], \
           sentences[:int(len(sentences)*TRAIN_TEST_SPLIT)]


def get_ritter_dataset_test(max_sentence_len):
    """
    Returns the test portion of the ritter data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_ritter_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[int(len(tokenized_padded_tag2idx)*TRAIN_TEST_SPLIT):], \
           tokenized_padded_sentences[int(len(tokenized_padded_sentences)*TRAIN_TEST_SPLIT):], \
           sentences[int(len(sentences)*TRAIN_TEST_SPLIT):]


def get_wsj_dataset(max_sentence_len):
    """
    A super small, around 120 sentence, self labeled, twitts from the Wall Streat Journal Twitter channel
    :param max_sentence_len:
    :return:
    """
    DATASET_RAW_FILE = 'data/wsj_twitts/wsj_twitts_raw.json'
    DATASET_FILE = 'data/wsj_twitts/wsj_twitts.json'
    DATASET_CONLL_FILE = 'data/wsj_twitts/wsj_twitts.conll'

    def raw_to_mini():
        with open(DATASET_RAW_FILE, encoding="utf8") as f:
            tweet_data_raw = json.load(f)

        tweet_data = [{'text': tweet_data_raw[tweet]['full_text'], 'lang': tweet_data_raw[tweet]['lang']} for tweet in tweet_data_raw]
        with open(DATASET_FILE, 'w', encoding="utf8") as f:
            json.dump(tweet_data, f)

    # raw_to_mini()

    def read_json():
        with open(DATASET_FILE, encoding="utf8") as f:
            tweet_data = json.load(f)

        sentences = [tweet['text'] for tweet in tweet_data]

        tokenized_tag2idx = [[TAGS.index('O') for w in s.split(' ')] for s in sentences]
        tokenized_padded_tag2idx = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_tag2idx, padding="post",
                                                 value=TAGS.index(PAD_TAG))

        tokenized_sentences = [[w for w in s.split(' ')] for s in sentences]
        tokenized_padded_sentences = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_sentences,
                                                   padding="post", value=PAD_TAG, dtype=object)

        return tokenized_padded_tag2idx, tokenized_padded_sentences

    def read_conll():
        sentences = []
        line_index = 1
        with open(DATASET_CONLL_FILE, encoding="utf8") as f:
            line = f.readline()
            line_index += 1
            while line != '':
                line = line.replace('\n', '')
                sentence = []
                while line != '':
                    sentence.append((line.split('\t')[0], line.split('\t')[1]))
                    line = f.readline().replace('\n', '')
                if len(sentence) > 0:
                    sentences.append(sentence)
                line = f.readline().replace('\n', '')
                line_index += 1

        tokenized_tag2idx = [[TAGS.index(w[1]) for w in s] for s in sentences]
        tokenized_padded_tag2idx = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_tag2idx, padding="post",
                                                 value=TAGS.index(PAD_TAG))

        tokenized_sentences = [[w[0] for w in s] for s in sentences]
        tokenized_padded_sentences = pad_sequences(maxlen=max_sentence_len, sequences=tokenized_sentences,
                                                   padding="post", value=PAD_TAG, dtype=object)

        return tokenized_padded_tag2idx, tokenized_padded_sentences, sentences

    return read_conll()


def get_ritter_wsj_dataset_train(max_sentence_len):
    """
    Returns the train portion of the wsj data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    r_tokenized_padded_tag2idx, r_tokenized_padded_sentences, r_sentences = get_ritter_dataset(max_sentence_len)
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_wsj_dataset(max_sentence_len)
    return np.concatenate((tokenized_padded_tag2idx[:int(len(tokenized_padded_tag2idx)*MICRO_TRAIN_TEST_SPLIT)], r_tokenized_padded_tag2idx)), \
           np.concatenate((tokenized_padded_sentences[:int(len(tokenized_padded_sentences)*MICRO_TRAIN_TEST_SPLIT)], r_tokenized_padded_sentences)), \
           np.concatenate((sentences[:int(len(sentences)*MICRO_TRAIN_TEST_SPLIT)], r_sentences))


def get_wsj_dataset_test(max_sentence_len):
    """
    Returns the test portion of the wsj data-set. See TRAIN_TEST_SPLIT param for split ratio.
    :param max_sentence_len:
    :return:
    """
    tokenized_padded_tag2idx, tokenized_padded_sentences, sentences = get_wsj_dataset(max_sentence_len)
    return tokenized_padded_tag2idx[int(len(tokenized_padded_tag2idx)*MICRO_TRAIN_TEST_SPLIT):], \
           tokenized_padded_sentences[int(len(tokenized_padded_sentences)*MICRO_TRAIN_TEST_SPLIT):], \
           sentences[int(len(sentences)*MICRO_TRAIN_TEST_SPLIT):]
