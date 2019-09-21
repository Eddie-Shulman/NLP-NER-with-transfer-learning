import os

import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras import Model
from keras_contrib.layers import CRF

# the folder for TF cache hub, uncomment for a non default location
os.environ['TFHUB_CACHE_DIR'] = 'D:\\Dev\\IntroToNLP_22933\\final\\NER Transfer Learning\\tf_cache'

"""
Helper methods to dynamically compile a model by a given configuration and load weights from a TF checkpoint, 
if such is given. 
"""

def _get_crf_layer(input_layer, tags_len, layer_name):
    """
    Returns an implementation of the CRF layer
    :param input_layer:
    :param tags_len:
    :param layer_name:
    :return:
    """
    crf = CRF(tags_len, name=layer_name)
    return crf, crf(input_layer)


def _get_time_distributed_dense_layer(tags_len, bi_lstm_layers, layer_name):
    """
    return an implementation of the TDD layer
    :param tags_len:
    :param bi_lstm_layers:
    :param layer_name:
    :return:
    """
    lstm_units = 512
    return TimeDistributed(Dense(tags_len, activation="softmax"), name=layer_name)(bi_lstm_layers)


def _get_bi_lstm_layers(embedding_layer):
    """
    compiles 2 residually connected bi-lstm layers with the embedding layer and input
    :param embedding_layer:
    :return:
    """
    lstm_units=512

    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name='bidirectional_1')(embedding_layer)
    x_rnn = Bidirectional(LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), name='bidirectional_2')(x)
    x_conn = add([x, x_rnn])  # residual connection to the first biLSTM
    return x_conn


def _get_embedding_layer(tf_session, tf_batch_size, sentences, max_sentence_len, embedding_type):
    """
    returns a naive or elmo embedding layer
    :param tf_session:
    :param tf_batch_size:
    :param sentences:
    :param max_sentence_len:
    :param embedding_type: simple or elmo
    :return:
    """
    words_len = 0
    for s in sentences:
        words_len += len(set(s))

    def simple_embedding_layer(input_text):
        return Embedding(input_dim=100*1000, output_dim=50, input_length=max_sentence_len, mask_zero=True, name='simple_embedding')(input_text)

    def elmo_embedding_layer(input_text):
        TENSORFLOW_HUB_ELMO_MODULE = 'https://tfhub.dev/google/elmo/2'
        EMBEDDING_VECTOR_SIZE = 1024

        elmo_model = hub.Module(TENSORFLOW_HUB_ELMO_MODULE, trainable=True)
        tf_session.run(tf.compat.v1.global_variables_initializer())
        tf_session.run(tf.compat.v1.tables_initializer())

        def elmo_embedding(x):
            return elmo_model(inputs={
                "tokens": tf.squeeze(tf.cast(x, tf.string)),
                "sequence_len": tf.constant(tf_batch_size * [max_sentence_len])
            },
                signature="tokens",
                as_dict=True)["elmo"]

        embedding = Lambda(elmo_embedding, output_shape=(max_sentence_len, EMBEDDING_VECTOR_SIZE), name='elmo_embedding')(input_text)
        return embedding

    if embedding_type == 'simple':
        input_text = Input(shape=(max_sentence_len,))
        return input_text, simple_embedding_layer(input_text)
    elif embedding_type == 'elmo':
        input_text = Input(shape=(max_sentence_len,), dtype=tf.string)
        return input_text, elmo_embedding_layer(input_text)


def get_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
              crf=False, crf_layer_name='crf_layer', tdd_layer_name='tdd_layer',
              tf_batch_size=128, max_sentence_len=50, embedding_type='simple', model_checkpoint_path=None):
    """
    compiles a complete model out of the given configuration and load weights from a TF checkpoint,
    if such is given.
    :param tf_session:
    :param tags:
    :param sentences:
    :param freeze_bi_lstm:
    :param freeze_output_layer:
    :param crf:
    :param crf_layer_name:
    :param tdd_layer_name:
    :param tf_batch_size:
    :param max_sentence_len:
    :param embedding_type:
    :param model_checkpoint_path:
    :return:
    """
    input_text, embedding_layer = _get_embedding_layer(tf_session, tf_batch_size, sentences,
                                                       max_sentence_len, embedding_type)
    bi_lstm_layers = _get_bi_lstm_layers(embedding_layer)

    if crf:
        crf_obj, output_layer = _get_crf_layer(bi_lstm_layers, len(tags), crf_layer_name)
        loss = crf_obj.loss_function
        metrics = [crf_obj.accuracy]
    else:
        output_layer = _get_time_distributed_dense_layer(len(tags), bi_lstm_layers, tdd_layer_name)
        loss = tf.keras.losses.categorical_crossentropy #tf.keras.losses.sparse_categorical_crossentropy
        metrics = ['accuracy']

    model = Model(input_text, output_layer)

    if freeze_bi_lstm:
        for i in range(2, 5):
            model.layers[i].trainable = False

    if freeze_output_layer:
        model.layers[5].trainable = False

    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    model.summary()

    if model_checkpoint_path is not None and os.path.isfile('%s' % model_checkpoint_path):
        model.load_weights(model_checkpoint_path, True)

    return model

"""
Helper methods with some params preset to easily use the get_model function
"""
def get_crf_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
              crf_layer_name='crf_layer', tf_batch_size=128, max_sentence_len=50,
                  embedding_type='simple', model_checkpoint_path=None):

    return get_model(tf_session, tags, sentences, freeze_bi_lstm, freeze_output_layer, True, crf_layer_name, '',
                     tf_batch_size, max_sentence_len, embedding_type, model_checkpoint_path)


def get_crf_simple_embedding_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
              crf_layer_name='crf_layer', tf_batch_size=128, max_sentence_len=50, model_checkpoint_path=None):

    return get_crf_model(tf_session, tags, sentences, freeze_bi_lstm, freeze_output_layer, crf_layer_name,
                         tf_batch_size, max_sentence_len, 'simple', model_checkpoint_path)


def get_crf_elmo_embedding_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
              crf_layer_name='crf_layer', tf_batch_size=128, max_sentence_len=50, model_checkpoint_path=None):

    return get_crf_model(tf_session, tags, sentences, freeze_bi_lstm, freeze_output_layer, crf_layer_name,
                         tf_batch_size, max_sentence_len, 'elmo', model_checkpoint_path)


def get_tdd_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
                  tdd_layer_name='tdd_layer', tf_batch_size=128, max_sentence_len=50, embedding_type='simple',
                  model_checkpoint_path=None):

    return get_model(tf_session, tags, sentences, freeze_bi_lstm, freeze_output_layer, False, '',
                     tdd_layer_name, tf_batch_size, max_sentence_len, embedding_type, model_checkpoint_path)


def get_tdd_simple_embedding_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
              tdd_layer_name='tdd_layer', tf_batch_size=128, max_sentence_len=50, model_checkpoint_path=None):

    return get_tdd_model(tf_session, tags, sentences, freeze_bi_lstm, freeze_output_layer, tdd_layer_name,
                         tf_batch_size, max_sentence_len, 'simple', model_checkpoint_path)


def get_tdd_elmo_embedding_model(tf_session, tags, sentences, freeze_bi_lstm=False, freeze_output_layer=False,
              tdd_layer_name='tdd_layer', tf_batch_size=128, max_sentence_len=50, model_checkpoint_path=None):

    return get_tdd_model(tf_session, tags, sentences, freeze_bi_lstm, freeze_output_layer, tdd_layer_name,
                         tf_batch_size, max_sentence_len, 'elmo', model_checkpoint_path)

