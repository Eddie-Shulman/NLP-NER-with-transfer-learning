import tensorflow as tf
from tensorflow.python.keras.api.keras import backend


def create_tf_session():
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)
    return sess
