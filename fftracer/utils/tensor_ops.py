import tensorflow as tf

def drop_axis(batch, axis, name=None):
    return tf.squeeze(batch, axis, name)

def add_axis(batch, axis, name=None):
    return tf.expand_dims(batch, axis, name)