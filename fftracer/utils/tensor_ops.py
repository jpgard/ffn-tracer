import tensorflow
import tensorflow as tf

def drop_axis(batch, axis, name=None):
    return tf.squeeze(batch, axis, name)

def add_axis(batch, axis, name=None):
    return tf.expand_dims(batch, axis, name)


def clip_gradients(max_gradient_entry_mag, grads_and_vars):
    if max_gradient_entry_mag > 0.0:
        grads_and_vars = [(tf.clip_by_value(g,
                                            -max_gradient_entry_mag,
                                            +max_gradient_entry_mag), v)
                          for g, v, in grads_and_vars]
    return grads_and_vars
