"""Feature functions for working with Tensorflow data."""

import tensorflow as tf

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(values: list):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values: list):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(values: list):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
