"""Feature functions for working with Tensorflow data."""

import tensorflow as tf
from fftracer.training.input import get_dense_array_from_element, get_shape_xy_from_element


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


# a dictionary describing the features from mozak data
FEATURE_SCHEMA = {
    'shape_x': tf.io.FixedLenFeature(1, tf.int64),
    'shape_y': tf.io.FixedLenFeature(1, tf.int64),
    'seed_x': tf.io.FixedLenFeature(1, tf.int64),
    'seed_y': tf.io.FixedLenFeature(1, tf.int64),
    'seed_z': tf.io.FixedLenFeature(1, tf.int64),
    'image_raw': tf.io.VarLenFeature(tf.float32),
    'image_label': tf.io.VarLenFeature(tf.float32),
}


def get_image_shape(volume_map, volume_name):
    """Get the input shape as an (x,y) Tensor"""
    for example_element in volume_map[volume_name]:
        return tf.concat([example_element["shape_x"], example_element["shape_y"]],
                         axis=-1)

def get_image_mean_and_stddev(volume_map, volume_name):
    """Fetch the mean and stddev pixel values for the raw image.

    Returns two Tensors of shape [1,] containing image mean and stddev, respectively."""
    # convert volume_name to string representation for indexing into volume_map
    volume_name = volume_name.numpy()[0].decode("utf-8")
    # the volume is a dataset of size one; take the first(only) element, fetch its
    # corresponding image, and reshape to a 2d array
    element = volume_map[volume_name].__iter__().next()
    shape_xy = get_shape_xy_from_element(element)
    volume = get_dense_array_from_element(element, 'image_raw', shape_xy)
    image_mean = tf.math.reduce_mean(tf.cast(volume, tf.float32))
    image_stddev = tf.math.reduce_std(tf.cast(volume, tf.float32))

    return image_mean, image_stddev
