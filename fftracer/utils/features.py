"""Feature functions for working with Tensorflow data."""

import tensorflow as tf
from fftracer.training.input import get_dense_array_from_element, \
    get_shape_xy_from_element


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


def get_image_mean_and_stddev(volume_map, volume_names, name=None,
                              feature_name='image_raw'):
    """Fetch the mean and stddev pixel values for the raw image.

    :param volume_map: a dictionary mapping volume names to volume objects.  See above
        for API requirements of the Numpy-like volume objects.
    :param volume_names: tensor of shape [1] containing names of volumes to load data
        from.
    :param name: name for the op.
    :param feature_name: the name of the feature to grab from the volume.

    Returns two Tensors of shape [1,] containing image mean and stddev, respectively."""

    def _compute_feature_mean(volname):
        """Load from coord and volname, handling 3d or 4d volumes."""
        volume_data = volume_map[volname.decode('ascii')]
        volume = tf.sparse.to_dense(volume_data[feature_name])
        image_mean = tf.math.reduce_mean(tf.cast(volume, tf.float32))
        return image_mean

    def _compute_feature_stddev(volname):
        """Load from coord and volname, handling 3d or 4d volumes."""
        volume_data = volume_map[volname.decode('ascii')]
        volume = tf.sparse.to_dense(volume_data[feature_name])
        image_stddev = tf.math.reduce_std(tf.cast(volume, tf.float32))
        return image_stddev

    with tf.name_scope(name, 'GetImageMeanAndStddev',
                       [ volume_names]) as scope:
        # For historical reasons these have extra flat dims.
        volume_names = tf.squeeze(volume_names, axis=0)
        image_mean = tf.py_func(
            _compute_feature_mean, [volume_names], [tf.float32],
            name='GetImageMean')[0]
        image_stddev = tf.py_func(
            _compute_feature_stddev, [volume_names], [tf.float32],
            name='GetImageStddev')[0]
        return image_mean, image_stddev
