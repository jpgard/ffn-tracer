"""
Train a flood-filling tracer model.

This script mostly follows the logic of the original ffn train.py script, with custom
data loading for mozak/allen institute imaging.

usage:
python train.py --tfrecord_dir ./data/tfrecords \
    --out_dir . --coordinate_dir ./data/coords
"""

import os

from fftracer.utils import features
from fftracer import utils
from fftracer.training import input
from fftracer.training import _get_offset_and_scale_map, _get_permutable_axes, \
    _get_reflectable_axes
from fftracer.training.models.model import FFNTracerModel
from fftracer.training.input import offset_and_scale_patches
from fftracer.training import augmentation
import argparse
import tensorflow as tf
import numpy as np
from absl import flags
from absl import app


FLAGS = flags.FLAGS


flags.DEFINE_string('tfrecord_dir', None, "directory containng tfrecord files of "
                                          "labeled input data volumes")
flags.DEFINE_string("coordinate_dir", None, "directory containng tfrecord files of "
                                            "patch coodinates")
flags.DEFINE_string("out_dir", None, "directory to save to")

flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_integer("epochs", 1, "training epochs")
flags.DEFINE_boolean("debug", False, "produces debugging output")
flags.DEFINE_list('image_offset_scale_map', None,
                  'Optional per-volume specification of mean and stddev. '
                  'Every entry in the list is a colon-separated tuple of: '
                  'volume_label, offset, scale.')
flags.DEFINE_list('permutable_axes', ['1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be permuted '
                  'in order to augment the training data.')

flags.DEFINE_list('reflectable_axes', ['0', '1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be reflected '
                  'in order to augment the training data.')


def define_data_input():
    """Adds TF ops to load input data.
    Mimics structure of function of the same name in ffn.train.py
    """
    permutable_axes = np.array(FLAGS.permutable_axes, dtype=np.int32)
    reflectable_axes = np.array(FLAGS.reflectable_axes, dtype=np.int32)

    volume_map = input.load_tfrecord_dataset(FLAGS.tfrecord_dir,
                                             utils.features.FEATURE_SCHEMA)
    model = FFNTracerModel(deltas=[8, 8, 0], batch_size=FLAGS.batch_size)
    volume_name = "507727402"

    # Fetch sizes of images and labels
    # todo(jpgard): pass these to load_from_numpylike_2d as shape parameter
    label_size = features.get_image_shape(volume_map, volume_name).numpy()
    image_size = features.get_image_shape(volume_map, volume_name).numpy()

    # Fetch a single coordinate and volume name from a queue reading the
    # coordinate files or from saved hard/important examples
    coord, volname = input.load_patch_coordinates(FLAGS.coordinate_dir)
    labels = input.load_from_numpylike_2d(coord, volname, shape=[1, 49, 49],
                                          volume_map=volume_map,
                                          feature_name='image_label')
    label_shape = [1, 49, 49, 1]  # [batch_size, x, y, z]
    loss_weights = tf.constant(np.ones(label_shape, dtype=np.float32))

    patch = input.load_from_numpylike_2d(coord, volname, shape=[1, 49, 49],
                                         volume_map=volume_map, feature_name='image_raw')
    data_shape = [1, 49, 49, 1]  # [batch_size, x, y, z]

    # fetch image_stddev and image_mean
    image_mean, image_stddev = features.get_image_mean_and_stddev(volume_map, volname)

    if ((image_stddev is None or image_mean is None) and
            not FLAGS.image_offset_scale_map):
        raise ValueError('--image_mean, --image_stddev or --image_offset_scale_map '
                         'need to be defined')

    # Apply basic augmentations.
    transform_axes = augmentation.PermuteAndReflect(
        rank=4, permutable_axes=_get_permutable_axes(permutable_axes),
        reflectable_axes=_get_reflectable_axes(reflectable_axes))
    labels = transform_axes(labels)
    patch = transform_axes(patch)
    loss_weights = transform_axes(loss_weights)

    # Normalize image data.
    # TODO(jpgard): the _get_offset_and_scale_map() can be refactored to use the new
    #  tf.py_function as suggested in the deprecation warning which arises when the
    #  train.py script is run; the image_offset_scale_map command line can be
    #  deprecated as well (per-volume mean/std is already being performed, elimintating
    #  the need for this function) which simplifies the function considerably.
    patch = offset_and_scale_patches(
        patch, volname[0],
        offset_scale_map=_get_offset_and_scale_map(),
        default_offset=image_mean.numpy(),
        default_scale=image_stddev.numpy())

    # Create a batches of examples corresponding to the patches, labels, and loss weights.

    patches = tf.data.Dataset.from_tensors(patch).batch(FLAGS.batch_size)
    labels = tf.data.Dataset.from_tensors(labels).batch(FLAGS.batch_size)
    loss_weights = tf.data.Dataset.from_tensors(loss_weights).batch(FLAGS.batch_size)

    return patches, labels, loss_weights, coord, volname


def main(argv):
    """

    :param tfrecord_dir:
    :param out_dir:
    :param debug:
    :param coordinate_dir:
    :param batch_size:
    :param epochs:
    :param image_offset_scale_map:
    permutable_axes: 1-D int32 numpy array specifying the axes that may be
      permuted.
    reflectable_axes: 1-D int32 numpy array specifying the axes that may be
      reflected.
    :return:
    """

    load_data_ops = define_data_input()


    # TODO(jpgard): continue following logic of ffn training here; see ffn train.py L#624


if __name__ == "__main__":
    app.run(main)
