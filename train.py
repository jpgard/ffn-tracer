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
# from fftracer.training import augmentation
from ffn.training import augmentation
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

flags.DEFINE_list('fov_size', [1, 49, 49], '[z, y, x] size of training fov')


def define_data_input(queue_batch=None):
    """Adds TF ops to load input data.
    Mimics structure of function of the same name in ffn.train.py
    """
    permutable_axes = np.array(FLAGS.permutable_axes, dtype=np.int32)
    reflectable_axes = np.array(FLAGS.reflectable_axes, dtype=np.int32)

    volume_map = input.load_tfrecord_dataset(FLAGS.tfrecord_dir,
                                             utils.features.FEATURE_SCHEMA)
    volume_name = "507727402"

    # Fetch a single coordinate and volume name from a queue reading the
    # coordinate files or from saved hard/important examples
    coord, volname = input.load_patch_coordinates(FLAGS.coordinate_dir)
    labels = input.load_from_numpylike_2d(coord, volname, shape=FLAGS.fov_size,
                                          volume_map=volume_map,
                                          feature_name='image_label')
    # give labels shape [batch_size, x, y, z]
    label_shape = [1] + FLAGS.fov_size[::-1]  # [batch_size, x, y, z]
    labels = tf.reshape(labels, label_shape)

    loss_weights = tf.constant(np.ones(label_shape, dtype=np.float32))

    patch = input.load_from_numpylike_2d(coord, volname, shape=FLAGS.fov_size,
                                         volume_map=volume_map, feature_name='image_raw')
    data_shape = label_shape
    patch = tf.reshape(patch, shape=data_shape)

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
    patch = offset_and_scale_patches(
        patch, volname[0],
        offset_scale_map=_get_offset_and_scale_map(),
        default_offset=image_mean,
        default_scale=image_stddev)



    # Create a batches of examples corresponding to the patches, labels, and loss weights.

    patches = tf.data.Dataset.from_tensors(patch).batch(FLAGS.batch_size)
    labels = tf.data.Dataset.from_tensors(labels).batch(FLAGS.batch_size)
    loss_weights = tf.data.Dataset.from_tensors(loss_weights).batch(FLAGS.batch_size)

    ## Previous (broken) code to generate batches from ffn.train.py; this is likely
    ## broken due to updates to the API since tf 1.4 which break backward compatibility.

    ## Set the shapes for all tensors; must have predefined shape
    # patch.set_shape(data_shape)
    # labels.set_shape(label_shape)
    # loss_weights.set_shape(label_shape)

    ## Create a batch of examples. Note that any TF operation before this line
    ## will be hidden behind a queue, so expensive/slow ops can take advantage
    ## of multithreading.

    # patches, labels, loss_weights = tf.train.shuffle_batch(
    #     [patch, labels, loss_weights], queue_batch,
    #     num_threads=max(1, FLAGS.batch_size // 2),
    #     capacity=32 * FLAGS.batch_size,
    #     min_after_dequeue=4 * FLAGS.batch_size,
    #     enqueue_many=True
    # )

    return patches, labels, loss_weights, coord, volname


def prepare_ffn(model):
  """Creates the TF graph for an FFN.

  Ported from ffn.train.py.
  """
  shape = [FLAGS.batch_size] + list(model.pred_mask_size[::-1]) + [1]

  model.labels = tf.placeholder(tf.float32, shape, name='labels')
  model.loss_weights = tf.placeholder(tf.float32, shape, name='loss_weights')
  model.define_tf_graph()


def main(argv):
    model = FFNTracerModel(deltas=[8, 8, 0], batch_size=FLAGS.batch_size,
                           fov_size=FLAGS.fov_size[::-1])
    load_data_ops = define_data_input()
    prepare_ffn(model)



if __name__ == "__main__":
    app.run(main)
