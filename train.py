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
from fftracer.training import _get_offset_and_scale_map, _get_permutable_axes, _get_reflectable_axes
from fftracer.training.models.model import FFNTracerModel
from fftracer.training.input import offset_and_scale_patches
from fftracer.training import augmentation
import argparse
import tensorflow as tf
import numpy as np


def define_data_input(model, queue_batch=None):
    # todo(jpgard): move below logic to here to mimic structure of function of the same
#  name in ffn.train.py



def main(tfrecord_dir, out_dir, debug, coordinate_dir, batch_size, epochs,
         image_offset_scale_map=False, permutable_axes=[0,1],
         reflectable_axes=[0,1]):
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
    permutable_axes = np.array(permutable_axes, dtype=np.int32)
    reflectable_axes = np.array(reflectable_axes, dtype=np.int32)

    volume_map = input.load_tfrecord_dataset(tfrecord_dir,
                                             utils.features.FEATURE_SCHEMA)
    model = FFNTracerModel(deltas=[8, 8, 0], batch_size=batch_size)
    volume_name = "507727402"

    # Fetch sizes of images and labels

    label_size = features.get_image_shape(volume_map, volume_name).numpy()
    image_size = features.get_image_shape(volume_map, volume_name).numpy()

    # TODO(jpgard): verify that this is the size of the input images, not of the
    #  training frames
    label_radii = (label_size // 2).tolist()
    label_size = label_size.tolist()
    image_radii = (image_size // 2).tolist()
    image_size = image_size.tolist()

    # Fetch a single coordinate and volume name from a queue reading the
    # coordinate files or from saved hard/important examples
    coord, volname = input.load_patch_coordinates(coordinate_dir)
    labels = input.load_from_numpylike_2d(coord, volname, shape=[1, 49, 49],
                                          volume_map=volume_map,
                                          feature_name='image_label')
    label_shape = [1, 49, 49, 1]  # [batch_size, x, y, z]
    loss_weights = tf.constant(np.ones(label_shape, dtype=np.float32))

    patch = input.load_from_numpylike_2d(coord, volname, shape=[1, 49, 49],
                                         volume_map=volume_map, feature_name='image_raw')
    data_shape = [1, 49, 49, 1]  # [batch_size, x, y, z]

    # todo(jpgard): fetch image_stddev and image_mean
    image_mean, image_stddev = features.get_image_mean_and_stddev(volume_map, volname)

    if ((image_stddev is None or image_mean is None) and
            not image_offset_scale_map):
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
        default_offset=image_mean.numpy(),
        default_scale=image_stddev.numpy())

    # Create a batches of examples corresponding to the patches, labels, and loss weights.

    patches = tf.data.Dataset.from_tensors(patch).batch(batch_size)
    labels = tf.data.Dataset.from_tensors(labels).batch(batch_size)
    loss_weights = tf.data.Dataset.from_tensors(loss_weights).batch(batch_size)

    # TODO(jpgard): continue following logic of ffn training here; see ffn train.py L#624


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_dir", help="directory containng tfrecord files of "
                                               "labeled input data volumes",
                        required=True)
    parser.add_argument("--coordinate_dir", help="directory containng tfrecord files of "
                                                 "patch coodinates",
                        required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--image_offset_scale_map", default=None)
    args = parser.parse_args()
    main(**vars(args))
