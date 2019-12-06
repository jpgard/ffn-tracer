"""
Functions for processing model input.
"""

import tensorflow as tf
import os
from fftracer import utils
from ffn.utils import bounding_box
import numpy as np


def load_tfrecord_dataset(tfrecord_dir, feature_schema):
    """
    Load and parse the dataset in tfrecord files.
    :param tfrecord_files: directory of files containg tfrecords
    :param feature_schema: dict of {feature_name, tf.io.<feature_type>}
    :return: a dictionary mapping dataset_ids to datasets, which are parsed
    according to the specified feature schema.
    """
    tfrecord_files = [os.path.join(tfrecord_dir, x)
                      for x in os.listdir(tfrecord_dir)
                      if x.endswith(utils.TFRECORD)]
    volume_map = dict()
    for file in tfrecord_files:
        basename = os.path.basename(file)
        dataset_id = os.path.splitext(basename)[0]
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_files)

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_schema)

        parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
        volume_map[dataset_id] = parsed_image_dataset
        # TODO(jpgard): verify this is being parsed correctly; the dataset should
        #  contain instances of examples which contain Tensors that can be access for
        #  training
    return volume_map


def get_batch(dataset, parse_example_fn):
    """
    Create a batch of training instances from a tf.Example of a complete neuron.
    :param dataset: a dataset containing training examples.
    :param parse_example_fn: callable which returns a dict with keys "x", "y", "seed".
    Yields:
        tuple of:
          seed array, shape [b, z, y, x, 1]
          image array, shape [b, z, y, x, 1]
          label array, shape [b, z, y, x, 1]

        where 'b' is the batch_size.
    """
    for element in dataset:
        parsed_example = parse_example_fn(element)
    return


def parse_example(example: dict):
    """
    Parse an example in the default format.
    :param example: an element of a tensorflow.python.data.ops.dataset_ops.MapDataset
    :return: dict with keys "x", "y", "seed" with corresponding Tensors.
    """
    shape = tf.concat([example['shape_x'], example['shape_y']], axis=-1)
    image_raw = tf.reshape(tf.sparse.to_dense(example['image_raw']), shape)
    image_label = tf.reshape(tf.sparse.to_dense(example['image_label']), shape)
    seed = tf.concat([example['seed_x'], example['seed_y'], example['seed_z']], axis=-1)
    #     to see images:
    #     print("[DEBUG] saving debug images of raw image and trace")
    #     skimage.io.imsave("tfrecord_image_raw.jpg", image_raw.numpy())
    #     skimage.io.imsave("tfrecord_image_label.jpg", image_label.numpy())
    return {"x": image_raw, "y": image_label, "seed": seed}


def load_patch_coordinates(coordinate_dir):
    """
    Loads coordinates and volume names from file of coordinates.

    Mimics logic from ffn's function with equivalent name.
    :param coordinate_dir: directory containing tfrecord files of coordinates.
    :return: Tuple of coordinates (shape `[1, 3]`) and volume name (shape `[1]`) tensors.
    """
    tfrecord_files = [os.path.join(coordinate_dir, x)
                      for x in os.listdir(coordinate_dir)
                      if x.endswith(utils.TFRECORD)]
    dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        compression_type='GZIP')
    record = dataset.__iter__().next()

    feature_description = {
        "center": tf.io.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
        "label_volume_name": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_example = _parse_function(record)

    coord = parsed_example['center']
    volname = parsed_example['label_volume_name']
    return (coord, volname)


def get_dense_array_from_element(element, feature_name, shape):
    """Fetch the sparse array for feature_name, densify, and reshape."""
    volume_sparse = element[feature_name]
    volume = tf.sparse.to_dense(volume_sparse)
    volume = tf.reshape(volume, shape)  # volume now has shape (X,Y)
    return volume


def get_shape_xy_from_element(element):
    shape_xy = [element['shape_x'].numpy().tolist()[0],
                element['shape_y'].numpy().tolist()[0]]
    return shape_xy


def load_from_numpylike_2d(coordinates, volume_name, shape, volume_map, feature_name):
    """
    Load data from Numpy-like volumes.

    The volume object must support Numpy-like indexing, as well as shape, ndim,
    and dtype properties.  The volume can be 3d or 4d.
    :param coordinates: tensor of shape [1, 3] containing ZYX coordinates of the
        center of the subvolume to load.
    :param volume_name: tensor of shape [1] containing names of volumes to load data
        from.
    :param shape: a 3-sequence giving the ZYX shape of the data to load (where Z is 1).
    :param volume_map: a dictionary mapping volume names to volume objects.  See above
        for API requirements of the Numpy-like volume objects.
    :param feature_name: the name of the feature to grab from the volume.
    :return: Tensor result of reading data of shape [1] + shape[::-1] + [num_channels]
  from given center coordinate and volume name.  Dtype matches input volumes.
    """
    start_offset = (np.array(shape) - 1) // 2
    # convert volume_name to string representation for indexing into volume_map
    volume_name = volume_name.numpy()[0].decode("utf-8")
    # the volume is a dataset of size one; take the first(only) element, fetch its
    # corresponding image, and reshape to a 2d array
    element = volume_map[volume_name].__iter__().next()
    shape_xy = get_shape_xy_from_element(element)
    volume = get_dense_array_from_element(element, feature_name, shape_xy)
    volume = tf.expand_dims(volume, axis=-1) # volume now has shape (X,Y,Z)
    starts = np.array(coordinates) - start_offset
    # BoundingBox returns slice in XYZ order, so these can be used to slice the volume
    slc = bounding_box.BoundingBox(start=starts.ravel(), size=shape).to_slice()
    data = volume[slc]
    # Add flat batch dim
    data = np.expand_dims(data, 0)
    # return data with shape [batch_dim, X, Y, Z]
    return data
