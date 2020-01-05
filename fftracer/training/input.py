"""
Functions for processing model input.
"""

import tensorflow as tf
import os
from fftracer import utils
from ffn.utils import bounding_box
import numpy as np


def load_img_and_label_maps_from_tfrecords(tfrecord_dir):
    """
    Load the dataset in tfrecord files, returning maps with {id:image} and {id:labels}.

    :param tfrecord_files: directory of files containg tfrecords
    :return: two dictionaries, each mapping dataset_ids to numpy arrays, which are parsed
    according to the specified feature schema.
    """
    tfrecord_files = [os.path.join(tfrecord_dir, x)
                      for x in os.listdir(tfrecord_dir)
                      if x.endswith(utils.TFRECORD)]
    label_volume_map = dict()
    image_volume_map = dict()
    for tfr_file in tfrecord_files:
        record_iterator = tf.python_io.tf_record_iterator(path=tfr_file)
        basename = os.path.basename(tfr_file)
        dataset_id = os.path.splitext(basename)[0]
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            shape_y = int(example.features.feature['shape_y'].int64_list.value[0])

            shape_x = int(example.features.feature['shape_x'].int64_list.value[0])

            img_1d = example.features.feature['image_raw'].float_list.value

            label_1d = example.features.feature['image_label'].float_list.value

            img = np.array(img_1d).reshape(
                (shape_y, shape_x))
            label = np.array(label_1d).reshape(
                (shape_y, shape_x))

            image_volume_map[dataset_id] = img
            label_volume_map[dataset_id] = label

    return image_volume_map, label_volume_map


def load_patch_coordinates(coordinate_dir):
    """
    Loads coordinates and volume names from file of coordinates.

    Mimics logic from ffn's function with equivalent name.
    :param coordinate_dir: directory containing tfrecord files of coordinates.
    :return: Tuple of coordinates Tensor with (shape `[1, 3]`) and volume name Tensor
    (shape `[1]`). Coordinates have format (z, y, x) with type Int, and volume
    has type String.
    """
    tfrecord_files = [os.path.join(coordinate_dir, x)
                      for x in os.listdir(coordinate_dir)
                      if x.endswith(utils.TFRECORD)]

    record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    filename_queue = tf.train.string_input_producer(tfrecord_files, shuffle=True)
    keys, protos = tf.TFRecordReader(options=record_options).read(filename_queue)
    examples = tf.parse_single_example(protos, features=dict(
        center=tf.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
        label_volume_name=tf.FixedLenFeature(shape=[1], dtype=tf.string),
    ))
    coord = examples['center']
    volname = examples['label_volume_name']
    return (coord, volname)


def load_from_numpylike_2d(coordinates, volume_names, shape, volume_map, name=None):
    """
    Load data from Numpy-like volumes.

    The volume object must support Numpy-like indexing, as well as shape, ndim,
    and dtype properties.  The volume can be 3d or 4d.
    :param coordinates: tensor of shape [1, 3] containing XYZ coordinates of the
        center of the subvolume to load.
    :param volume_names: tensor of shape [1] containing names of volumes to load data
        from.
    :param shape: a 3-sequence giving the XYZ shape of the data to load (where Z is 1).
    :param volume_map: a dictionary mapping volume names to volume objects.  See above
        for API requirements of the Numpy-like volume objects.
    :param name: the op name.
    :return: Tensor result of reading data of shape [1] + shape[::-1] + [num_channels]
    from given center coordinate and volume name.  Dtype matches input volumes.
    """
    start_offset = (np.array(shape) - 1) // 2
    num_channels = 1

    def _load_from_numpylike(coord, volname):
        """Load from coord and volname, handling 3d or 4d volumes."""
        volume = volume_map[volname.decode('ascii')]
        volume = np.expand_dims(volume, axis=-1)  # volume now has shape (X,Y,Z)
        starts = np.array(coord) - start_offset
        slc_zyx = bounding_box.BoundingBox(start=starts, size=shape).to_slice()
        # if volume.ndim == 4:
        #     slc_zyx = np.index_exp[:] + slc_zyx

        data = volume[slc_zyx[::-1]]

        # If 4d, move channels to back.  Otherwise, just add flat channels dim.
        # if data.ndim == 4:
        #     data = np.rollaxis(data, 0, start=4)
        # else:
        data = np.expand_dims(data, 4)  # shape (X,Y,Z, n_channels)

        # Add flat batch dim and return shape (batch_size, X, Y, Z, n_channels)
        data = np.expand_dims(data, 0).astype(np.float32)
        return data

    with tf.name_scope(name, 'LoadFromNumpyLike',
                       [coordinates, volume_names]) as scope:
        # For historical reasons these have extra flat dims.
        coordinates = tf.squeeze(coordinates, axis=0)
        volume_names = tf.squeeze(volume_names, axis=0)
        loaded = tf.py_func(
            _load_from_numpylike, [coordinates, volume_names], [tf.float32],
            name=scope)[0]
        loaded.set_shape([1] + shape[::-1] + [num_channels])
        return loaded


def offset_and_scale_patches(patches,
                             volname,
                             offset_scale_map=(),
                             default_offset=0.0,
                             default_scale=1.0,
                             scope='offset_and_scale_patches'):
    """Apply offset and scale from map matching volname, or defaults.

    Ported from ffn.training.inputs.py.

    Args:
      patches: tensor to apply offset and scale to.
      volname: scalar string tensor (note LoadPatchCoordinates returns a 1-vector
               instead.)
      offset_scale_map: map of string volnames to (offset, scale) pairs.
      default_offset: used if volname is not in offset_scale_map.
      default_scale: used if volname is not in offset_scale_map.
      scope: TensorFlow scope for subops.

    Returns:
      patches cast to float32, less offset, divided by scale for given volname, or
      else defaults.
    """
    with tf.name_scope(scope):
        offset, scale = get_offset_scale(
            volname,
            offset_scale_map=offset_scale_map,
            default_offset=default_offset,
            default_scale=default_scale)
        return (tf.cast(patches, tf.float32) - offset) / scale


def get_offset_scale(volname,
                     offset_scale_map=(),
                     default_offset=0.0,
                     default_scale=1.0,
                     name='get_offset_scale'):
    """Gets offset and scale from map matching volname, or defaults.

    Ported from ffn.training.inputs.py.

    Args:
      volname: scalar string tensor (note LoadPatchCoordinates returns a
               1-vector instead).
      offset_scale_map: map of string volnames to (offset, scale) pairs.
      default_offset: used if volname is not in offset_scale_map.
      default_scale: used if volname is not in offset_scale_map.
      name: scope name.

    Returns:
      Tuple of offset, scale scalar float32 tensors.
    """
    def _get_offset_scale(volname):
        # TODO(jpgard): get this to work with offset_scale_map; produced "TypeError:
        #  unhashable type: 'numpy.ndarray' " at line "if volname in offset_scale_map:"
        # if volname in offset_scale_map:
        #     offset, scale = offset_scale_map[volname]
        # else:
        offset = default_offset
        scale = default_scale
        return np.float32(offset), np.float32(scale)

    offset, scale = tf.py_func(
        _get_offset_scale, [volname], [tf.float32, tf.float32],
        stateful=False,
        name=name)
    offset.set_shape([])
    scale.set_shape([])
    return offset, scale
