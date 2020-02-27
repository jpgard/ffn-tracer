"""
Classes for representing FFN datasets.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional

import pandas as pd
import os
import os.path as osp
import tensorflow as tf

from fftracer.utils.features import _int64_feature, _bytes_feature

# a class to represent a seed location
Seed = namedtuple('Seed', ['x', 'y', 'z'])


def offset_dict_to_csv(offset_dict, out_dir):
    """Write a dictionary with {dataset_id: (mean, std)} structure to a csv at out_dir. """
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    out_fp = osp.join(out_dir, "offsets.csv")
    df = pd.DataFrame.from_dict(offset_dict, orient="index").reset_index()
    df.columns = ["dataset_id", "mean", "std"]
    df.to_csv(out_fp, index=False)


class PairedDataset2d(ABC):
    """A dataset consisting of an image (x) and pixel-wise labels (y)."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        # the input grayscale image; an array with shape (height, width) and dtype uint8
        self.x = None
        # the pixel-wise labels for the image; an array with shape (height, width)
        self.y = None
        # pom_pad is the value by which zero labels are increased/1 labels are decreased
        self.pom_pad = 0.05

    @abstractmethod
    def load_data(self, gs_dir, data_dir):
        raise

    def check_xy_shapes_match(self):
        assert self.x.shape == self.y.shape

    @property
    def shape(self):
        self.check_xy_shapes_match()
        return self.x.shape

    @property
    def label_value(self):
        return 1 - self.pom_pad

    def serialize_example(self):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'shape_x': _int64_feature([self.shape[1]]),
            'shape_y': _int64_feature([self.shape[0]]),
            'image_raw': tf.train.Feature(
                float_list=tf.train.FloatList(value=self.x.flatten().tolist())
            ),
            'image_label': tf.train.Feature(
                float_list=tf.train.FloatList(value=self.y.flatten().tolist())
            ),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write_training_coordiates(self, coords, out_dir):
        """Write coords to out_dir as tfrecord."""
        coord_dir = osp.join(out_dir, "coords")
        if not os.path.exists(coord_dir):
            os.makedirs(coord_dir)
        tfrecord_filepath = osp.join(coord_dir,
                                     self.dataset_id + "_coords.tfrecord")
        record_options = tf.io.TFRecordOptions(
            tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        with tf.io.TFRecordWriter(tfrecord_filepath,
                                  options=record_options) as writer:
            for x, y in coords:
                coord_zyx = [0, y, x]  # store in reverse to match ffn formatting
                coord = tf.train.Example(features=tf.train.Features(feature=dict(
                    center=_int64_feature(coord_zyx),
                    label_volume_name=_bytes_feature([self.dataset_id.encode('utf-8')])
                )))
                writer.write(coord.SerializeToString())

    @abstractmethod
    def generate_training_coordinates(self, out_dir, n, **kwargs):
        """Sample a set of training coordinates and write to tfrecord file.

        This method does the work of ffn's build_coordinates.py, but as a class method
        instead of a standalone script.
        """
        pass

    def write_tfrecord(self, out_dir):
        tfrecord_dir = osp.join(out_dir, "tfrecords")
        if not osp.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir)

        tfrecord_filepath = osp.join(tfrecord_dir, self.dataset_id + ".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
            example = self.serialize_example()
            writer.write(example)

    def fetch_mean_and_std(self):
        """Fetch the mean and std for use as offsets during training."""
        return self.x.mean(), self.x.std()
