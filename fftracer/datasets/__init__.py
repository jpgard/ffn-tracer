"""
Classes for representing FFN datasets.
"""

import numpy as np
import pandas as pd
import os.path as osp
import tensorflow as tf

from abc import ABC, abstractmethod
from collections import namedtuple

from fftracer.utils.features import _int64_feature, _bytes_feature

# a class to represent a seed location
Seed = namedtuple('Seed', ['x', 'y', 'z'])


def offset_dict_to_csv(offset_dict, fp):
    """Write a dictionary with {dataset_id: (mean, std)} structure to a csv at fp. """
    df = pd.DataFrame.from_dict(offset_dict, orient="index").reset_index()
    df.columns = ["dataset_id", "mean", "std"]
    df.to_csv(fp, index=False)


class PairedDataset2d(ABC):
    """A dataset consisting of an image (x) and pixel-wise labels (y)."""

    def __init__(self, dataset_id: str, seed: Seed):
        self.dataset_id = dataset_id
        self.x = None  # the input grayscale image
        self.y = None  # the pixel-wise labels for the image
        self.seed = seed
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

    def serialize_example(self):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'shape_x': _int64_feature([self.shape[0]]),
            'shape_y': _int64_feature([self.shape[1]]),
            'seed_x': _int64_feature([self.seed.x]),
            'seed_y': _int64_feature([self.seed.y]),
            'seed_z': _int64_feature([self.seed.z]),
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
        tfrecord_filepath = osp.join(out_dir, self.dataset_id + "_coords.tfrecord")
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
    def generate_training_coordinates(self, out_dir, n):
        """Sample a set of training coordinates and write to tfrecord file.

        This method does the work of ffn's build_coordinates.py, but as a class method
        instead of a standalone script.
        """
        pass

    def write_tfrecord(self, out_dir):
        tfrecord_filepath = osp.join(out_dir, self.dataset_id + ".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
            example = self.serialize_example()
            writer.write(example)

    def fetch_mean_and_std(self):
        """Fetch the mean and std for use as offsets during training."""
        return self.x.mean(), self.x.std()



class SeedDataset:
    def __init__(self, seed_csv):
        self.seeds = pd.read_csv(seed_csv,
                                 dtype={"dataset_id": object, "x": int, "y": int,
                                        "z": int}).set_index("dataset_id")

    def get_seed_loc(self, dataset_id: str):
        try:
            seed_loc = self.seeds.loc[dataset_id, :]
            return Seed(seed_loc.seed_x, seed_loc.seed_y, seed_loc.seed_z)
        except Exception as e:
            print("[WARNING]: see not found for dataset_id %s" % (dataset_id))
