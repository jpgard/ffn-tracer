"""
Classes for representing FFN datasets.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from collections import namedtuple

# a class to represent a seed location
Seed = namedtuple('Seed', ['x', 'y', 'z'])


def offset_dict_to_csv(offset_dict, fp):
    """Write a dictionary with {dataset_id: (mean, std)} structure to a csv at fp. """
    df = pd.DataFrame.from_dict(offset_dict, orient="index").reset_index()
    df.columns = ["dataset_id", "mean", "std"]
    import ipdb;ipdb.set_trace()
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

    @abstractmethod
    def serialize_example(self):
        """create a serialized tf.Example"""
        pass

    @abstractmethod
    def generate_training_coordinates(self, out_dir, n):
        """Sample a set of training coordinates and write to tfrecord file.

        This method does the work of ffn's build_coordinates.py, but as a class method
        instead of a standalone script.
        """
        pass

    @abstractmethod
    def write_tfrecord(self, out_dir):
        """write the dataset to a tfrecord file."""

    def fetch_mean_and_std(self):
        """Fetch the mean and std for use as offsets during training."""
        return self.x.mean(), self.x.std()



class SeedDataset:
    def __init__(self, seed_csv):
        self.seeds = pd.read_csv(seed_csv,
                                 dtype={"dataset_id": object, "x": int, "y": int,
                                        "z": int}).set_index("dataset_id")

    def get_seed_loc(self, dataset_id: str):
        seed_loc = self.seeds.loc[dataset_id, :]
        return Seed(seed_loc.seed_x, seed_loc.seed_y, seed_loc.seed_z)
