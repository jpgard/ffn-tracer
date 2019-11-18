"""
Classes for representing FFN datasets.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from collections import namedtuple

# a class to represent a seed location
Seed = namedtuple('Seed', ['x', 'y', 'z'])


class PairedDataset2d(ABC):
    """A dataset consisting of an image (x) and pixel-wise labels (y)."""

    def __init__(self, dataset_id: str, seed: Seed):
        self.dataset_id = dataset_id
        self.x = None  # the input grayscale image
        self.y = None  # the pixel-wise labels for the image
        self.seed = seed


    @abstractmethod
    def load_data(self):
        raise

    def check_xy_shapes_match(self):
        assert self.x.shape == self.y.shape

    @property
    def shape(self):
        self.check_xy_shapes_match()
        return self.x.shape



class SeedDataset:
    def __init__(self, seed_csv):
        self.seeds = pd.read_csv(seed_csv,
                                 dtype={"dataset_id": object, "x": int, "y": int,
                                        "z": int}).set_index("dataset_id")

    def get_seed_loc(self, dataset_id: str):
        seed_loc = self.seeds.loc[dataset_id, :]
        return Seed(seed_loc.seed_x, seed_loc.seed_y, seed_loc.seed_z)
