import numpy as np
from fftracer.datasets import PairedDataset2d


class SyneticDataset2D(PairedDataset2d):
    """A simple synthetic dataset for testing."""

    def __init__(self, dataset_id="synthetic"):
        super(SyneticDataset2D, self).__init__(dataset_id)

    def load_data(self, gs_dir, data_dir):
        raise

    def initialize_synthetic_data_patch(self, dataset_shape, patch_size):
        """
        Generate a synthetic dataset.

        Creates an array by setting the top-left region of patch_size to the positive
        class label; this creates a dataset where a region in the upper-left corner is
        positively-labeled and the remaning dataset has the negative label.

        :param dataset_shape: (x,y) shape of dataset to generate.
        :param patch_size: (x,y) size of the patch to set to positive class label.
        """
        self.x = np.zeros(dataset_shape, np.float32)
        # the pixel-wise labels for the image
        self.y = np.full(dataset_shape, fill_value=0.05, dtype=np.float32)
        px,py = patch_size
        self.x[0:px, 0:py] = 255
        self.y[0:px, 0:py] = 1 - self.pom_pad

    def generate_training_coordinates(self, out_dir, n):
        """Write n replicates of a fixed training coordinate to tfrecord."""
        coords = [(65, 65) for _ in range(n)]
        self.write_training_coordiates(coords, out_dir)



