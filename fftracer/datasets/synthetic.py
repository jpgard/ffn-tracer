import numpy as np
from fftracer.datasets import PairedDataset2d
from fftracer.datasets.mozak import MozakDataset2d
from scipy.ndimage import gaussian_filter
import skimage


class SyntheticDataset2D(PairedDataset2d):
    """A simple synthetic dataset for testing."""

    def __init__(self, dataset_id="synthetic"):
        super(SyntheticDataset2D, self).__init__(dataset_id)

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
        px, py = patch_size
        self.x[0:px, 0:py] = 255
        self.y[0:px, 0:py] = 1 - self.pom_pad

    def generate_and_write_training_coordinates(self, out_dir, n):
        """Write n replicates of a fixed training coordinate to tfrecord."""
        coords = [(65, 65) for _ in range(n)]
        self._write_training_coordinates(coords, out_dir)


def make_blurry_image(y, intensity_reduction=0.75, zero_proportion=0.1,
                      smoothing_factor=0.125, noise_variance=1e-4):
    """
    Generate a 'real'-looking version of the image data for y.

    This is accomplished by reducing the image intensity, smoothing it, adding noise,
    and randomly setting parts of the image to zero.

    :param y: the input image.
    :param intensity_reduction: the factor by which to increase the intensity.
    :param zero_proportion: the fraction of pixels in the image to set to zero.
    :param smoothing_factor: higher values result in more smoothing but preserve less
    structure in the image.
    :param noise_variance: variance of the gaussian random noise added.
    :return: a noised/blurred version of the input array.
    """
    # Smooth the image and add noise.
    y = gaussian_filter(y, sigma=smoothing_factor * y.shape[0])
    y /= y.max()  # gaussian filtering reduces intensity; rescale to range [0,1]
    y = skimage.util.random_noise(y, var=noise_variance)
    # Randomly mask out zero_proportion of pixels in image
    zero_mask = np.random.uniform(low=0.0, high=1.0, size=y.shape) > zero_proportion
    y *= zero_mask.astype(y.dtype)
    # Reduce the intensity by the specified factor.
    y *= intensity_reduction
    return y


class BlurryDataset2D(MozakDataset2d):
    """A dataset where the x is a blurrier, darker version of the ground truth."""

    def load_and_generate_data(self, gs_dir):
        """Load the ground-truth data and generate noisy x from it."""
        self.load_y_data(gs_dir)
        import ipdb;ipdb.set_trace()
        self.x = make_blurry_image(self.y)
