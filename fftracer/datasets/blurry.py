import skimage
from scipy.ndimage import gaussian_filter

from fftracer.datasets.mozak import MozakDataset2d
import glob
import numpy as np
from skimage.io import imread
from fftracer.utils import VALID_IMAGE_EXTENSIONS
import os.path as osp
import cv2


class BlurryDataset2D(MozakDataset2d):
    """A dataset where the x is a blurrier, darker version of the ground truth."""

    def get_x_shape(self, img_dir):
        """Fetch the shape of the input image x, and initialize an empty x with the
        same shape."""
        x_file = [f for f in glob.glob(osp.join(img_dir, self.dataset_id + ".*"))
                  if f.endswith(VALID_IMAGE_EXTENSIONS)][0]
        print("[INFO] loading image from {}".format(x_file))
        x = imread(x_file, as_gray=True)
        self.x = np.empty_like(x)

    def load_and_generate_data(self, gs_dir, img_dir):
        """Load the ground-truth data and generate noisy x from it."""
        self.get_x_shape(img_dir)
        self.load_y_data(gs_dir)
        self.x = make_blurry_image(self.y)


def make_blurry_image(y, intensity_reduction=0.75, zero_proportion=0.1,
                      kernel_size=(10,10), noise_variance=1e-4):
    """
    Generate a 'real'-looking version of the image data for y.

    This is accomplished by reducing the image intensity, smoothing it, adding noise,
    and randomly setting parts of the image to zero.

    :param y: the input image.
    :param intensity_reduction: the factor by which to increase the intensity.
    :param zero_proportion: the fraction of pixels in the image to set to zero.
    :param kernel_size: kernel size to use for blurring. Higher values result in more
    blurring but less shape in the resulting image.
    :param noise_variance: variance of the gaussian random noise added.
    :return: a noised/blurred version of the input array.
    """
    print("[INFO] blurring image...")
    # Smooth the image and add noise.
    y = cv2.blur(y, kernel_size)
    y /= y.max()  # gaussian filtering reduces intensity; rescale to range [0,1]
    y = skimage.util.random_noise(y, var=noise_variance)
    # Randomly mask out zero_proportion of pixels in image
    zero_mask = np.random.uniform(low=0.0, high=1.0, size=y.shape) > zero_proportion
    y *= zero_mask.astype(y.dtype)
    # Reduce the intensity by the specified factor.
    y *= intensity_reduction
    print("[INFO] blurring image complete.")
    return y
