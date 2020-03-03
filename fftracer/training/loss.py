import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
import ot

from fftracer.utils.testing import assert_array_finite, assert_array_nonnegative


def make_distance_matrix(d):
    """Generate a dense distance matrix of shape (d**2, d**2), where the (i,j)th entry
    represents the euclidean distance between pixels i and j in an image."""
    indices = np.array([i for i in np.ndindex((d, d))])

    # D is a condensed distance matrix Y. For each i and j (where i<j<n),
    # the metric dist(u=X[i], v=X[j]) is computed and stored in entry ij.
    D = pdist(indices)

    # Convert D to a dense matrix of shape (d, d)
    D = squareform(D)
    return D


def normalize_to_histogram(im: np.ndarray, dout=np.float64) -> np.ndarray:
    """
    Flatten im, and normalize so its values sum to 1.

    :param im: the image to normalize.
    :param dout: the dtype to return. Note that POT prefers np.float64 for numerical
    precision.
    :smoothing_constant: small positive float that is added to the sum to ensure it is
    nonzero.
    :return: the flattened, normalized histogram array
    """
    assert_array_finite(im)
    x = im.flatten().astype(dout)
    x_sum = x.sum()
    if x_sum != 0.:
        hist = x / x_sum
    else:  # if all pixels are zero, hist should be a uniform distribution over pixels.
        hist = np.full_like(x, fill_value=float(1) / np.prod(x.shape))
    np.testing.assert_almost_equal(
        hist.sum(), 1.0, err_msg=
        "sum of vector after normalization is %8.9f; expected sum of 1.0" % hist.sum())
    return hist


def compute_ot_loss_matrix(y: np.ndarray, y_hat: np.ndarray, D: np.ndarray):
    """
    Solve the optimal transport problem for the image pixels, and return the OT
    permutation matrix Pi.
    :param y: the ground-truth image.
    :param y_hat: the predicted image.
    :param D: the distance matrix; generate via make_distance_matrix(y.shape[0])
    :param y_hat_as_logits: if True, y_hat is provided as logits.
    :return: Pi, the optimal transport matrix, of shape [d**2, d**2]. The (i,j) entry
    in Pi represents the cost of moving pixel i in y_hat to pixel j in y.
    """
    assert_array_finite(y)
    assert_array_finite(y_hat)
    assert_array_nonnegative(y)
    assert_array_nonnegative(y_hat)

    np.testing.assert_array_equal(y.shape[0], y.shape[1])  # check images are square
    np.testing.assert_array_equal(y.shape, y_hat.shape)  # check images same size
    y_hist = normalize_to_histogram(y)
    y_hat_hist = normalize_to_histogram(y_hat)
    # TODO(jpgard): check for GPU implementation here; if a GPU is visible, use that
    #  instead.
    PI = ot.emd(y_hat_hist, y_hist, D)
    return PI


def compute_ot_loss_matrix_batch(y: np.ndarray, y_hat: np.ndarray, D: np.ndarray):
    """
    Apply compute_ot_loss_matrix() along batch_dim to get Pi for each image.

    :param y: array of shape [batch_size, d, d, 1] representing ground truth.
    :param y_hat: array of shape [batch_size, d, d, 1] representing predictions.
    :param D: distance matrix; this represents the cost of pixel-to-pixel transport.
    :return: np.ndarray of shape [batch_dim, d**2, d**2] where d is the
    total number of pixels in an image. Assumes first axis of y and y_hat is batch_dim.
    """
    assert_array_finite(y)
    assert_array_finite(y_hat)
    pi_batch = list()
    assert y.shape[-1] == 1, "only one-channel images currently supported"
    assert y.shape == y_hat.shape, "y shape is {} but y_hat shape is {}".format(
        y.shape, y_hat.shape)
    # TODO(jpgard): figure out a faster method than iterating over array.
    for i in np.arange(y.shape[0]):
        y_i = y[i, :, :, 0]
        y_hat_i = y_hat[i, :, :, 0]
        PI = compute_ot_loss_matrix(y_i, y_hat_i, D)
        pi_batch.append(PI)
    # concatenate the results into an array
    pi_batch = np.array(pi_batch)
    return pi_batch


def compute_pixel_loss(Pi: np.ndarray, D: np.ndarray, alpha=0.5):
    """

    :param Pi: the optimal transport matrix; shape [dim_source, dim_target]
    :param D: the distance matrix, shape [dim_source, dim_target]
    :param alpha: the weight coefficient for source loss; target loss will be weighted
    by (1 - alpha)
    :return:
    """
    assert 0 <= alpha <= 1, "alpha must be in range [0,1]; you passed {}".format(alpha)
    assert Pi.shape == D.shape, \
        "Pi and D should have same shape; passed shapes {} {}".format(Pi.shape, D.shape)
    PI_D = np.multiply(Pi, D)  # elementwise product of Pi and D.
    source_loss = - PI_D.sum(axis=1)  # sum over j, the target pixels
    target_loss = PI_D.sum(axis=0)
    pixel_loss = alpha * source_loss + (1 - alpha) * target_loss
    d = np.sqrt(Pi.shape[0]).astype(int)  # the original square image dimension
    pixel_loss = pixel_loss.reshape((d, d))
    return pixel_loss


def compute_pixel_loss_batch(Pi: np.ndarray, alpha: np.ndarray, D: np.ndarray):
    """
    Apply compute_pixel_loss() along batch_dim to get loss for each pixel.

    :param Pi: array of shape [batch_size, d**2, d**2]
    :param alpha: array of shape [batch_size,] containing the alpha constant for each
    element in the batch.
    :param D: array of shape [d**2, d**2]
    :return: np.ndarray of shape [batch_dim, d, d] where where d is the length of one
    side of the square input image. Assumes first axis of Pi is batch_dim.
    """
    assert len(Pi.shape) == 3, \
        "Pi should have dimension [batch_size, d**2, d**2, received {}".format(Pi.shape)
    assert Pi.shape[-2:] == D.shape[-2:], \
        "Pi and D must match in final two dimensions; got shapes {} {}".format(
            Pi.shape[-2:], D.shape[-2:]
        )
    assert Pi.shape[0] == alpha.shape[0], \
    "batch_dim of Pi and alpha do not match; batch_dim " \
    "shapes are {} {}".format(Pi.shape[0], alpha.shape[0])
    image_pixel_loss = list()
    for i in np.arange(Pi.shape[0]):
        Pi_i = Pi[i, ...]
        pixel_loss_i = compute_pixel_loss(Pi_i, D, alpha[i])
        image_pixel_loss.append(pixel_loss_i)
    pixel_loss_batch = np.array(image_pixel_loss)
    return pixel_loss_batch


def compute_alpha(y: np.ndarray, y_hat: np.ndarray):
    A = y.astype(np.float64).sum()
    B = y_hat.astype(np.float64).sum()
    alpha = min(B / (2.0 * A), 1.0)
    return alpha


def compute_alpha_batch(y, y_hat):
    batch_size = y.shape[0]
    alphas = np.empty(batch_size)
    for i in np.arange(y.shape[0]):
        y_i = y[i, ...]
        y_hat_i = y_hat[i, ...]
        alpha_i = compute_alpha(y_i, y_hat_i)
        alphas[i] = alpha_i
    return alphas
