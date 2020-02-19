import numpy as np

from scipy.spatial.distance import pdist, squareform
import ot

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
    :param dout: the dtype to return. Note that pot prefers np.float64 for numerical
    precision.
    :return: the flattened, normalized histogram array
    """
    hist = im.flatten().astype(dout)
    hist = hist / hist.sum()
    assert hist.sum() == 1.0
    return hist


def compute_ot_loss_matrix(y: np.ndarray, y_hat: np.ndarray, D: np.ndarray):
    """
    Solve the optimal transport problem for the image pixels, and return the OT
    permutation matrix Pi.
    :param y: the ground-truth image.
    :param y_hat: the predicted image.
    :param D: the distance matrix; generate via make_distance_matrix(y.shape[0])
    :return: Pi, the optimal transport matrix. The (i,j) entry in Pi represents the
    cost of moving pixel i in y_hat to pixel j in y.
    """
    np.testing.assert_array_equal(y.shape[0], y.shape[1]) # check images are square
    np.testing.assert_array_equal(y.shape, y_hat.shape) # check images same size
    y_hist = normalize_to_histogram(y)
    y_hat_hist = normalize_to_histogram(y_hat)
    PI = ot.emd(y_hat_hist, y_hist, D)
    return PI
