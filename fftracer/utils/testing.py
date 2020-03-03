"""
Functions for testing and validating.
"""

import numpy as np

def assert_array_finite(ary: np.ndarray):
    """
    Check whether array contains any null values.
    :param ary: array to check.
    :return: No return value.
    :raises: AssertionError if array contains any null values.
    """
    assert np.isnan(ary).sum() == 0, "nan values detected in array"
    return

def assert_array_nonnegative(ary: np.ndarray):
    """
    Check whether array contains all nonnegative values.
    :param ary: array to check.
    :return: No return value.
    :raises: AssertionError if array contains any values <= 0.
    """
    assert np.all(ary >= 0), "expect nonnegative values; contains {}".format(ary.min())
    return
