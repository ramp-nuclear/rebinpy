import numpy as np

from binpy.utils import assert_sorted, bin_mids


def fold(y0, x0, x1, axis=0):
    x0 = np.asfarray(x0)
    y0 = np.asfarray(y0)
    x1 = np.asfarray(x1)
    assert_sorted(x0)
    assert_sorted(x1)
    if x1[0] < x0[0] or x1[-1] > x0[-1]:
        raise ValueError("boundaries of x1 are greater than x0")

    if np.size(np.setdiff1d(x1, x0)):
        raise ValueError("boundaries of x1 must be in x0")

    idx = np.searchsorted(x0, x1)
    return np.add.reduceat(y0, idx[:-1], axis=axis)
