import numpy as np

from rebinpy.utils import assert_sorted, simple_slice


def fold(y0, x0, x1, axis=0):
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    assert_sorted(x0)
    assert_sorted(x1)
    if x1[0] < x0[0] or x1[-1] > x0[-1]:
        raise ValueError("boundaries of x1 are greater than x0")

    if np.size(np.setdiff1d(x1, x0)):
        raise ValueError("boundaries of x1 must be in x0")

    idx = np.searchsorted(x0, x1)
    y0 = simple_slice(y0, slice(None, idx[-1]), axis)
    return np.add.reduceat(y0, idx[:-1], axis=axis)
