import numpy as np

from binpy.utils import bin_mids


def _refine(y0, x0, x1, *, axis, integral):
    dims = tuple(i for i in range(y0.ndim) if i != axis)
    idx = np.searchsorted(x0, bin_mids(x1)) - 1
    idx2 = np.searchsorted(x1, x0[:-1])
    isin = np.isin(x1, x0)
    unchanged = isin[1:] & isin[:-1]
    integral /= np.add.reduceat(integral, idx2)[idx]
    integral[unchanged] = 1.0
    integral = np.expand_dims(integral, axis=dims)
    return np.take_along_axis(y0, np.expand_dims(idx, axis=dims), axis) * integral
