import numpy as np


def assert_sorted(arr):
    if not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"array must be monotonically increasing")


def simple_slice(arr, inds, axis):
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


def simple_assign(arr, inds, vals, axis):
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    arr[tuple(sl)] = vals


def bin_mids(arr):
    return 0.5 * (arr[1:] + arr[:-1])
