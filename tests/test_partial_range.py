import numpy as np
import pytest

from rebinpy import fold, rebin


@pytest.mark.parametrize(
    ("x1", "y1"),
    [
        ([1, 2, 3], [2, 3]),
        ([0, 2], [3]),
        ([2, 4], [7]),
    ],
)
def test_fold_partial_range(x1, y1):
    np.testing.assert_allclose(fold([1, 2, 3, 4], [0, 1, 2, 3, 4], x1), y1)


def test_2d_fold_partial_range():
    y0 = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
    np.testing.assert_allclose(fold(y0, [0, 1, 2, 3, 4], [0, 1, 3], axis=1), [[1, 5], [10, 50]])
    np.testing.assert_allclose(fold(y0.T, [0, 1, 2, 3, 4], [0, 1, 3], axis=0), [[1, 10], [5, 50]])


@pytest.mark.parametrize(
    ("x1", "y1"),
    [
        ([1, 2, 3], [2, 3]),
        ([0, 2], [3]),
        ([0.5, 1.5], [1.5]),
        ([0.5, 1.5, 3.5], [1.5, 6]),
    ],
)
def test_rebin_partial_range(x1, y1):
    np.testing.assert_allclose(rebin([1, 2, 3, 4], [0, 1, 2, 3, 4], x1), y1)
