import numpy as np
import pytest

from binpy import rebin


@pytest.mark.parametrize(
    ("y0", "x0", "x1", "y1"),
    [
        ([1, 1, 1], [1, 2, 3, 4], [1.5], [0.5, 0.5, 1, 1]),
        ([1, 1, 1], [1, 2, 3, 4], [2.5], [1, 0.5, 0.5, 1]),
        ([1, 1, 1], [1, 2, 3, 4], [3.5], [1, 1, 0.5, 0.5]),
        ([1, 1, 1], [1, 2, 3, 4], [1.5, 2.5, 3.5], [0.5]),
    ],
)
def test_1d_rebin(y0, x0, x1, y1):
    assert np.allclose(rebin(y0, x0, np.union1d(x0, x1)), y1)
