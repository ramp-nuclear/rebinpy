import numpy as np
import pytest

from binpy import fold


@pytest.mark.parametrize(
    ("y0", "x0", "x1", "y1"),
    [
        ([1, 1, 1], [1, 2, 3, 4], [1, 3, 4], [2, 1]),
        ([1, 1, 1], [1, 2, 3, 4], [1, 2, 4], [1, 2]),
        ([1, 1, 1], [1, 2, 3, 4], [1, 4], [3]),
    ],
)
def test_1d_fold(y0, x0, x1, y1):
    assert np.allclose(fold(y0, x0, x1), y1)
