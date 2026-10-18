import numpy as np
import pytest

from rebinpy import rebin


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_nd_rebin_axis(axis):
    x0 = [0, 1, 2, 3, 4]
    x1 = [0, 1.5, 4]
    y0 = np.moveaxis(np.arange(1, 5) * np.ones((2, 3, 1)), -1, axis)
    expected = np.moveaxis(np.array([2, 8]) * np.ones((2, 3, 1)), -1, axis)
    np.testing.assert_allclose(rebin(y0, x0, x1, axis=axis), expected)


@pytest.mark.parametrize("axis", [2, -3])
def test_rebin_axis_out_of_bounds(axis):
    with pytest.raises(ValueError):
        rebin(np.ones((2, 4)), [0, 1, 2, 3, 4], [0, 2, 4], axis=axis)
