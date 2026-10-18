import numpy as np
import pytest

from rebinpy import rebin, refine


@pytest.mark.parametrize("x1", [[-1, 2, 4], [0, 2, 5], [-1, 5]])
def test_rebin_x1_outside_x0(x1):
    with pytest.raises(ValueError, match="within the boundaries"):
        rebin([1, 2, 3, 4], [0, 1, 2, 3, 4], x1)


def test_refine_x1_outside_x0():
    with pytest.raises(ValueError, match="within the boundaries"):
        refine([1, 2, 3, 4], [0, 1, 2, 3, 4], [5])


@pytest.mark.parametrize(("y0", "axis"), [([1, 2, 3], 0), (np.ones((4, 3)), 1)])
def test_rebin_shape_mismatch(y0, axis):
    with pytest.raises(ValueError, match="one less than"):
        rebin(y0, [0, 1, 2, 3, 4], [0, 4], axis=axis)
