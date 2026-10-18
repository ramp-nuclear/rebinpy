import numpy as np
import pytest

from rebinpy.integrator import PiecewiseConstantIntegrator


@pytest.mark.parametrize(
    ("x", "integral"),
    [
        ([0, 0.5, 1, 1.5, 2], [0.5, 0.5, 1.5, 1.5]),
        ([0, 1, 2], [1, 3]),
        ([0.25, 0.75], [0.5]),
        ([1.5, 2], [1.5]),
    ],
)
def test_piecewise_constant_integrator(x, integral):
    integrator = PiecewiseConstantIntegrator([1, 3], [0, 1, 2])
    np.testing.assert_allclose(integrator(np.asarray(x, dtype=float)), integral)


def test_piecewise_constant_integrator_shape_mismatch():
    with pytest.raises(ValueError):
        PiecewiseConstantIntegrator([1, 2, 3], [0, 1, 2])
