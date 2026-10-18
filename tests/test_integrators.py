import numpy as np
import pytest

from rebinpy import refine
from rebinpy.integrator import InverseIntegrator


@pytest.fixture
def inverse_integrator():
    return InverseIntegrator()


@pytest.mark.parametrize(
    ("y0", "x0", "x1", "y1"),
    [
        ([1, 1, 1], [1, 2, 3, 4], [1.5], [0.5, 0.5, 1, 1]),
        ([1, 1, 1], [1, 2, 3, 4], [2.5], [1, 0.5, 0.5, 1]),
        ([1, 1, 1], [1, 2, 3, 4], [3.5], [1, 1, 0.5, 0.5]),
        ([1, 1, 1], [1, 2, 3, 4], [1.5, 2.5, 3.5], [0.5] * 6),
        ([1, 1, 1], [10, 20, 30, 40], [15], [0.5, 0.5, 1, 1]),
        ([1, 1, 1], [10, 20, 30, 40], [25], [1, 0.5, 0.5, 1]),
        ([1, 1, 1], [10, 20, 30, 40], [35], [1, 1, 0.5, 0.5]),
        ([1, 1, 1], [10, 20, 30, 40], [15, 25, 35], [0.5] * 6),
    ],
)
def test_inverse_integrator(y0, x0, x1, y1):
    np.testing.assert_allclose(refine(y0, np.exp(x0), np.exp(x1), integrator=InverseIntegrator()), y1)
