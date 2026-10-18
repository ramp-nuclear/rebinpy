import numpy as np
import pytest

from rebinpy import rebin


class StepIntegrator:
    """Weight 1 below 1.2 and 3 above it, exact only on grids containing the step."""

    def __init__(self, breakpoints):
        self.breakpoints = breakpoints

    def __call__(self, x):
        mids = 0.5 * (x[1:] + x[:-1])
        return np.diff(x) * np.where(mids < 1.2, 1.0, 3.0)


@pytest.mark.parametrize("breakpoints", [[1.2], [-1.0, 1.2, 5.0]])
@pytest.mark.parametrize(
    ("x1", "y1"),
    [
        ([0, 0.6, 2], [1 / 6, 5 / 6]),
        ([0, 1.8, 2], [5 / 6, 1 / 6]),
        ([0, 1.2, 2], [1 / 3, 2 / 3]),
    ],
)
def test_rebin_uses_integrator_breakpoints(breakpoints, x1, y1):
    np.testing.assert_allclose(rebin([1], [0, 2], x1, integrator=StepIntegrator(breakpoints)), y1)
