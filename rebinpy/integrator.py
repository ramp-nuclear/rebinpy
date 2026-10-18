from abc import abstractmethod
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike

from rebinpy.utils import assert_sorted, bin_mids


class Integrator(Protocol):
    breakpoints: Sequence[float]

    @abstractmethod
    def __call__(self, x: Sequence) -> ArrayLike:
        pass


class ConstantIntegrator:
    def __init__(self):
        self.breakpoints = []

    def __call__(self, x: Sequence):
        return np.diff(x)


class InverseIntegrator:
    def __init__(self):
        self.breakpoints = []

    def __call__(self, x: Sequence):
        return np.log(x[1:] / x[:-1])


class PiecewiseConstantIntegrator:
    def __init__(self, /, y, x):
        assert_sorted(x)
        if len(y) + 1 != len(x):
            raise ValueError("shapes of values and breakpoints do not match")

        self.breakpoints = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def __call__(self, x: Sequence):
        mids = bin_mids(x)
        idx = np.searchsorted(self.breakpoints, mids, side="right") - 1
        return np.diff(x) * self._y[idx]
