from functools import reduce

import numpy as np

from binpy import fold
from binpy.integrator import ConstantIntegrator, Integrator
from binpy.refine import _refine
from binpy.utils import assert_sorted


def rebin(y0: np.array, x0: np.array, x1: np.array, integrator: Integrator = ConstantIntegrator(),
          axis=0) -> np.array:
    """
    Parameters
    ----------
    y0: np.array
     The values inside the original bins.
    x0: np.array
     The sorted boundaries of the original bins, one-dimensional array, whose length is the length of y plus 1.
    x1: np.array
     The sorted boundaries of the new bins. The end and start should be the same as x0.
    integrator: Integrator
     Integrator to compute the integral inside a bin.
    axis: int
     The axis of y over which to rebin.

    Returns
    -------
    np.array

    Examples
    --------
    >>> y = np.arange(0, 8)
    >>> x0 = np.arange(0, 9)
    >>> x1 = np.array([0, 4, 6, 8])
    >>> rebin(y,x0,x1)
    array([ 6.,  9., 13.])

    """
    x0 = np.asfarray(x0)
    y0 = np.asfarray(y0)
    x1 = np.asfarray(x1)

    assert_sorted(x0)
    assert_sorted(x1)

    xb = np.asfarray(integrator.breakpoints)
    xb = [(x0[0] <= xb) & (xb <= x0[1])]
    xn = reduce(np.union1d, [x0, x1, xb])
    y = _refine(y0, x0, xn, axis=axis, integral=integrator(xn))
    return fold(y, xn, x1, axis=axis)


def refine(y0, x0, x1, integrator=ConstantIntegrator(), axis=0):
    assert_sorted(x1)
    return rebin(y0, x0, np.union1d(x1, x0), integrator=integrator, axis=axis)
