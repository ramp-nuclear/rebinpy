import numpy as np
import xarray

import binpy
from binpy.integrator import ConstantIntegrator


def rebin(
    xarr, x1, integrator=ConstantIntegrator(), *, axis=0, prepend=None, append=None
):
    dims = list(xarr.dims)
    naxis = dims.index(axis) if axis in dims else range(len(dims))[axis]
    axis = dims[axis] if axis not in dims else axis
    coords = dict(xarr.coords)
    if prepend is not None and append is None:
        x0 = np.hstack((prepend, xarr[axis].values))
        coords[axis] = x1[1:]
    elif prepend is None and append is not None:
        x0 = np.hstack((xarr[axis].values, append))
        coords[axis] = x1[:-1]
    else:
        raise ValueError("either prepend or append must be defined")
    y = binpy.rebin(xarr.values, x0, x1, integrator=integrator, axis=naxis)
    return xarray.DataArray(data=y, dims=dims, coords=coords)
