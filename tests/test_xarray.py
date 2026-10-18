import numpy as np
import pytest
import xarray

from rebinpy.xarray import rebin


@pytest.fixture
def lower_edges():
    return xarray.DataArray([1.0, 2.0, 3.0, 4.0], dims=["e"], coords={"e": [0.0, 1.0, 2.0, 3.0]})


def test_rebin_append(lower_edges):
    result = rebin(lower_edges, np.array([0.0, 2.0, 4.0]), axis="e", append=4.0)
    np.testing.assert_allclose(result.values, [3, 7])
    np.testing.assert_allclose(result["e"].values, [0, 2])


def test_rebin_prepend(lower_edges):
    upper_edges = lower_edges.assign_coords(e=[1.0, 2.0, 3.0, 4.0])
    result = rebin(upper_edges, np.array([0.0, 2.0, 4.0]), axis="e", prepend=0.0)
    np.testing.assert_allclose(result.values, [3, 7])
    np.testing.assert_allclose(result["e"].values, [2, 4])


@pytest.mark.parametrize("axis", ["e", 1, -1])
def test_rebin_2d(axis):
    xarr = xarray.DataArray(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        dims=["t", "e"],
        coords={"t": [0, 1], "e": [0.0, 1.0, 2.0, 3.0]},
    )
    result = rebin(xarr, np.array([0.0, 1.5, 4.0]), axis=axis, append=4.0)
    assert result.dims == ("t", "e")
    np.testing.assert_allclose(result.values, [[2, 8], [20, 80]])
    np.testing.assert_allclose(result["t"].values, [0, 1])
    np.testing.assert_allclose(result["e"].values, [0, 1.5])


@pytest.mark.parametrize(("prepend", "append"), [(None, None), (0.0, 4.0)])
def test_rebin_requires_exactly_one_edge(lower_edges, prepend, append):
    with pytest.raises(ValueError):
        rebin(lower_edges, np.array([0.0, 4.0]), axis="e", prepend=prepend, append=append)
