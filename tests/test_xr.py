from typing import List

import numpy as np
import pytest
import xarray

from fastnanquantile import xr


def _assert_indexes_equal(expected: xarray.DataArray, result: xarray.DataArray) -> None:
    """Assert that the indexes of two xarray.DataArray are equal."""
    expected_idxs = [[name, index] for name, index in expected.coords.indexes.items()]
    expected_idxs = sorted(expected_idxs, key=lambda x: x[0])
    result_idxs = [[name, index] for name, index in result.coords.indexes.items()]
    result_idxs = sorted(result_idxs, key=lambda x: x[0])
    assert len(expected_idxs) == len(result_idxs)
    for expected_idx, result_idx in zip(expected_idxs, result_idxs):
        assert expected_idx[0] == result_idx[0]
        assert expected_idx[1].equals(result_idx[1])


def _generate_dataarray(dimensions: List[str], sizes: List[int], nan_fraction: float = 0.1):
    """
    Generate a xarray DataArray with specified dimensions and sizes.

    Args:
        dimensions (list of str): Names of the dimensions.
        sizes (list of int): Sizes of the dimensions.
        nan_fraction (float, optional): Fraction of values to be set to NaN. Defaults to 0.1.

    Returns:
        xarray.DataArray: The generated DataArray.
    """
    # Check if dimensions and sizes have the same length
    if not len(dimensions) == len(sizes):
        raise ValueError("dimensions and sizes must have the same length")
    # Generate random data
    rng = np.random.default_rng(0)  # Create a generator with seed 0
    sample_data = rng.random(sizes)  # Use the generator to create the array
    sample_data[sample_data < nan_fraction] = np.nan
    # Create a DataArray
    return xarray.DataArray(
        sample_data, coords={dim: list(range(size)) for dim, size in zip(dimensions, sizes)}
    )


@pytest.mark.parametrize(
    "q,dim",
    [
        (0.5, "t"),
        (0.5, "x"),
        (0.5, "y"),
        (0.5, ["t", "x"]),
        (0.5, ["t", "x", "y"]),
        ([0.5], "t"),
        ([0.5], "x"),
        ([0.5], "y"),
        ([0.5], ["t", "x"]),
        ([0.5], ["t", "x", "y"]),
        ([0.2, 0.5, 0.3, 0.75], "t"),
        ([0.2, 0.5, 0.3, 0.75], "x"),
        ([0.2, 0.5, 0.3, 0.75], "y"),
        ([0.2, 0.5, 0.3, 0.75], ["t", "x"]),
        ([0.2, 0.5, 0.3, 0.75], ["t", "x", "y"]),
    ],
)
def test_xr_apply_nanquantile_against_xarray(q, dim):
    # Generate sample data
    da = _generate_dataarray(["t", "x", "y"], [50, 100, 100])
    expected = da.quantile(q=q, dim=dim, skipna=True)
    result = xr.xr_apply_nanquantile(da, q=q, dim=dim)
    np.testing.assert_almost_equal(result.values, expected.values, decimal=4)
    _assert_indexes_equal(expected, result)
