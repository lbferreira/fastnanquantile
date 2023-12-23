import copy
from typing import List, Union

import numpy as np
import xarray

from . import fastnanquantile as fnq


def xr_apply_nanquantile(
    data_array: xarray.DataArray,
    dim: Union[str, List[str]],
    q: Union[float, List[float]],
    keep_attrs: bool = True,
) -> xarray.DataArray:
    """Apply a fast nanquantile function to xarray.DataArray. Dask parallelization is supported.
    It produces the same result as xarray.quantile(skipna=True), which is based on np.nanquantile, but it's faster.
    When using dask, the dimension to be reduced must be fully contained in a single chunk.

    Args:
        data_array (xarray.DataArray): xarray.DataArray object to calculate quantile.
        dim (Union[str, List[str]]): Dimension(s) to be reduced.
        q (Union[float, List[float]]): Quantile value(s) to be calculated. It can be a float or a list of floats.
        keep_attrs (bool, optional): If True, the attributes of the input data_array are copied to the output. Defaults to True.

    Returns:
        xarray.DataArray: DataArray with the quantile value(s) calculated.
    """
    # The implementation of this function is based on xarray quantile implementation.
    # in Xarray, if you call the method quantile from an xarray.DataArray, it will be internally
    # converted to a xarray.Dataset and them for each variable inside the dataset the quantile is calculated.
    # Useful links:
    # https://github.com/pydata/xarray/blob/main/xarray/core/dataarray.py#L5015-L5131
    # https://github.com/pydata/xarray/blob/main/xarray/core/dataset.py#L7900
    # https://github.com/pydata/xarray/blob/main/xarray/core/variable.py

    if not isinstance(data_array, xarray.DataArray):
        raise TypeError("Expected data_array to be a xarray.DataArray")
    if isinstance(dim, str):
        dim = [dim]
    is_q_scalar = False
    if isinstance(q, float):
        q = [q]
        is_q_scalar = True

    def _wrapper(npa, **kwargs):
        # move quantile axis to end. required for apply_ufunc
        return np.moveaxis(fnq.nanquantile(npa, **kwargs), 0, -1)

    axis = np.arange(-1, -1 * len(dim) - 1, -1)
    kwargs = {"q": q, "axis": axis}
    # The func nanquantile returns float32 if input is float16 and float64 otherwise
    input_dtype = data_array.dtype
    output_dtype = np.float32 if input_dtype == np.float16 else input_dtype

    result_var = xarray.apply_ufunc(
        _wrapper,
        data_array.variable,
        input_core_dims=[dim],
        exclude_dims=set(dim),
        output_core_dims=[["quantile"]],
        output_dtypes=[output_dtype],
        dask_gufunc_kwargs=dict(output_sizes={"quantile": len(q)}),
        dask="parallelized",
        kwargs=kwargs,
    )

    # transpose is used only to keep the same order used in xarray.quantile
    result_var = result_var.transpose("quantile", ...)
    # The the function _replace_maybe_drop_dims is used to recreate a DataArray without
    # copying the result generated in the previous step (function apply_ufunc). Passing a DataArray directly
    # to the function apply_ufunc is supported but it was noted that when it is done,
    # the input dataarray has the attrs of its coords removed. Copying the input dataarray at the beginning
    # of this function is possible and avoid changing the original dataarray, but it is not efficient.
    result_da = data_array._replace_maybe_drop_dims(result_var)
    result_da = result_da.assign_coords(quantile=q)
    # To replicate the same behavior of xarray, the quantile dimension is removed if q is a scalar
    if is_q_scalar:
        result_da = result_da.squeeze("quantile")
    if keep_attrs:
        result_da.attrs = copy.deepcopy(data_array.attrs)
    return result_da
