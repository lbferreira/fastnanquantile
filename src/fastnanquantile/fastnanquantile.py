import math
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from numba import njit


def nanquantile(
    x: np.ndarray, q: Union[float, List[float]], axis: Optional[Union[int, List[int]]] = None
) -> np.ndarray:
    """An optimized version of numpy.nanquantile. It's faster than numpy.nanquantile.
    However, in some cases (depending on the array shape and reduction axis),
    numpy.nanquantile can be faster. But this function is generally much faster.
    Some performance comparisons are shown below.
    If x is a MaskedArray, the behavior of this function is different from numpy.nanquantile.
    In this function, masked values are handled as NaNs and the returned array is a conventional numpy array.
    It's equivalent to pass x.filled(np.nan) to numpy.nanquantile.

    Considering x = np.random.random((50, 500, 500)),
    >>> for axis=0, numpy.nanquantile needs 14000 ms (14s) and this function needs 228 ms (after first run);
    >>> for axis=1, numpy.nanquantile needs 1580 ms (1.58s) and this function needs 344 ms;
    >>> for axis=[1,2], numpy.nanquantile needs 200 ms and this function needs 640 ms.

    Args:
        x (np.ndarray): input array.
        q (Union[float, List[float]]): quantile value(s) to be calculated. It can be a float or a list of floats.
        axis (Optional[Union[int, List[int]]]): axis along which the quantile(s) are calculated. If None, all indices are used. Defaults to None.

    Returns:
        np.ndarray: array with the quantile value(s) calculated.
    """
    # Check input type
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    # Handle empty array to make the output compatible with numpy.nanquantile
    if x.size == 0:
        return np.nanquantile(x, q, axis)
    # Numba doesn't support float16, so it's converted to float32.
    if x.dtype == np.float16:
        x = x.astype(np.float32)
    # If axis is None, use all indices
    if axis is None:
        axis = list(range(len(x.shape)))
    # If axis is not an iterable, convert it to a list
    axis = _axis_to_iterable(axis)
    # int is also considered for the cases q = 0 or q = 1
    if isinstance(q, float) or isinstance(q, int):
        is_q_scalar = True
        q = (q,)
    else:
        is_q_scalar = False
        q = tuple(q)
    # Check if all q values are between 0 and 1
    _validate_q(q)
    # Convert int to float. It may happen for q = 0 or q = 1. Numba requires all values to be of the same type.
    q = tuple(map(float, q))
    # Determine the final shape.
    # It's necessary to handle negative axis (e.g. axis=-1)
    drop_axis_idx = np.arange(len(x.shape))[axis]
    final_shape = [x.shape[idx] for idx in range(len(x.shape)) if idx not in drop_axis_idx]
    # Add the quantile dimension
    final_shape = [len(q)] + final_shape
    # Move reduction axis to the end
    axis_destination = [-1 * i for i in range(1, len(axis) + 1)]
    x = np.moveaxis(x, axis, axis_destination)
    # Reduction axis is kept as the last (second) dimension and the rest is flattened to be compatible with _jit_nanquantile_sorted
    reduction_dim_size = 1
    for axis_value in axis_destination:
        reduction_dim_size *= x.shape[axis_value]
    x = x.reshape(-1, reduction_dim_size)
    # Sort values along the specified axis since the next step requires sorted values
    x = np.sort(x, axis=1)
    # Calculate quantiles
    result = _jit_nanquantile_sorted(x, q)
    # Reshape result to the expected shape
    result = result.reshape(final_shape)
    # If q is a scalar, remove the first dimension. It's done only to keep the same output as numpy.
    if is_q_scalar:
        result = result[0]
    return result


def _validate_q(q: Iterable):
    for q in q:
        if q < 0 or q > 1:
            raise ValueError("Quantile value must be between 0 and 1")


def _axis_to_iterable(axis: Any):
    """Convert axis to a list if it's not an iterable. If it's an iterable, it's returned as it is."""
    try:
        len(axis)
    except TypeError:
        axis = [axis]
    return axis


@njit
def _jit_nanquantile_sorted(x2d: np.ndarray, q: Tuple[float]) -> np.ndarray:
    """An optimized version of numpy.nanquantile. It's generally faster than numpy.nanquantile.
    Numba is used to speed up the calculation. The data in the reduction dimention (second dimension) must be sorted.

    Args:
        x2d (np.ndarray): input array. It must be a 2D array. The second dimension is the reduction axis.
        The data in the second dimension must be sorted.
        q (Tuple[float]): quantile value(s) to be calculated. It must be a tuple.

    Returns:
        np.ndarray: 2D-array with the quantile value(s) calculated. The first dimension
        stores the results for each quantile value.
    """
    # Calculation is based on numpy.percentile implementation. The method "linear" was adopted.
    # Info: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy.quantile
    # Data is assumed to be sorted
    nb_quantiles = len(q)
    out = np.zeros((nb_quantiles, x2d.shape[0]), dtype=x2d.dtype)
    for x2d_idx in range(x2d.shape[0]):
        data = x2d[x2d_idx, :]
        data = data[~np.isnan(data)]

        # Quantile calculation
        n = len(data)
        # Handle cases in which we have zero or one element
        if n == 0:
            out[:, x2d_idx] = np.nan
            continue
        if n == 1:
            out[:, x2d_idx] = data[0]
            continue

        # These alpha and beta values are referring to the method "linear" in numpy.percentile
        alpha = 1
        beta = 1
        for ith_q_idx, ith_q in enumerate(q):
            i_plus_g = ith_q * (n - alpha - beta + 1) + alpha
            i = math.floor(i_plus_g)
            g = i_plus_g - i
            i_weight = 1 - g
            j_weight = g
            i = i - 1  # As python index starts at 0
            # Data is assumed to be sorted
            result = i_weight * data[i] + j_weight * data[i + 1]
            # Assign result
            out[ith_q_idx, x2d_idx] = result
    return out
