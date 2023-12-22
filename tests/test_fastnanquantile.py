from typing import List, Optional, Union

import numpy as np
import pytest

from fastnanquantile import fastnanquantile as fnq


def _generate_test_cases():
    options = []
    q_values = [0.1, 0.25, 0.5, 0.75, 0.9, [0.1, 0.25, 0.5, 0.75, 0.9]]
    for q in q_values:
        options.extend(
            [
                ((100,), q, 0),
                ((100,), q, None),
                ((100, 100), q, 0),
                ((100, 100), q, 1),
                ((100, 100), q, [0, 1]),
                ((100, 100), q, None),
                ((100, 100, 100), q, 0),
                ((100, 100, 100), q, 1),
                ((100, 100, 100), q, 2),
                ((100, 100, 100), q, [0, 1]),
                ((100, 100, 100), q, [0, 2]),
                ((100, 100, 100), q, [1, 2]),
                ((100, 100, 100), q, [0, 1, 2]),
                ((100, 100, 100), q, None),
            ]
        )
    return options


@pytest.mark.parametrize("array_shape,q,axis", _generate_test_cases())
def test_nanquantile_against_numpy(
    array_shape: tuple, q: Union[float, List[float]], axis: Optional[Union[int, List[int]]]
) -> None:
    # Generate sample data
    rng = np.random.default_rng(0)  # Create a generator with seed 0
    sample_data = rng.random(array_shape)  # Use the generator to create the array
    sample_data[sample_data > 0.95] = np.nan
    # Check if the result is the same as numpy
    expected = np.nanquantile(sample_data, q=q, axis=axis)
    result = fnq.nanquantile(sample_data, q=q, axis=axis)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_nanquantile_against_numpy_empty_array():
    # Test case: Empty array
    x = np.array([[]])
    q = 0.5
    axis = 0
    expected = np.nan
    result = fnq.nanquantile(x, q, axis)
    np.testing.assert_almost_equal(result, expected)


def test__validate_q():
    # Test case: Valid q values (No exception should be raised)
    q = [0.1, 0.5, 0.9]
    fnq._validate_q(q)
    # Test case: Invalid q values
    q = [-0.1, 0.5, 1.2]
    with pytest.raises(ValueError):
        fnq._validate_q(q)


def test__axis_to_iterable():
    # Test case: axis is already an iterable
    axis = [0, 1, 2]
    expected = axis
    result = fnq._axis_to_iterable(axis)
    assert result == expected

    # Test case: axis is not an iterable
    axis = 0
    expected = [0]
    result = fnq._axis_to_iterable(axis)
    assert result == expected


def test__jit_nanquantile_sorted():
    # Test case: Sorted data with multiple quantile values
    x2d = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    q = (0.25, 0.5, 0.75)
    expected = np.array([[2, 7], [3, 8], [4, 9]])
    result = fnq._jit_nanquantile_sorted(x2d, q)
    np.testing.assert_almost_equal(result, expected, decimal=4)

    # Test case: Sorted data with single quantile value
    x2d = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    q = (0.5,)
    expected = np.array([[3, 8]])
    result = fnq._jit_nanquantile_sorted(x2d, q)
    np.testing.assert_almost_equal(result, expected, decimal=4)

    # Test case: Sorted data with NaN values
    x2d = np.array([[1, 2, np.nan, 4, 5], [6, np.nan, 8, 9, 10]])
    q = (0.25, 0.5, 0.75)
    expected = np.array([[1.75, 7.5], [3, 8.5], [4.25, 9.25]])
    result = fnq._jit_nanquantile_sorted(x2d, q)
    np.testing.assert_almost_equal(result, expected, decimal=4)

    # Test case: Sorted data with duplicate values
    x2d = np.array([[1, 2, 2, 4, 5], [6, 7, 8, 8, 10]])
    q = (0.25, 0.5, 0.75)
    expected = np.array([[2, 7], [2, 8], [4, 8]])
    result = fnq._jit_nanquantile_sorted(x2d, q)
    np.testing.assert_almost_equal(result, expected, decimal=4)
