import numpy as np
import pytest

from wfdb import processing


@pytest.mark.parametrize(
    "dtype,offset", [(np.int64, 2**60), (np.int64, -(2**60)), (np.uint64, 2**63)]
)
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("shape", [[0, 1, 0], [0, 1, 1, 0], [0, 2, 1, 3, 1, 0]])
def test_large_integer_local_peaks_preserve_small_differences(
    dtype, offset, radius, shape
):
    signal = np.array([offset + value for value in shape], dtype=dtype)
    original = signal.copy()
    expected = []
    index = 0
    while index < len(shape):
        if shape[index] == max(shape[max(0, index - radius) : index + radius + 1]):
            expected.append(index)
            index += max(radius, 1)
        else:
            index += 1

    actual = processing.find_local_peaks(signal, radius)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.int32, np.float64])
def test_local_peaks_small_values_unchanged(dtype):
    signal = np.array([0, 1, 0, 2, 0], dtype=dtype)
    np.testing.assert_array_equal(processing.find_local_peaks(signal, 1), [1, 3])
