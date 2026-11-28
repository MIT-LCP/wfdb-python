import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import wfdb
from wfdb import processing


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16]
)
@pytest.mark.parametrize("direction", ["up", "down", "both", "compare"])
def test_correct_peaks_integer_indices(dtype, direction):
    signal = np.zeros(100)
    signal[[1, 70]] = -5 if direction == "down" else 5
    guesses = np.array([72, 2, 2], dtype=dtype)
    original = guesses.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = processing.correct_peaks(signal, guesses, 5, 9, direction)
    assert_array_equal(actual, [70, 1, 1])
    assert_array_equal(guesses, original)
    assert actual.dtype.kind in "iu"
    assert_array_equal(signal[actual], signal[[70, 1, 1]])


@pytest.mark.parametrize(
    "dtype,center,length", [(np.int8, 125, 128), (np.uint8, 253, 256)]
)
def test_correct_peaks_narrow_integer_right_boundary(dtype, center, length):
    signal = np.zeros(length)
    signal[-1] = 5
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = processing.correct_peaks(
            signal, np.array([center], dtype=dtype), 5, 9, "up"
        )
    assert_array_equal(actual, [length - 1])


def test_correct_peaks_uint64_remains_usable_as_index():
    signal = np.zeros(100)
    signal[70] = 5
    actual = processing.correct_peaks(
        signal, np.array([72], dtype=np.uint64), 5, 9, "up"
    )
    assert_array_equal(actual, [70])
    assert_array_equal(signal[actual], [5])


def test_correct_peaks_unsigned_public_record(tmp_path):
    wfdb.wrsamp(
        "indices",
        fs=10,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=np.array([0, 5] + [0] * 98, dtype=np.int64).reshape(-1, 1),
        fmt=["16"],
        adc_gain=[1],
        baseline=[0],
        write_dir=str(tmp_path),
    )
    signal, _ = wfdb.rdsamp(str(tmp_path / "indices"))
    assert_array_equal(
        processing.correct_peaks(
            signal[:, 0], np.array([2], dtype=np.uint32), 5, 9, "up"
        ),
        [1],
    )


def test_correct_peaks_signed_control():
    signal = np.zeros(100)
    signal[[1, 70]] = 5
    assert_array_equal(
        processing.correct_peaks(signal, np.array([72, 2], dtype=np.int64), 5, 9, "up"),
        [70, 1],
    )


def test_correct_peaks_python_integer_list():
    signal = np.zeros(100)
    signal[[1, 70]] = 5
    assert_array_equal(processing.correct_peaks(signal, [72, 2], 5, 9, "up"), [70, 1])
