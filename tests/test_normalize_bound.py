"""Integer dynamic-range regressions for signal normalization."""

import numpy as np
import pytest

import wfdb
from wfdb import processing


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("bounds", [(-1, 1), (0, 1), (-3, 7)])
def test_signed_integer_full_range(dtype, bounds):
    info = np.iinfo(dtype)
    sig = np.array([info.min + 1, 0, info.max], dtype=dtype)
    original = sig.copy()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = processing.normalize_bound(sig, *bounds)
    np.testing.assert_allclose(result, [bounds[0], sum(bounds) / 2, bounds[1]])
    np.testing.assert_array_equal(sig, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_unsigned_integer_full_range(dtype):
    info = np.iinfo(dtype)
    sig = np.array([0, info.max], dtype=dtype)
    np.testing.assert_allclose(processing.normalize_bound(sig, -1, 1), [-1, 1])


def test_normalize_digital_record(tmp_path):
    digital = np.array([[-30000], [0], [30000]], dtype=np.int16)
    wfdb.wrsamp(
        "normalize",
        fs=250,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=digital,
        fmt=["16"],
        adc_gain=[200],
        baseline=[0],
        write_dir=str(tmp_path),
    )
    record = wfdb.rdrecord(str(tmp_path / "normalize"), physical=False, return_res=16)
    with np.errstate(over="raise"):
        result = processing.normalize_bound(record.d_signal[:, 0], -1, 1)
    np.testing.assert_allclose(result, [-1, 0, 1])


def test_float_signal_retains_formula():
    sig = np.array([-3.5, -0.25, 1.75, 4.5])
    lb, ub = -3, 7
    mid = ub - (ub - lb) / 2
    lo, hi = np.min(sig), np.max(sig)
    center = hi - (hi - lo) / 2
    coef = (ub - lb) / (hi - lo)
    expected = sig * coef - center * coef + mid
    np.testing.assert_array_equal(processing.normalize_bound(sig, lb, ub), expected)


def test_integer_list_input():
    np.testing.assert_allclose(
        processing.normalize_bound([-30000, 0, 30000], -1, 1), [-1, 0, 1]
    )


@pytest.mark.parametrize(
    "sig",
    [
        np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64),
        np.array([-(2**63), -(2**63) + 1, -(2**63) + 2], dtype=np.int64),
        np.array([2**64 - 3, 2**64 - 2, 2**64 - 1], dtype=np.uint64),
    ],
)
def test_large_integer_offset_preserves_small_differences(sig):
    original = sig.copy()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = processing.normalize_bound(sig, 0, 1)
    np.testing.assert_array_equal(result, [0, 0.5, 1])
    np.testing.assert_array_equal(sig, original)


def test_large_integer_list_offset():
    sig = [2**53, 2**53 + 1, 2**53 + 2]
    np.testing.assert_array_equal(processing.normalize_bound(sig), [0, 0.5, 1])


@pytest.mark.parametrize(
    ("bound_dtype", "lb", "ub"),
    [
        (np.int8, -100, 100),
        (np.int16, -30000, 30000),
        (np.int32, -(2**30), 2**30),
        (np.int64, -(2**62), 2**62),
    ],
)
@pytest.mark.parametrize("signal_dtype", [np.int16, np.float64])
def test_numpy_integer_bounds_do_not_overflow(bound_dtype, lb, ub, signal_dtype):
    sig = np.array([0, 1], dtype=signal_dtype)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = processing.normalize_bound(sig, bound_dtype(lb), bound_dtype(ub))
    np.testing.assert_array_equal(result, [float(lb), float(ub)])
