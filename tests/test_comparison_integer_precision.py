import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from wfdb import processing


@pytest.mark.parametrize(
    "dtype,offset",
    [(np.uint32, 0), (np.uint64, 0), (np.uint64, 2**63), (np.int64, 2**60)],
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_integer_annotation_comparison(dtype, offset, delta):
    reference = np.array([offset + 100, offset + 200], dtype=dtype)
    detected = np.array([offset + 100 + delta, offset + 200 + delta], dtype=dtype)
    original_ref, original_test = reference.copy(), detected.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = processing.compare_annotations(reference, detected, 5)

    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    assert_array_equal(result.matching_sample_nums, [0, 1])
    assert_array_equal(reference, original_ref)
    assert_array_equal(detected, original_test)


def test_fractional_annotation_comparison_preserves_distances():
    result = processing.compare_annotations(
        np.array([100.2, 200.2]), np.array([100.8, 200.8]), 0.5
    )
    assert (result.tp, result.fp, result.fn) == (0, 2, 2)


def test_signed_integer_span_does_not_overflow():
    result = processing.compare_annotations(
        np.array([-(2**63)], dtype=np.int64), np.array([2**63 - 1], dtype=np.int64), 5
    )
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


@pytest.mark.parametrize("side", ["reference", "detected"])
@pytest.mark.parametrize("other_length", [1, 2])
def test_sorted_signed_integer_span_is_accepted(side, other_length):
    spanning = np.array([-(2**63), 2**63 - 1], dtype=np.int64)
    other = spanning.copy() if other_length == 2 else spanning[:1].copy()
    reference, detected = (
        (spanning, other) if side == "reference" else (other, spanning)
    )
    result = processing.compare_annotations(reference, detected, 5)
    assert result.tp == other_length
    assert result.fp == len(detected) - other_length
    assert result.fn == len(reference) - other_length


@pytest.mark.parametrize("side", ["reference", "detected"])
@pytest.mark.parametrize("other_length", [1, 2])
def test_descending_unsigned_samples_are_rejected(side, other_length):
    descending = np.array([200, 100], dtype=np.uint64)
    other = np.array([100, 200], dtype=np.uint64)[:other_length]
    reference, detected = (
        (descending, other) if side == "reference" else (other, descending)
    )
    with pytest.raises(ValueError, match="monotonically"):
        processing.compare_annotations(reference, detected, 5)
