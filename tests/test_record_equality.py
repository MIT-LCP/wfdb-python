"""Equality behavior for records with loaded signal arrays."""

import numpy as np

import wfdb


def test_record_equality_returns_false_for_different_uniform_signals():
    """A difference in loaded values is an unequal record, not an error."""
    first = wfdb.Record(d_signal=np.array([[1], [2]]))
    second = wfdb.Record(d_signal=np.array([[1], [3]]))

    assert first != second
    assert not first.__eq__(second, verbose=True)


def test_record_equality_preserves_nan_comparison():
    """Corresponding missing physical samples still compare equal."""
    first = wfdb.Record(p_signal=np.array([[1.0], [np.nan]]))
    second = wfdb.Record(p_signal=np.array([[1.0], [np.nan]]))

    assert first == second


def test_record_equality_handles_expanded_signal_mismatch():
    """Expanded arrays with different values or counts compare unequal."""
    first = wfdb.Record(e_d_signal=[np.array([1, 2]), np.array([3])])
    changed = wfdb.Record(e_d_signal=[np.array([1, 2]), np.array([4])])
    missing = wfdb.Record(e_d_signal=[np.array([1, 2])])

    assert first != changed
    assert first != missing


def test_record_equality_with_other_type_is_false():
    """An unrelated object is not a record with equal contents."""
    assert wfdb.Record() != object()
