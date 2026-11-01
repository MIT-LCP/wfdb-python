"""Tests for RR interval extraction from annotation files."""

import numpy as np
import pytest

import wfdb
from wfdb import processing


@pytest.fixture
def mixed_annotations(tmp_path):
    wfdb.wrann(
        "mixed",
        "atr",
        np.array([0, 100, 150, 200, 250, 300, 400]),
        symbol=["+", "N", "+", "V", "+", "N", '"'],
        aux_note=["(N", "", "(AFIB", "", "(N", "", "note"],
        fs=100,
        write_dir=str(tmp_path),
    )
    return str(tmp_path / "mixed")


@pytest.mark.parametrize(
    "format, divisor", [(None, 1), ("s", 100), ("m", 6000), ("h", 360000)]
)
def test_ann2rr_ignores_non_beats(mixed_annotations, format, divisor):
    intervals = processing.ann2rr(mixed_annotations, "atr", format=format)
    np.testing.assert_allclose(intervals, np.array([100, 100, 100]) / divisor)
    assert intervals.dtype.kind == ("i" if format is None else "f")


def test_ann2rr_all_annotations(mixed_annotations):
    intervals = processing.ann2rr(mixed_annotations, "atr", all_annotations=True)
    np.testing.assert_array_equal(intervals, [0, 100, 50, 50, 50, 50, 100])


@pytest.mark.parametrize("symbol", list("NLRaVFJASEj/QB?!enfr"))
def test_ann2rr_beat_types(tmp_path, symbol):
    # Include codes above the gaps in the standard annotation label table.
    wfdb.wrann(
        "beats",
        "atr",
        np.array([100, 200]),
        symbol=[symbol, symbol],
        fs=100,
        write_dir=str(tmp_path),
    )
    intervals = processing.ann2rr(str(tmp_path / "beats"), "atr")
    np.testing.assert_array_equal(intervals, [100, 100])


@pytest.mark.parametrize("symbol", list('~|sT*D"=p^t+u[]@x()'))
def test_ann2rr_no_beats(tmp_path, symbol):
    wfdb.wrann(
        "nonbeats",
        "atr",
        np.array([100, 200]),
        symbol=[symbol, symbol],
        fs=100,
        write_dir=str(tmp_path),
    )
    intervals = processing.ann2rr(str(tmp_path / "nonbeats"), "atr")
    assert intervals.shape == (0,)
    assert intervals.dtype == np.int64


@pytest.mark.parametrize("sample", [0, 100])
def test_ann2rr_single_beat(tmp_path, sample):
    wfdb.wrann(
        "single",
        "atr",
        np.array([sample, sample]),
        symbol=["+", "N"],
        fs=100,
        write_dir=str(tmp_path),
    )
    intervals = processing.ann2rr(str(tmp_path / "single"), "atr")
    # The first interval remains measured from time zero, not the first marker.
    np.testing.assert_array_equal(intervals, [sample])


def test_ann2rr_custom_symbol(tmp_path):
    wfdb.wrann(
        "custom",
        "atr",
        np.array([100, 200]),
        symbol=["Z", "Z"],
        custom_labels=[(1, "Z", "Renamed normal beat")],
        fs=100,
        write_dir=str(tmp_path),
    )
    intervals = processing.ann2rr(str(tmp_path / "custom"), "atr")
    np.testing.assert_array_equal(intervals, [100, 100])


def test_ann2rr_print(mixed_annotations, capsys):
    assert processing.ann2rr(mixed_annotations, "atr", as_array=False) is None
    assert capsys.readouterr().out == "100\n100\n100\n"


def test_ann2rr_empty_annotation(monkeypatch):
    ann = wfdb.Annotation(
        record_name="empty",
        extension="atr",
        fs=100,
        sample=np.array([], dtype=np.int64),
        label_store=np.array([], dtype=int),
    )
    monkeypatch.setattr(processing.hr, "rdann", lambda *args, **kwargs: ann)
    for all_annotations in [False, True]:
        intervals = processing.ann2rr("empty", "atr", all_annotations=all_annotations)
        assert intervals.shape == (0,)


def test_ann2rr_sample_record():
    intervals = processing.ann2rr("sample-data/100", "atr")
    np.testing.assert_array_equal(intervals[:3], [77, 293, 292])
    assert len(intervals) == 2273


@pytest.mark.parametrize(
    "start_time, stop_time, expected",
    [
        (None, None, [100, 100, 100]),
        (1.5, None, [50, 100]),
        (None, 2.5, [100, 100]),
        (1.5, 3.0, [50, 100]),
        (2.0, 3.0, [0, 100]),
        (3.5, None, []),
        # Half a sample rounds up, as in WFDB.
        (0.125, None, [87, 100, 100]),
    ],
)
def test_ann2rr_time_range(mixed_annotations, start_time, stop_time, expected):
    # Beats are at 1, 2 and 3 seconds into the record.
    intervals = processing.ann2rr(
        mixed_annotations, "atr", start_time=start_time, stop_time=stop_time
    )
    np.testing.assert_array_equal(intervals, expected)


def test_ann2rr_start_after_stop(mixed_annotations):
    with pytest.raises(ValueError, match="start_time"):
        processing.ann2rr(mixed_annotations, "atr", start_time=2, stop_time=1)
