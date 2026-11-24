"""Tests for the beat-by-beat comparison of annotations (bxb).

The expected outputs in tests/target-output/bxb were made with the bxb and
sumstats programs of the WFDB Software Package (10.7.0), from the reference
(.atr) and test (.tst, and sample-data/100.qrs) annotation files there.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wfdb
from wfdb.processing import bxb, bxb_record, sumstats
from wfdb.processing.beat_comparison import LINE_HEADINGS

TARGET = Path(__file__).parent / "target-output" / "bxb"
SAMPLE_DATA = Path(__file__).parent.parent / "sample-data"

# The records compared for the expected outputs, and the options of each
CASES = [
    (SAMPLE_DATA / "100", "qrs", {}),
    (TARGET / "bxb360", "tst", {}),
    (TARGET / "bxb250", "tst", {"start": 81000, "stop": 110000, "window": 0.12}),
]


def results():
    return [bxb_record(str(path), "atr", test, **kw) for path, test, kw in CASES]


@pytest.mark.parametrize(
    ("case", "expected"),
    [(0, "100-report.txt"), (1, "bxb360-report.txt"), (2, "bxb250-report.txt")],
    ids=["100-qrs", "vf-and-shutdowns", "250hz-with-options"],
)
def test_report_matches_bxb(case, expected):
    """The report is the same as that of bxb -S."""
    path, test, kw = CASES[case]
    result = bxb_record(str(path), "atr", test, **kw)
    assert result.report() == (TARGET / expected).read_text()


def test_line_format_matches_bxb():
    """The line-format reports are the same as those of bxb -L."""
    lines = [r.lines() for r in results()]
    assert LINE_HEADINGS[0] + "".join(a + "\n" for a, b in lines) == (
        (TARGET / "bxb.out").read_text()
    )
    assert LINE_HEADINGS[1] + "".join(b + "\n" for a, b in lines) == (
        (TARGET / "sd.out").read_text()
    )


def test_sumstats_matches_wfdb_sumstats():
    summary, shutdown = sumstats(results())
    assert summary == (TARGET / "sumstats-bxb.txt").read_text()
    assert shutdown == (TARGET / "sumstats-sd.txt").read_text()


def test_matrix_and_statistics():
    result = bxb_record(str(SAMPLE_DATA / "100"), "atr", "qrs")

    matrix = result.matrix
    assert list(matrix.index) == list("NSVFQOX")
    assert list(matrix.columns) == list("nsvfqox")
    assert matrix.loc["N", "n"] == 1872
    assert matrix.loc["S", "n"] == 29
    # Test beats with no reference beat have no "no test beat" columns
    assert pd.isna(matrix.loc["O", "o"]) and pd.isna(matrix.loc["X", "x"])

    stats = result.statistics()
    assert stats["QRS sensitivity"] == (1902, 1902)
    assert stats["SVEB sensitivity"] == (0, 29)
    assert stats["VEB positive predictivity"] == (0, 0)
    assert result.rms_rr_error == pytest.approx(1.96, abs=0.005)
    assert result.shutdown_seconds == 0


def test_annotations_with_symbols_only():
    """Annotations read without label_store are compared by their symbols."""
    reference = wfdb.rdann(str(SAMPLE_DATA / "100"), "atr")
    test = wfdb.rdann(str(SAMPLE_DATA / "100"), "qrs")
    assert reference.label_store is None

    result = bxb(reference, test, sig_len=650000)

    expected = bxb_record(str(SAMPLE_DATA / "100"), "atr", "qrs")
    assert result.counts == expected.counts
    assert result.rr_errors == expected.rr_errors


def test_undefined_rr_error():
    """With a single beat, there are no RR intervals to compare."""
    one = wfdb.Annotation("rec", "atr", sample=np.array([200000]), symbol=["N"], fs=360)
    result = bxb(one, one, sig_len=650000)
    assert result.rms_rr_error is None
    assert "RMS RR interval error:      -\n" in result.report()


def test_sampling_frequency_is_needed():
    ann = wfdb.Annotation("rec", "atr", sample=np.array([200000]), symbol=["N"])
    with pytest.raises(ValueError, match="sampling frequency"):
        bxb(ann, ann)
