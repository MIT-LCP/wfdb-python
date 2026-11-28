"""Typed source-header reads alongside the legacy rdheader API."""

import datetime
import pickle
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import wfdb


def test_single_segment_header_is_typed_and_immutable():
    """A signal header groups calibration and storage fields per channel."""
    header = wfdb.read_header("sample-data/100")

    assert isinstance(header, wfdb.SingleSegmentHeader)
    assert isinstance(header, wfdb.io.SingleSegmentHeader)
    assert (header.record_name, header.n_signals, header.fs, header.n_frames) == (
        "100",
        2,
        360,
        650000,
    )
    assert isinstance(header.signals, tuple)
    assert not hasattr(header, "p_signal")
    first = header.signals[0]
    assert isinstance(first, wfdb.SignalHeader)
    assert (first.name, first.file_name, first.format) == (
        "MLII",
        "100.dat",
        "212",
    )
    assert (first.samples_per_frame, first.adc_gain, first.baseline) == (
        1,
        200.0,
        1024,
    )
    with pytest.raises(FrozenInstanceError):
        header.n_frames = 5
    with pytest.raises(FrozenInstanceError):
        first.name = "changed"
    assert pickle.loads(pickle.dumps(header)) == header


def test_signal_header_keeps_source_skew():
    """The typed header describes file skew even when a read aligns values."""
    header = wfdb.read_header("sample-data/100skew")

    assert [signal.skew for signal in header.signals] == [None, 3]


def test_read_header_needs_no_signal_file(tmp_path):
    """Reading metadata does not open the data file named in the header."""
    path = Path(tmp_path) / "100"
    path.with_suffix(".hea").write_bytes(Path("sample-data/100.hea").read_bytes())

    header = wfdb.read_header(path)

    assert header.n_frames == 650000
    assert header.signals[0].file_name == "100.dat"
    assert not path.with_suffix(".dat").exists()


def test_omitted_optional_signal_fields_remain_none(tmp_path):
    """Keep omitted fields distinct from effective format defaults."""
    path = Path(tmp_path) / "t.hea"
    path.write_text("t 1 100 1\nt.dat 16\n")

    signal = wfdb.read_header(path.with_suffix("")).signals[0]

    assert signal.baseline == 0
    assert signal.adc_zero is None
    assert signal.adc_res is None
    assert signal.init_value is None


def test_multisegment_headers_expose_segment_lines_only(tmp_path):
    """A multi-segment header does not require its segment header files."""
    fixed = wfdb.read_header("sample-data/multi-segment/041s/041s")
    assert isinstance(fixed, wfdb.MultiSegmentHeader)
    assert fixed.layout == "fixed"
    assert fixed.n_segments == 2
    assert isinstance(fixed.segments[0], wfdb.SegmentReference)
    assert isinstance(fixed.segments[0], wfdb.io.SegmentReference)
    assert [(segment.name, segment.n_frames) for segment in fixed.segments] == [
        ("041s01", 1000),
        ("041s02", 1000),
    ]
    first_segment = wfdb.read_header(
        Path("sample-data/multi-segment/041s") / fixed.segments[0].name
    )
    assert isinstance(first_segment, wfdb.SingleSegmentHeader)
    assert first_segment.signals[0].name == "III"
    assert fixed.base_datetime == datetime.datetime(1994, 10, 26, 8, 26, 4)

    source = Path("sample-data/multi-segment/p000878/p000878-2137-10-26-16-57.hea")
    target = Path(tmp_path) / source.name
    target.write_bytes(source.read_bytes())
    variable = wfdb.read_header(target.with_suffix(""))
    assert isinstance(variable, wfdb.MultiSegmentHeader)
    assert variable.layout == "variable"
    assert [(segment.name, segment.n_frames) for segment in variable.segments[:2]] == [
        ("3269321_layout", 0),
        ("~", 3750),
    ]


def test_omitted_length_remains_unknown_in_source_header():
    """A header read does not infer missing length from the signal file."""
    header = wfdb.read_header("sample-data/100-no-len")
    selected = wfdb.rdrecord("sample-data/100-no-len", sampfrom=2, sampto=6)

    assert header.n_frames is None
    assert selected.sig_len == 4


def test_read_header_from_archive(tmp_path):
    """The typed reader accepts the same local archive form as rdheader."""
    path = Path(tmp_path) / "100.wfdb"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write("sample-data/100.hea", "100.hea")

    header = wfdb.read_header(path)

    assert isinstance(header, wfdb.SingleSegmentHeader)
    assert header.signals[0].name == "MLII"


def test_rdheader_keeps_legacy_return_type():
    """Existing callers still receive a mutable Record or MultiRecord."""
    single = wfdb.rdheader("sample-data/100")
    multi = wfdb.rdheader("sample-data/multi-segment/041s/041s")

    assert isinstance(single, wfdb.Record)
    assert isinstance(multi, wfdb.MultiRecord)
    single.sig_name[0] = "changed"
    assert single.sig_name[0] == "changed"
