"""Keep source metadata separate from selected single-segment records."""

import datetime

import numpy as np
import pytest

import wfdb
from wfdb.io import _header, record


@pytest.mark.parametrize(
    ("date_suffix", "expected_date"),
    [("", None), (" 01/02/2020", datetime.date(2020, 2, 1))],
)
def test_parsed_record_line_preserves_base_time(date_suffix, expected_date):
    """Materializing a typed record line keeps time with or without a date."""
    parsed = _header._parse_header_fields(f"test 0 360 10 12:34:56{date_suffix}\n")
    header = record._record_from_header_fields(parsed)

    assert isinstance(parsed.record_fields, _header._ParsedRecordLine)
    assert parsed.record_fields.base_time == datetime.time(12, 34, 56)
    assert header.base_time == datetime.time(12, 34, 56)
    assert header.base_date == expected_date
    if expected_date is not None:
        assert header.base_datetime == datetime.datetime.combine(
            expected_date, datetime.time(12, 34, 56)
        )


def test_cached_single_segment_reads_keep_source_fields_unchanged():
    """Editing one loaded window does not change another or the cached source."""
    path = "sample-data/100"
    cache = {}

    def read(sampfrom, sampto, channels):
        return record._rdrecord(
            record_name=path,
            sampfrom=sampfrom,
            sampto=sampto,
            channels=channels,
            physical=False,
            pn_dir=None,
            m2s=True,
            smooth_frames=True,
            ignore_skew=False,
            return_res=64,
            force_channels=True,
            channel_names=None,
            warn_empty=False,
            wfdb_archive=None,
            header_cache=cache,
        )

    first = read(2, 6, [1])
    first.sig_name[0] = "changed"
    first.file_name[0] = "changed.dat"
    first.sig_len = 999

    second = read(10, 12, [0])
    expected = wfdb.rdrecord(path, sampfrom=10, sampto=12, channels=[0], physical=False)
    assert second.sig_name == ["MLII"]
    assert second.sig_len == 2
    assert second.n_sig == 1
    assert second.file_name == ["100.dat"]
    np.testing.assert_array_equal(second.d_signal, expected.d_signal)

    source = cache[(path, None)]
    assert isinstance(source, _header._ParsedSingleSegmentHeader)
    assert source.record_fields.sig_len == 650000
    assert [signal.sig_name for signal in source.signals] == ["MLII", "V5"]
    assert [signal.file_name for signal in source.signals] == ["100.dat"] * 2


def test_parsed_signal_lines_keep_skew_with_their_signals():
    """Signals sharing a file can carry different skew values."""
    source = record._read_header_fields("sample-data/100skew", pn_dir=None)

    assert isinstance(source, _header._ParsedSingleSegmentHeader)
    assert [signal.file_name for signal in source.signals] == ["100.dat"] * 2
    assert [signal.skew for signal in source.signals] == [None, 3]
    header = record._record_from_header_fields(source)
    assert header.skew == [None, 3]


def test_parsed_segment_lines_keep_names_with_lengths():
    """Empty and layout segments retain their lengths in parsed headers."""
    path = "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57"
    source = record._read_header_fields(path, pn_dir=None)

    assert isinstance(source, _header._ParsedMultiSegmentHeader)
    assert [(segment.seg_name, segment.seg_len) for segment in source.segments[:3]] == [
        ("3269321_layout", 0),
        ("~", 3750),
        ("3269321_0001", 2000),
    ]

    header = record._record_from_header_fields(source)
    assert header.seg_name[:3] == ["3269321_layout", "~", "3269321_0001"]
    assert header.seg_len[:3] == [0, 3750, 2000]
    header.seg_name[1] = "changed"
    assert source.segments[1].seg_name == "~"
