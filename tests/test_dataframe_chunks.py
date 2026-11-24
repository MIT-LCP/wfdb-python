"""Tests for reading WFDB records as bounded DataFrame chunks."""

import datetime

import numpy as np
import pandas as pd
import pytest

import wfdb
from wfdb.io.archive import create_archive


@pytest.mark.parametrize(
    "base_datetime", [None, datetime.datetime(2024, 6, 1, 12, 30, 1, 123000)]
)
def test_chunks_match_full_dataframe(tmp_path, base_datetime):
    signal = np.arange(23, dtype=np.int16)[:, None]
    wfdb.wrsamp(
        "short",
        fs=62.4725,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=signal,
        fmt=["16"],
        adc_gain=[200],
        baseline=[0],
        base_datetime=base_datetime,
        write_dir=str(tmp_path),
    )
    path = str(tmp_path / "short")

    chunks = list(wfdb.iter_dataframes(path, chunk_size=7))
    assert [len(chunk) for chunk in chunks] == [7, 7, 7, 2]
    pd.testing.assert_frame_equal(pd.concat(chunks), wfdb.rdrecord(path).to_dataframe())


def test_range_and_channel_selection():
    path = "sample-data/100"
    chunks = list(
        wfdb.iter_dataframes(
            path,
            chunk_size=6,
            sampfrom=3,
            sampto=17,
            channel_names=["V5"],
            physical=False,
        )
    )

    expected = wfdb.rdrecord(path, sampto=17, physical=False).to_dataframe()
    pd.testing.assert_frame_equal(pd.concat(chunks), expected.iloc[3:17][["V5"]])


def test_chunks_cross_multi_segment_boundary():
    path = "sample-data/multi-segment/fixed1/v102s"
    chunks = list(
        wfdb.iter_dataframes(path, chunk_size=7, sampfrom=74995, sampto=75005)
    )

    assert [len(chunk) for chunk in chunks] == [7, 3]
    expected = wfdb.rdrecord(path, sampto=75005).to_dataframe().iloc[74995:75005]
    pd.testing.assert_frame_equal(pd.concat(chunks), expected)


def test_chunks_cross_variable_layout_boundary():
    path = "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57"
    chunks = list(wfdb.iter_dataframes(path, 7, sampfrom=3745, sampto=3760))

    expected = wfdb.rdrecord(path, sampto=3760).to_dataframe().iloc[3745:3760]
    pd.testing.assert_frame_equal(pd.concat(chunks), expected)


def test_missing_header_length_is_inferred():
    path = "sample-data/drive02-no-len"
    chunks = list(wfdb.iter_dataframes(path, chunk_size=7, sampto=20))

    expected = wfdb.rdrecord(path, sampto=20).to_dataframe()
    pd.testing.assert_frame_equal(pd.concat(chunks), expected)


def test_archive_chunks(tmp_path):
    signal = np.arange(10, dtype=np.int16)[:, None]
    wfdb.wrsamp(
        "archived",
        fs=100,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=signal,
        fmt=["16"],
        adc_gain=[200],
        baseline=[0],
        write_dir=str(tmp_path),
    )
    archive_path = tmp_path / "archived.wfdb"
    create_archive([tmp_path / "archived.hea", tmp_path / "archived.dat"], archive_path)

    chunks = list(wfdb.iter_dataframes(str(archive_path), chunk_size=4))
    expected = wfdb.rdrecord(str(archive_path)).to_dataframe()
    pd.testing.assert_frame_equal(pd.concat(chunks), expected)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_invalid_chunk_size(chunk_size):
    error = TypeError if isinstance(chunk_size, (float, bool)) else ValueError
    with pytest.raises(error):
        next(wfdb.iter_dataframes("sample-data/100", chunk_size))


def test_read_is_lazy(monkeypatch):
    from wfdb.io import record

    calls = []
    read_record = record.rdrecord

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return read_record(*args, **kwargs)

    monkeypatch.setattr(record, "rdrecord", spy)
    chunks = wfdb.iter_dataframes("sample-data/100", chunk_size=5, sampto=12)
    assert calls == []
    assert len(next(chunks)) == 5
    assert len(calls) == 1
    assert len(next(chunks)) == 5
    assert len(calls) == 2
