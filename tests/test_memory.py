from pathlib import Path

import fsspec
import numpy as np
import pytest

import wfdb

SAMPLE_DATA = Path("sample-data")


@pytest.fixture
def memory_fs():
    """Provide the in-memory filesystem and remove the test files afterwards."""
    fs = fsspec.filesystem("memory")
    yield fs
    if fs.exists("/wfdb-test"):
        fs.rm("/wfdb-test", recursive=True)


def upload(fs, pattern):
    """Copy the sample files that match a pattern to the in-memory filesystem."""
    for path in SAMPLE_DATA.glob(pattern):
        if path.is_file():
            name = path.relative_to(SAMPLE_DATA).as_posix()
            fs.pipe(f"/wfdb-test/{name}", path.read_bytes())


@pytest.mark.parametrize(
    ("record_name", "pattern", "kwargs"),
    [
        pytest.param("100", "100.*", {"sampto": 1000}, id="format-212"),
        pytest.param("100", "100.*", {"sampfrom": 500, "sampto": 1000}, id="sampfrom"),
        pytest.param("flacformats", "flacformats.*", {}, id="flac"),
        pytest.param("s0010_re", "s0010_re.*", {"sampto": 100}, id="multiple-dat"),
        pytest.param(
            "multi-segment/fixed1/v102s",
            "multi-segment/fixed1/*",
            {"sampto": 1000},
            id="multi-segment-fixed",
        ),
        pytest.param(
            "multi-segment/041s/041s",
            "multi-segment/041s/*",
            {"sampto": 1000},
            id="multi-segment-variable",
        ),
    ],
)
def test_rdrecord(memory_fs, record_name, pattern, kwargs):
    upload(memory_fs, pattern)

    record = wfdb.rdrecord(f"memory://wfdb-test/{record_name}", **kwargs)

    expected = wfdb.rdrecord(str(SAMPLE_DATA / record_name), **kwargs)
    assert record.sig_name == expected.sig_name
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_rdheader(memory_fs):
    upload(memory_fs, "100.*")

    header = wfdb.rdheader("memory://wfdb-test/100")

    assert header.sig_name == wfdb.rdheader("sample-data/100").sig_name


def test_rdann(memory_fs):
    upload(memory_fs, "100.*")

    annotation = wfdb.rdann("memory://wfdb-test/100", "atr")

    expected = wfdb.rdann("sample-data/100", "atr")
    np.testing.assert_array_equal(annotation.sample, expected.sample)
    assert annotation.symbol == expected.symbol


@pytest.mark.parametrize("uri", ["memory://100", "memory:///100"])
def test_record_without_directory(uri):
    fs = fsspec.filesystem("memory")
    names = ["100.hea", "100.dat"]
    for name in names:
        fs.pipe(f"/{name}", (SAMPLE_DATA / name).read_bytes())

    try:
        header = wfdb.rdheader(uri)
        record = wfdb.rdrecord(uri, sampto=1000)
    finally:
        for name in names:
            fs.rm(f"/{name}")

    expected = wfdb.rdrecord("sample-data/100", sampto=1000)
    assert header.sig_name == expected.sig_name
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_missing_file(memory_fs):
    upload(memory_fs, "100.*")

    with pytest.raises(FileNotFoundError):
        wfdb.rdrecord("memory://wfdb-test/nonexistent")
