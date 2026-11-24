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


def test_wrsamp_to_cloud(memory_fs):
    signal = np.array([[0], [1], [-1], [2]], dtype=np.int16)
    wfdb.wrsamp(
        "cloud_record",
        fs=100,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=signal,
        fmt=["16"],
        adc_gain=[200],
        baseline=[0],
        write_dir="memory://wfdb-test/subdir",
    )

    assert memory_fs.exists("/wfdb-test/subdir/cloud_record.hea")
    assert memory_fs.exists("/wfdb-test/subdir/cloud_record.dat")
    result = wfdb.rdrecord("memory://wfdb-test/subdir/cloud_record", physical=False)
    np.testing.assert_array_equal(result.d_signal, signal)


def test_record_wrsamp_to_cloud(memory_fs):
    record = wfdb.rdrecord("sample-data/100", sampto=100, physical=False)
    record.record_name = "cloud_copy"
    record.file_name = ["cloud_copy.dat"] * record.n_sig
    record.wrsamp(write_dir="memory://wfdb-test")

    result = wfdb.rdrecord("memory://wfdb-test/cloud_copy", physical=False)
    np.testing.assert_array_equal(result.d_signal, record.d_signal)


def test_multi_record_wrsamp_to_cloud(memory_fs):
    source = "sample-data/multi-segment/fixed1/v102s"
    record = wfdb.rdrecord(source, sampto=1000, m2s=False, physical=False)
    record.wrsamp(write_dir="memory://wfdb-test/multi")

    assert memory_fs.exists("/wfdb-test/multi/v102s.hea")
    assert memory_fs.exists("/wfdb-test/multi/v102s_1.hea")
    result = wfdb.rdrecord("memory://wfdb-test/multi/v102s", physical=False)
    expected = wfdb.rdrecord(source, sampto=1000, physical=False)
    np.testing.assert_array_equal(result.d_signal, expected.d_signal)


def test_wrann_to_cloud(memory_fs):
    annotation = wfdb.rdann("sample-data/100", "atr")
    annotation.record_name = "cloud_annotation"
    annotation.wrann(write_fs=True, write_dir="memory://wfdb-test/annotations")

    result = wfdb.rdann("memory://wfdb-test/annotations/cloud_annotation", "atr")
    np.testing.assert_array_equal(result.sample, annotation.sample)
    assert result.symbol == annotation.symbol
    assert result.fs == annotation.fs


def test_wrann_gateway_to_cloud(memory_fs):
    wfdb.wrann(
        "gateway",
        "atr",
        np.array([10, 20]),
        symbol=["N", "V"],
        write_dir="memory://wfdb-test",
    )

    result = wfdb.rdann("memory://wfdb-test/gateway", "atr")
    assert result.sample.tolist() == [10, 20]
    assert result.symbol == ["N", "V"]


def test_cloud_write_validation_does_not_publish(memory_fs):
    record = wfdb.rdrecord("sample-data/100", sampto=10, physical=False)
    record.record_name = "invalid"
    record.fmt = ["invalid"] * record.n_sig

    with pytest.raises((ValueError, TypeError)):
        record.wrsamp(write_dir="memory://wfdb-test")

    assert not memory_fs.exists("/wfdb-test/invalid.hea")


def test_cloud_write_warns_when_replacing_another_records_signal(memory_fs):
    record = wfdb.rdrecord("sample-data/100", sampto=10, physical=False)
    record.record_name = "copy"
    memory_fs.pipe("/wfdb-test/100.dat", b"original")

    with pytest.warns(UserWarning, match="replaces existing signal files"):
        record.wrsamp(write_dir="memory://wfdb-test")


def test_physionet_write_is_rejected():
    with pytest.raises(ValueError, match="physionet://"):
        wfdb.wrann(
            "test",
            "atr",
            np.array([1]),
            symbol=["N"],
            write_dir="physionet://mitdb",
        )


def test_cloud_archive_write_is_rejected(memory_fs):
    record = wfdb.rdrecord("sample-data/100", sampto=10, physical=False)
    with pytest.raises(ValueError, match="archives to cloud storage"):
        record.wrsamp(wfdb_archive="memory://wfdb-test/100.wfdb")
