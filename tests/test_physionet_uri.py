"""Tests for reading records and annotations with physionet:// URIs."""

import io
import shutil
from pathlib import Path

import fsspec
import numpy as np
import pytest
import requests

import wfdb

SAMPLE_DATA = Path("sample-data")


@pytest.fixture
def physionet_cache(tmp_path, monkeypatch):
    """A physionet package cache, with a fresh filesystem for each test."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("PHYSIONET_CACHE", str(cache))
    fsspec.filesystem("physionet").clear_instance_cache()
    yield cache
    fsspec.filesystem("physionet").clear_instance_cache()


@pytest.fixture
def offline(monkeypatch):
    """Fail any request that is made."""

    def refuse(session, request, **kwargs):
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    monkeypatch.setattr(requests.Session, "send", refuse)


def cache_sample_files(cache, project, names, folder=SAMPLE_DATA):
    """Copy sample files into the cache, as if physionet had downloaded them."""
    target = cache / "physionet.org" / project / "1.0"
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy(folder / name, target / name)


@pytest.mark.parametrize(
    ("record_name", "names", "kwargs"),
    [
        ("100", ["100.hea", "100.dat"], {"sampfrom": 1000, "sampto": 5000}),
        (
            "flacformats",
            ["flacformats.hea", "flacformats.d0", "flacformats.d1", "flacformats.d2"],
            {},
        ),
        (
            "041s",
            ["041s.hea", "041s01.hea", "041s01.dat", "041s02.hea", "041s02.dat"],
            {"sampto": 1000},
        ),
    ],
    ids=["format-212", "flac", "multi-segment"],
)
def test_rdrecord_reads_cached_files_without_requests(
    physionet_cache, offline, record_name, names, kwargs
):
    """Files already in the physionet cache are read without any requests.

    That is only so with stream=True: otherwise the physionet package first gets
    the project's checksum list, which is not cached here.
    """
    folder = (
        SAMPLE_DATA / "multi-segment" / "041s" if record_name == "041s" else SAMPLE_DATA
    )
    cache_sample_files(physionet_cache, "demo", names, folder)

    record = wfdb.rdrecord(f"physionet://demo/1.0/{record_name}", **kwargs)

    expected = wfdb.rdrecord(str(folder / record_name), **kwargs)
    assert record.sig_name == expected.sig_name
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_rdheader_and_rdann(physionet_cache, offline):
    cache_sample_files(physionet_cache, "demo", ["100.hea", "100.atr"])

    header = wfdb.rdheader("physionet://demo/1.0/100")
    annotation = wfdb.rdann("physionet://demo/1.0/100", "atr")

    assert header.sig_name == wfdb.rdheader("sample-data/100").sig_name
    expected = wfdb.rdann("sample-data/100", "atr")
    np.testing.assert_array_equal(annotation.sample, expected.sample)
    assert annotation.symbol == expected.symbol


def test_files_not_in_the_cache_are_requested(physionet_cache, offline):
    cache_sample_files(physionet_cache, "demo", ["100.hea"])
    # 100.dat is not cached, so it is requested from PhysioNet
    with pytest.raises(AssertionError, match="Unexpected request"):
        wfdb.rdrecord("physionet://demo/1.0/100")


@pytest.mark.network
def test_rdrecord_from_physionet(physionet_cache, capsys):
    """Read part of a record from PhysioNet, without downloading whole files."""
    record = wfdb.rdrecord("physionet://mitdb/1.0.0/100", sampfrom=1000, sampto=4600)
    annotation = wfdb.rdann("physionet://mitdb/1.0.0/100", "atr", sampto=4600)

    expected = wfdb.rdrecord("sample-data/100", sampfrom=1000, sampto=4600)
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)
    np.testing.assert_array_equal(
        annotation.sample, wfdb.rdann("sample-data/100", "atr", sampto=4600).sample
    )
    # Nothing is downloaded into the cache, and no progress bars are shown
    assert not any(
        path.suffix in {".dat", ".hea"} for path in physionet_cache.rglob("*")
    )
    assert "%|" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("record_name", "names", "kwargs"),
    [
        ("100", ["100.hea", "100.dat"], {"sampfrom": 1000, "sampto": 5000}),
        # The signal length is worked out from the size of the signal file
        ("100-no-len", ["100-no-len.hea", "100.dat"], {}),
        (
            "flacformats",
            ["flacformats.hea", "flacformats.d0", "flacformats.d1", "flacformats.d2"],
            {},
        ),
        (
            "041s",
            ["041s.hea", "041s01.hea", "041s01.dat", "041s02.hea", "041s02.dat"],
            {"sampto": 1000},
        ),
    ],
    ids=["format-212", "no-length", "flac", "multi-segment"],
)
def test_pn_dir_reads_through_physionet(
    physionet_cache, offline, record_name, names, kwargs
):
    """pn_dir reads go through physionet://, so cached files need no requests."""
    folder = (
        SAMPLE_DATA / "multi-segment" / "041s" if record_name == "041s" else SAMPLE_DATA
    )
    cache_sample_files(physionet_cache, "demo", names, folder)

    record = wfdb.rdrecord(record_name, pn_dir="demo/1.0", **kwargs)

    expected = wfdb.rdrecord(str(folder / record_name), **kwargs)
    assert record.sig_name == expected.sig_name
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_pn_dir_rdheader_and_rdann(physionet_cache, offline):
    cache_sample_files(physionet_cache, "demo", ["100.hea", "100.atr"])

    header = wfdb.rdheader("100", pn_dir="demo/1.0")
    annotation = wfdb.rdann("100", "atr", pn_dir="demo/1.0")

    assert header.sig_name == wfdb.rdheader("sample-data/100").sig_name
    expected = wfdb.rdann("sample-data/100", "atr")
    np.testing.assert_array_equal(annotation.sample, expected.sample)


@pytest.fixture
def versions(monkeypatch):
    """Replace physionet.latest_version, recording calls, and clear the cache."""
    import physionet

    calls = []
    results = {}

    def latest_version(db_dir):
        calls.append(db_dir)
        result = results.get(db_dir, "1.0.0")
        if isinstance(result, list):
            result = result.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(physionet, "latest_version", latest_version)
    monkeypatch.setattr(wfdb.io.download.time, "sleep", lambda seconds: None)
    wfdb.io.download._latest_version.cache_clear()
    yield calls, results
    wfdb.io.download._latest_version.cache_clear()


def api_error(status):
    from physionet.api.exceptions import PhysioNetAPIError

    response = requests.Response()
    response.status_code = status
    return PhysioNetAPIError(f"HTTP {status}", response=response)


def test_get_version_is_looked_up_once_per_database(versions):
    calls, _ = versions
    assert wfdb.io.download.get_version("mitdb") == "1.0.0"
    assert wfdb.io.download.get_version("mitdb/x") == "1.0.0"
    assert wfdb.io.download.get_version("afdb") == "1.0.0"
    assert calls == ["mitdb", "afdb"]


def test_get_version_retries_while_physionet_is_unavailable(versions):
    calls, results = versions
    results["mitdb"] = [api_error(502), ConnectionError("unreachable"), "1.0.0"]
    assert wfdb.io.download.get_version("mitdb") == "1.0.0"
    assert calls == ["mitdb"] * 3


def test_get_version_errors(versions):
    from physionet.api.exceptions import NotFoundError, PhysioNetAPIError

    calls, results = versions
    results["nosuchdb"] = NotFoundError("Not found")
    with pytest.raises(FileNotFoundError, match="no database 'nosuchdb'"):
        wfdb.io.download.get_version("nosuchdb")

    # Other errors are not retried, and server errors only so many times
    results["private"] = api_error(403)
    with pytest.raises(PhysioNetAPIError):
        wfdb.io.download.get_version("private")
    results["down"] = [api_error(503)] * 5
    with pytest.raises(PhysioNetAPIError):
        wfdb.io.download.get_version("down")
    assert calls.count("private") == 1
    assert calls.count("down") == wfdb.io.download.VERSION_ATTEMPTS


def test_other_servers_are_read_without_physionet(monkeypatch):
    """With set_db_index_url, pn_dir files are read from that server, as before."""
    opened = []

    def openurl(url, mode="r", **kwargs):
        opened.append(url)
        return io.BytesIO(b"100 1 360 10\n")

    monkeypatch.setattr(wfdb.io._url, "openurl", openurl)
    wfdb.io.download.set_db_index_url("https://mirror.example.org/files/")
    try:
        assert wfdb.io.download._physionet_uri("mitdb/1.0.0", "100.hea") is None
        assert wfdb.io.download._stream_header("100.hea", "mitdb/1.0.0").startswith(
            "100"
        )
    finally:
        wfdb.io.download.set_db_index_url()
    assert opened == ["https://mirror.example.org/files/mitdb/1.0.0/100.hea"]
    assert (
        wfdb.io.download._physionet_uri("mitdb/1.0.0", "100.hea")
        == "physionet://mitdb/1.0.0/100.hea"
    )
