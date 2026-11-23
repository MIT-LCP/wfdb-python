"""Tests for reading records and annotations with physionet:// URIs."""

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
