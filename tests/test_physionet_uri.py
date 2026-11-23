"""Tests for reading records and annotations with physionet:// URIs."""

import importlib
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


@pytest.fixture
def demo_database(physionet_cache, offline, versions, monkeypatch):
    """A database 'demo', version 1.0, in the physionet cache, with downloads recorded.

    Returns the list of keyword arguments that physionet.download was called with.
    """
    import physionet

    _, results = versions
    results["demo"] = "1.0"
    project = physionet_cache / "physionet.org" / "demo" / "1.0"
    (project / "sub").mkdir(parents=True)
    (project / "RECORDS").write_text("100\n041s\n")
    (project / "ANNOTATORS").write_text("atr\tReference beat annotations\n")
    (project / "sub" / "RECORDS").write_text("200\n")
    (project / "sub" / "200.hea").write_text(
        "200 1 360 10\n200.dat 16 200 16 0 0 0 0 ECG\n"
    )
    shutil.copy(SAMPLE_DATA / "100.hea", project)
    for path in (SAMPLE_DATA / "multi-segment" / "041s").glob("*.hea"):
        shutil.copy(path, project)

    # The checksum list: 100 has annotations, 041s does not
    manifest = physionet_cache / "physionet.org" / ".manifests" / "demo" / "1.0"
    manifest.mkdir(parents=True)
    names = ["RECORDS", "ANNOTATORS", "100.hea", "100.dat", "100.atr", "041s.hea"]
    names += ["041s01.hea", "041s01.dat", "041s02.hea", "041s02.dat"]
    names += ["sub/RECORDS", "sub/200.hea", "sub/200.dat", "sub/200.atr"]
    (manifest / "SHA256SUMS.txt").write_text(
        "".join(f"{'0' * 64}  {n}\n" for n in names)
    )
    (manifest / "ACCESS").write_text("open\n")

    calls = []
    monkeypatch.setattr(physionet, "download", lambda *a, **kw: calls.append((a, kw)))
    # dl_database checks that the database's web page exists
    monkeypatch.setattr(wfdb.io._url, "openurl", lambda url, *a, **kw: io.BytesIO())
    return calls


def test_dl_database_downloads_with_physionet(demo_database, tmp_path):
    wfdb.dl_database("demo", str(tmp_path / "out"), annotators=["atr"])

    [(args, kwargs)] = demo_database
    assert args == ("demo", "1.0")
    files = kwargs.pop("files")
    assert kwargs == {
        "output_dir": str(tmp_path / "out"),
        "verbose": False,
        "show_progress": False,
        "jobs": 2,
        "overwrite": False,
    }
    # Each record's header and signal files, and the annotation files there are
    expected = ["100.hea", "100.dat", "100.atr", "041s.hea", "041s01.hea"]
    expected += ["041s01.dat", "041s02.hea", "041s02.dat"]
    assert files == {name: name for name in expected}


def test_dl_database_subdirectory_and_flat_layout(demo_database, tmp_path):
    out = tmp_path / "out"
    wfdb.dl_database("demo/sub", str(out), annotators=None)
    wfdb.dl_database(
        "demo", str(out), records=["100"], annotators=None, keep_subdirs=False
    )

    # Files are saved relative to the subdirectory, as before
    assert demo_database[0][1]["files"] == {
        "sub/200.hea": "200.hea",
        "sub/200.dat": "200.dat",
    }
    assert demo_database[1][1]["files"] == {"100.hea": "100.hea", "100.dat": "100.dat"}


@pytest.mark.parametrize("overwrite", [False, True])
def test_dl_database_overwrite(demo_database, tmp_path, overwrite):
    out = tmp_path / "out"
    out.mkdir()
    (out / "100.dat").write_bytes(b"saved before")
    wfdb.dl_database(
        "demo", str(out), records=["100"], annotators=None, overwrite=overwrite
    )

    # physionet replaces saved files itself, after checking every file and
    # destination, so wfdb removes nothing
    assert demo_database[0][1]["overwrite"] is overwrite
    assert (out / "100.dat").read_bytes() == b"saved before"


@pytest.mark.parametrize("keep_subdirs", [True, False])
def test_dl_files_downloads_with_physionet(demo_database, tmp_path, keep_subdirs):
    out = tmp_path / "out"
    wfdb.dl_files(
        "demo", str(out), ["100.hea", "sub/200.hea"], keep_subdirs=keep_subdirs
    )

    [(args, kwargs)] = demo_database
    assert args == ("demo", "1.0")
    expected = "sub/200.hea" if keep_subdirs else "200.hea"
    assert kwargs["files"] == {"100.hea": "100.hea", "sub/200.hea": expected}
    assert out.is_dir()


def test_dl_files_from_another_server(demo_database, monkeypatch, tmp_path):
    """With set_db_index_url, files are downloaded from that server, as before."""
    downloaded = []
    monkeypatch.setattr(wfdb.io.download, "dl_pn_file", downloaded.append)
    wfdb.io.download.set_db_index_url("https://mirror.example.org/files/")
    try:
        wfdb.dl_files("demo", str(tmp_path / "out"), ["100.hea"])
    finally:
        wfdb.io.download.set_db_index_url()
    assert demo_database == []
    assert [(name, subdir, db) for name, subdir, db, *_ in downloaded] == [
        ("100.hea", "", "demo/1.0")
    ]


@pytest.mark.parametrize(
    ("files", "keep_subdirs", "saved"),
    [
        # A path outside dl_dir
        (["../victim.dat"], True, "../victim.dat"),
        # Two files with one name, without their subdirectories
        (["RECORDS", "sub/RECORDS"], False, "RECORDS"),
        # Names that differ only in case, which are one file on macOS and Windows
        (["100.hea", "sub/100.HEA"], False, "100.hea"),
    ],
    ids=["outside", "same-name", "case"],
)
def test_dl_files_overwrite_replaces_nothing_for_an_unsafe_request(
    demo_database, monkeypatch, tmp_path, files, keep_subdirs, saved
):
    """physionet refuses these requests before replacing or downloading anything."""
    import physionet

    # The real physionet.download, which checks the request before any request
    monkeypatch.setattr(
        physionet, "download", importlib.import_module("physionet.download").download
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / saved).write_bytes(b"saved before")

    with pytest.raises(ValueError):
        wfdb.dl_files(
            "demo", str(out), files, keep_subdirs=keep_subdirs, overwrite=True
        )
    assert (out / saved).read_bytes() == b"saved before"


def test_dl_files_overwrite_removes_nothing_itself(demo_database, tmp_path):
    """wfdb leaves replacing saved files to physionet, which checks them first."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "100.hea").write_bytes(b"saved before")

    wfdb.dl_files("demo", str(out), ["100.hea", "999.hea"], overwrite=True)

    assert (out / "100.hea").read_bytes() == b"saved before"
    [(args, kwargs)] = demo_database
    assert kwargs["overwrite"] is True
    assert kwargs["files"] == {"100.hea": "100.hea", "999.hea": "999.hea"}


def test_get_dbs_lists_the_latest_version_of_each_database(monkeypatch):
    """get_dbs lists databases from PhysioNet's API, once each, sorted by slug."""
    import physionet
    from physionet.api.endpoints import ProjectsAPI

    def project(slug, version, title, resource_type="Database", latest=True):
        return physionet.api.models.PublishedProject.from_dict(
            {
                "slug": slug,
                "version": version,
                "title": title,
                "resource_type": resource_type,
                "is_latest_version": latest,
            }
        )

    projects = [
        project("mitdb", "1.0.0", "MIT-BIH Arrhythmia Database"),
        project("ludb", "1.0.0", "Old title", latest=False),
        project("ludb", "1.0.1", "Lobachevsky University Database"),
        project("wfdb-python", "4.0.0", "WFDB Python Package", "Software"),
        project("challenge-2015", "1.0.0", "Reducing False Alarms", "Challenge"),
    ]
    monkeypatch.setattr(ProjectsAPI, "list_published", lambda self: projects)

    assert wfdb.get_dbs() == [
        ["ludb", "Lobachevsky University Database"],
        ["mitdb", "MIT-BIH Arrhythmia Database"],
    ]


@pytest.mark.network
def test_get_dbs_from_physionet():
    dbs = wfdb.get_dbs()
    assert ["mitdb", "MIT-BIH Arrhythmia Database"] in dbs
    assert len({slug for slug, title in dbs}) == len(dbs)
