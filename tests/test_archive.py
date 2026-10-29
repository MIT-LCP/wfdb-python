import gc
import os
import shutil
import zipfile

import numpy as np
import pytest

import wfdb
from wfdb.io import archive as wfdb_archive_module
from wfdb.io.archive import WFDBArchive, create_archive


def record_files(record_name):
    """Return the header and signal file paths of a record on disk."""
    header = wfdb.rdheader(record_name)
    dir_name = os.path.dirname(record_name)
    files = [record_name + ".hea"]
    for file_name in dict.fromkeys(header.file_name or []):
        files.append(os.path.join(dir_name, file_name))
    return files


def archive_sample_record(record_name, tmp_path):
    """Put a record from sample-data into a .wfdb archive and return its path."""
    archive_path = os.fspath(tmp_path / (os.path.basename(record_name) + ".wfdb"))
    create_archive(record_files(record_name), archive_path)
    return archive_path


def assert_records_equal(record, expected):
    assert record.n_sig == expected.n_sig
    assert record.sig_len == expected.sig_len
    assert record.sig_name == expected.sig_name
    assert record.fmt == expected.fmt
    for attr in ["d_signal", "p_signal"]:
        if getattr(expected, attr) is not None:
            np.testing.assert_array_equal(
                getattr(record, attr), getattr(expected, attr)
            )
    for attr in ["e_d_signal", "e_p_signal"]:
        if getattr(expected, attr) is not None:
            for got, want in zip(getattr(record, attr), getattr(expected, attr)):
                np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "record_name",
    [
        "sample-data/100",
        "sample-data/test01_00s",
        "sample-data/flacformats",
        "sample-data/a103l",
    ],
)
@pytest.mark.parametrize("physical", [True, False])
def test_read_archive_matches_files(record_name, physical, tmp_path):
    """Reading from an archive gives the same result as reading the record files."""
    archive_path = archive_sample_record(record_name, tmp_path)

    record = wfdb.rdrecord(archive_path, physical=physical)
    expected = wfdb.rdrecord(record_name, physical=physical)

    assert_records_equal(record, expected)


def test_read_archive_expanded(tmp_path):
    """Signals with several samples per frame can be read without smoothing."""
    record_name = "sample-data/test01_00s_frame"
    archive_path = archive_sample_record(record_name, tmp_path)

    record = wfdb.rdrecord(archive_path, smooth_frames=False)
    expected = wfdb.rdrecord(record_name, smooth_frames=False)

    assert_records_equal(record, expected)


def test_read_archive_channels_and_range(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    record = wfdb.rdrecord(archive_path, sampfrom=1000, sampto=3000, channels=[1])
    expected = wfdb.rdrecord(
        "sample-data/100", sampfrom=1000, sampto=3000, channels=[1]
    )

    assert_records_equal(record, expected)


def test_rdsamp_and_rdheader_archive(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    signal, fields = wfdb.rdsamp(archive_path, sampto=500)
    expected_signal, expected_fields = wfdb.rdsamp("sample-data/100", sampto=500)
    np.testing.assert_array_equal(signal, expected_signal)
    assert fields == expected_fields

    header = wfdb.rdheader(archive_path)
    assert header.record_name == "100"
    assert header.sig_name == expected_fields["sig_name"]


def test_read_archive_without_sig_len(tmp_path):
    """The signal length is inferred from the dat file if the header lacks it."""
    expected = wfdb.rdrecord("sample-data/100")
    with open("sample-data/100.hea") as f:
        lines = f.read().splitlines()
    # Drop sig_len (and anything after it) from the record line
    lines[0] = " ".join(lines[0].split()[:3])
    archive_path = os.fspath(tmp_path / "100.wfdb")
    with WFDBArchive(archive_path, mode="w") as archive:
        archive.write("100.hea", ("\n".join(lines) + "\n").encode())
        archive.add_file("sample-data/100.dat")

    record = wfdb.rdrecord(archive_path)

    assert record.sig_len == expected.sig_len
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_read_archive_twice(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    first = wfdb.rdrecord(archive_path, sampto=100)
    second = wfdb.rdrecord(archive_path, sampto=100)

    np.testing.assert_array_equal(first.p_signal, second.p_signal)


def test_read_replaced_archive(tmp_path):
    """Replacing an archive and reading it again returns the new record."""
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    assert wfdb.rdrecord(archive_path).sig_name == ["MLII", "V5"]

    create_archive(record_files("sample-data/test01_00s"), archive_path)

    record = wfdb.rdrecord(archive_path)
    expected = wfdb.rdrecord("sample-data/test01_00s")
    assert_records_equal(record, expected)


def test_read_archive_header_named_differently(tmp_path):
    """A single header can be read even if it isn't named after the archive."""
    archive_path = os.fspath(tmp_path / "renamed.wfdb")
    create_archive(record_files("sample-data/100"), archive_path)

    record = wfdb.rdrecord(archive_path, sampto=100)

    assert record.record_name == "100"


def test_read_archive_several_headers(tmp_path):
    """An archive with several headers, none named after it, can't be read.

    Such an archive can only be made by other software, and it is not clear which record
    to read.
    """
    archive_path = os.fspath(tmp_path / "both.wfdb")
    with zipfile.ZipFile(archive_path, "w") as zf:
        for path in record_files("sample-data/100") + ["sample-data/test01_00s.hea"]:
            zf.write(path, arcname=os.path.basename(path))

    with pytest.raises(FileNotFoundError, match="several header files"):
        wfdb.rdrecord(archive_path)


def test_read_archive_ignores_files_in_folders(tmp_path):
    """Only header files at the top level of an archive are considered.

    macOS, for example, adds '__MACOSX/._100.hea' when it zips '100.hea', which
    shouldn't count as a second header.
    """
    archive_path = os.fspath(tmp_path / "renamed.wfdb")
    with zipfile.ZipFile(archive_path, "w") as zf:
        for path in record_files("sample-data/100"):
            file_name = os.path.basename(path)
            zf.write(path, arcname=file_name)
            zf.writestr(f"__MACOSX/._{file_name}", b"\x00\x05\x16\x07")

    record = wfdb.rdrecord(archive_path, sampto=100)

    assert record.record_name == "100"


def test_read_archive_missing_dat_file(tmp_path):
    """A dat file named in the header but missing from the archive is an error.

    This is the case even if the archive contains another dat file.
    """
    with open("sample-data/100.hea") as f:
        header = f.read().replace("100.dat", "other.dat")
    archive_path = os.fspath(tmp_path / "100.wfdb")
    with WFDBArchive(archive_path, mode="w") as archive:
        archive.write("100.hea", header.encode())
        archive.add_file("sample-data/100.dat")

    with pytest.raises(FileNotFoundError, match="other.dat"):
        wfdb.rdrecord(archive_path)


def test_read_archive_without_header(tmp_path):
    archive_path = os.fspath(tmp_path / "100.wfdb")
    create_archive(["sample-data/100.dat"], archive_path)

    with pytest.raises(FileNotFoundError, match="100.hea"):
        wfdb.rdrecord(archive_path)


def test_read_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="Archive not found"):
        wfdb.rdrecord(os.fspath(tmp_path / "missing.wfdb"))


def test_read_invalid_archive(tmp_path):
    archive_path = tmp_path / "invalid.wfdb"
    archive_path.write_bytes(b"not a zip file")

    with pytest.raises(ValueError, match="Not a valid WFDB archive"):
        wfdb.rdrecord(os.fspath(archive_path))


def test_read_archive_header_too_large(tmp_path, monkeypatch):
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    monkeypatch.setattr(wfdb_archive_module, "MAX_HEADER_SIZE", 10)

    with pytest.raises(ValueError, match="larger than 10 bytes"):
        wfdb.rdrecord(archive_path)


def test_read_multi_segment_archive_not_supported(tmp_path):
    record_dir = "sample-data/multi-segment/fixed1"
    archive_path = os.fspath(tmp_path / "v102s.wfdb")
    # create_archive only accepts one header, so build the archive the way
    # other software might
    with zipfile.ZipFile(archive_path, "w") as zf:
        for name in sorted(os.listdir(record_dir)):
            zf.write(os.path.join(record_dir, name), arcname=name)

    with pytest.raises(NotImplementedError, match="Multi-segment"):
        wfdb.rdrecord(archive_path)
    with pytest.raises(NotImplementedError, match="Multi-segment"):
        wfdb.rdheader(archive_path, rd_segments=True)


def test_read_archive_unsupported_locations():
    with pytest.raises(ValueError, match="pn_dir"):
        wfdb.rdrecord("100.wfdb", pn_dir="mitdb")
    with pytest.raises(NotImplementedError, match="cloud storage"):
        wfdb.rdrecord("s3://bucket/100.wfdb")


def test_create_archive_compression(tmp_path):
    """Headers are compressed, and signal files are stored uncompressed.

    This way, part of a signal file can be read without decompressing all of it.
    """
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with zipfile.ZipFile(archive_path) as zf:
        compress_types = {info.filename: info.compress_type for info in zf.infolist()}

    assert compress_types == {
        "100.hea": zipfile.ZIP_DEFLATED,
        "100.dat": zipfile.ZIP_STORED,
    }


def test_create_archive_replaces_existing(tmp_path):
    archive_path = os.fspath(tmp_path / "record.wfdb")
    create_archive(record_files("sample-data/test01_00s"), archive_path)

    create_archive(record_files("sample-data/100"), archive_path)

    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["100.dat", "100.hea"]


def test_archive_write_error_keeps_existing_archive(tmp_path):
    """An error while writing leaves the existing archive unchanged.

    No temporary files are left behind either.
    """
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    with open(archive_path, "rb") as f:
        original = f.read()

    with pytest.raises(RuntimeError):
        with WFDBArchive(archive_path, mode="w") as archive:
            archive.add_file("sample-data/test01_00s.hea")
            raise RuntimeError("failed while writing")

    with open(archive_path, "rb") as f:
        assert f.read() == original
    assert os.listdir(tmp_path) == ["100.wfdb"]


def test_archive_mode_checks(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with WFDBArchive(archive_path, mode="r") as archive:
        with pytest.raises(ValueError, match="cannot be written to"):
            archive.write("extra.txt", b"data")

    with WFDBArchive(os.fspath(tmp_path / "new.wfdb"), mode="w") as archive:
        archive.write("new.hea", b"new 0\n")
        with pytest.raises(ValueError, match="already written"):
            archive.write("new.hea", b"new 0\n")
        with pytest.raises(ValueError, match="cannot be read from"):
            archive.open("new.hea")

    with pytest.raises(ValueError, match="mode"):
        WFDBArchive(archive_path, mode="a")


def test_archive_holds_one_record(tmp_path):
    """Only one header file can be written to an archive.

    This way, it is clear which record the archive holds. Other files can be added.
    """
    archive_path = os.fspath(tmp_path / "100.wfdb")

    with pytest.raises(ValueError, match="holds a single record"):
        create_archive(
            record_files("sample-data/100") + ["sample-data/test01_00s.hea"],
            archive_path,
        )
    assert os.listdir(tmp_path) == []

    with WFDBArchive(archive_path, mode="w") as archive:
        for path in record_files("sample-data/100"):
            archive.add_file(path)
        archive.add_file("sample-data/100.atr")
        with pytest.raises(ValueError, match="already contains the header"):
            archive.write("other.hea", b"other 0\n")

    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["100.atr", "100.dat", "100.hea"]


def write_kwargs(fmt="16"):
    rng = np.random.default_rng(1234)
    return dict(
        fs=250,
        units=["mV", "mV"],
        sig_name=["I", "II"],
        p_signal=rng.standard_normal((1000, 2)),
        fmt=[fmt, fmt],
    )


@pytest.mark.parametrize("fmt", ["16", "24", "32", "516"])
def test_wrsamp_archive_matches_files(fmt, tmp_path):
    """Writing to an archive gives the same result as writing separate files."""
    kwargs = write_kwargs(fmt)
    archive_path = os.fspath(tmp_path / "archive" / "rec.wfdb")
    files_dir = tmp_path / "files"
    files_dir.mkdir()

    wfdb.wrsamp("rec", wfdb_archive=archive_path, **kwargs)
    wfdb.wrsamp("rec", write_dir=os.fspath(files_dir), **kwargs)

    for physical in [True, False]:
        record = wfdb.rdrecord(archive_path, physical=physical)
        expected = wfdb.rdrecord(os.fspath(files_dir / "rec"), physical=physical)
        assert_records_equal(record, expected)


def test_wrsamp_archive_object(tmp_path):
    """An open WFDBArchive is written to and left open, with no separate files."""
    archive_path = os.fspath(tmp_path / "rec.wfdb")

    with WFDBArchive(archive_path, mode="w") as archive:
        wfdb.wrsamp(
            "rec",
            write_dir=os.fspath(tmp_path),
            wfdb_archive=archive,
            **write_kwargs(),
        )
        archive.write("notes.txt", b"extra file")

    assert os.listdir(tmp_path) == ["rec.wfdb"]
    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["notes.txt", "rec.dat", "rec.hea"]
    assert wfdb.rdrecord(archive_path).sig_len == 1000


def test_record_wrsamp_archive_round_trip(tmp_path):
    """A record read from an archive can be written to a new archive."""
    source = archive_sample_record("sample-data/100", tmp_path)
    record = wfdb.rdrecord(source, physical=False)
    copy_path = os.fspath(tmp_path / "copy" / "100.wfdb")

    record.wrsamp(wfdb_archive=copy_path)

    assert_records_equal(
        wfdb.rdrecord(copy_path, physical=False),
        wfdb.rdrecord("sample-data/100", physical=False),
    )


def test_wrsamp_archive_replaces_existing(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    wfdb.wrsamp("100", wfdb_archive=archive_path, **write_kwargs())

    with zipfile.ZipFile(archive_path) as zf:
        assert sorted(zf.namelist()) == ["100.dat", "100.hea"]
    assert wfdb.rdrecord(archive_path).sig_name == ["I", "II"]


def test_wrsamp_archive_error_writes_nothing(tmp_path, monkeypatch):
    """A failed write leaves no archive or temporary file.

    This is the case even if the failure happens after the header was written.
    """

    def fail(*args, **kwargs):
        raise RuntimeError("failed while writing the dat file")

    monkeypatch.setattr(wfdb.io._signal, "wr_dat_file", fail)

    with pytest.raises(RuntimeError, match="dat file"):
        wfdb.wrsamp(
            "rec",
            wfdb_archive=os.fspath(tmp_path / "rec.wfdb"),
            **write_kwargs(),
        )

    assert os.listdir(tmp_path) == []


def test_wrsamp_two_records_in_one_archive(tmp_path):
    """Writing a second record into an archive fails.

    The archive is not saved.
    """
    archive_path = os.fspath(tmp_path / "rec.wfdb")

    with pytest.raises(ValueError, match="holds a single record"):
        with WFDBArchive(archive_path, mode="w") as archive:
            wfdb.wrsamp("rec", wfdb_archive=archive, **write_kwargs())
            wfdb.wrsamp("other", wfdb_archive=archive, **write_kwargs())

    assert os.listdir(tmp_path) == []


def test_wrsamp_archive_invalid_argument(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with pytest.raises(TypeError, match="wfdb_archive"):
        wfdb.wrsamp("rec", wfdb_archive=123, **write_kwargs())

    with WFDBArchive(archive_path, mode="r") as archive:
        with pytest.raises(ValueError, match="cannot be written to"):
            wfdb.wrsamp("rec", wfdb_archive=archive, **write_kwargs())


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
@pytest.mark.parametrize("umask", [0o022, 0o027])
def test_archive_permissions(umask, tmp_path, monkeypatch):
    """A new archive gets the same permissions as other new files.

    A replaced archive keeps its permissions. The umask is never changed, since that
    would affect files created by other threads at the same time.
    """
    archive_path = tmp_path / "rec.wfdb"
    other_path = tmp_path / "other.txt"
    old_umask = os.umask(umask)
    try:
        other_path.write_text("")

        def fail(mask):
            raise AssertionError("the umask was changed")

        monkeypatch.setattr(os, "umask", fail)
        wfdb.wrsamp("rec", wfdb_archive=os.fspath(archive_path), **write_kwargs())
    finally:
        monkeypatch.undo()
        os.umask(old_umask)
    assert archive_path.stat().st_mode == other_path.stat().st_mode

    archive_path.chmod(0o640)
    wfdb.wrsamp("rec", wfdb_archive=os.fspath(archive_path), **write_kwargs())
    assert archive_path.stat().st_mode & 0o777 == 0o640


def test_archive_temporary_file(tmp_path):
    """While an archive is written, its temporary file is named after it.

    If the archive is discarded without being closed, the temporary file is removed and
    nothing is saved.
    """
    archive = WFDBArchive(os.fspath(tmp_path / "rec.wfdb"), mode="w")
    archive.write("rec.hea", b"rec 0\n")
    (tmp_file,) = os.listdir(tmp_path)
    assert tmp_file.startswith(".rec.wfdb.") and tmp_file.endswith(".tmp")

    del archive
    gc.collect()

    assert os.listdir(tmp_path) == []


def test_rdrecord_opens_archive_once(tmp_path, monkeypatch):
    """The header and the signals are read from the same open archive."""
    archive_path = archive_sample_record("sample-data/100", tmp_path)
    opened = []
    original_init = WFDBArchive.__init__

    def counting_init(self, *args, **kwargs):
        opened.append(args)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(WFDBArchive, "__init__", counting_init)

    wfdb.rdrecord(archive_path, sampto=100)

    assert len(opened) == 1


def test_rdann_archive(tmp_path):
    """Annotations read from an archive match the annotation file on disk."""
    archive_path = os.fspath(tmp_path / "100.wfdb")
    create_archive(
        record_files("sample-data/100") + ["sample-data/100.atr"],
        archive_path,
    )

    ann = wfdb.rdann(archive_path, "atr", sampfrom=1000, sampto=50000)
    expected = wfdb.rdann("sample-data/100", "atr", sampfrom=1000, sampto=50000)

    assert ann.record_name == "100"
    assert ann.extension == "atr"
    assert ann.fs == expected.fs
    np.testing.assert_array_equal(ann.sample, expected.sample)
    assert ann.symbol == expected.symbol
    assert ann.aux_note == expected.aux_note


def test_rdann_archive_header_named_differently(tmp_path):
    """Annotation files are named after the record's header, not the archive."""
    archive_path = os.fspath(tmp_path / "renamed.wfdb")
    create_archive(
        record_files("sample-data/100") + ["sample-data/100.atr"],
        archive_path,
    )

    ann = wfdb.rdann(archive_path, "atr")

    assert ann.record_name == "100"
    assert len(ann.sample) == len(wfdb.rdann("sample-data/100", "atr").sample)


def test_rdann_archive_errors(tmp_path):
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    with pytest.raises(FileNotFoundError, match="100.atr"):
        wfdb.rdann(archive_path, "atr")
    with pytest.raises(ValueError, match="pn_dir"):
        wfdb.rdann(archive_path, "atr", pn_dir="mitdb")


def test_create_archive_old_file_dates(tmp_path):
    """Files dated before 1980, which ZIP can't represent, can be archived."""
    for path in record_files("sample-data/100"):
        copy = tmp_path / os.path.basename(path)
        shutil.copyfile(path, copy)
        os.utime(copy, (0, 0))
    archive_path = os.fspath(tmp_path / "archive" / "100.wfdb")

    create_archive([tmp_path / "100.hea", tmp_path / "100.dat"], archive_path)

    expected = wfdb.rdrecord("sample-data/100", sampto=100)
    record = wfdb.rdrecord(archive_path, sampto=100)
    np.testing.assert_array_equal(record.p_signal, expected.p_signal)


def test_archive_abort_removes_temporary_file(tmp_path):
    """The temporary file is removed even if closing the zip file fails."""
    archive = WFDBArchive(os.fspath(tmp_path / "rec.wfdb"), mode="w")
    archive.write("rec.hea", b"rec 0\n")

    zip_file = archive._zipfile
    real_close = zip_file.close

    def fail():
        # Close the file, as when writing the end of the zip file fails
        zip_file.close = real_close
        real_close()
        raise OSError("disk full")

    zip_file.close = fail

    with pytest.raises(OSError, match="disk full"):
        archive.abort()

    assert os.listdir(tmp_path) == []


def test_closed_archive(tmp_path):
    """Using a closed archive raises a clear error."""
    archive_path = archive_sample_record("sample-data/100", tmp_path)

    archive = WFDBArchive(archive_path, mode="r")
    archive.close()
    for call in [
        archive.namelist,
        lambda: archive.exists("100.hea"),
        lambda: archive.file_size("100.hea"),
        lambda: archive.open("100.hea"),
        archive.read_header,
    ]:
        with pytest.raises(ValueError, match="is closed"):
            call()

    archive = WFDBArchive(os.fspath(tmp_path / "new.wfdb"), mode="w")
    archive.close()
    with pytest.raises(ValueError, match="is closed"):
        archive.write("new.hea", b"new 0\n")
    with pytest.raises(ValueError, match="is closed"):
        archive.add_file("sample-data/100.hea")
    with pytest.raises(ValueError, match="is closed"):
        wfdb.wrsamp("rec", wfdb_archive=archive, **write_kwargs())


def test_wrsamp_archive_write_dir(tmp_path, monkeypatch):
    """A relative archive path is relative to `write_dir`; an absolute one isn't."""
    write_dir = tmp_path / "write_dir"
    write_dir.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    wfdb.wrsamp(
        "rec", write_dir=os.fspath(write_dir), wfdb_archive="rec.wfdb", **write_kwargs()
    )
    record = wfdb.rdrecord(os.fspath(write_dir / "rec.wfdb"), physical=False)
    record.wrsamp(write_dir=os.fspath(write_dir), wfdb_archive="copy.wfdb")
    record.wrsamp(
        write_dir=os.fspath(write_dir), wfdb_archive=os.fspath(tmp_path / "abs.wfdb")
    )

    assert os.listdir(cwd) == []
    assert sorted(os.listdir(write_dir)) == ["copy.wfdb", "rec.wfdb"]
    assert os.path.exists(tmp_path / "abs.wfdb")


def test_archive_extension_case(tmp_path):
    """The .wfdb extension is recognized in any case."""
    archive_path = os.fspath(tmp_path / "100.WFDB")
    create_archive(record_files("sample-data/100"), archive_path)

    assert os.listdir(tmp_path) == ["100.WFDB"]
    record = wfdb.rdrecord(archive_path, sampto=100)
    assert record.record_name == "100"
    assert wfdb.rdheader(archive_path).sig_name == ["MLII", "V5"]
    with WFDBArchive(archive_path) as archive:
        assert archive.record_name == "100"
        assert archive.path == archive_path


def test_archive_write_bytes_like(tmp_path, monkeypatch):
    """`write` streams any bytes-like object, with the same entries as `writestr`.

    ZIP64 is used for large files: the limit is lowered so that a small file needs it.
    """
    signal = np.arange(1000, dtype=np.int16)
    archive_path = os.fspath(tmp_path / "rec.wfdb")
    with WFDBArchive(archive_path, mode="w") as archive:
        archive.write("rec.hea", b"rec 1 250 1000\n")
        archive.write("array.dat", signal)
        archive.write("view.dat", memoryview(signal.tobytes()))
        archive.write("bytearray.dat", bytearray(b"abc"))
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 100)
        archive.write("zip64.dat", signal)
        monkeypatch.undo()

    with zipfile.ZipFile(archive_path) as zf:
        assert zf.read("array.dat") == signal.tobytes()
        assert zf.read("view.dat") == signal.tobytes()
        assert zf.read("bytearray.dat") == b"abc"
        assert zf.read("zip64.dat") == signal.tobytes()
        for info in zf.infolist():
            assert info.date_time[0] >= 2020
            assert info.external_attr == 0o600 << 16
        assert zf.getinfo("rec.hea").compress_type == zipfile.ZIP_DEFLATED
        assert zf.getinfo("array.dat").compress_type == zipfile.ZIP_STORED
