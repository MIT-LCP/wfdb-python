import gc
import os
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
