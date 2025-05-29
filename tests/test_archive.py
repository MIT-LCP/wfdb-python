import os
import tempfile
import zipfile

import numpy as np
import pytest

from wfdb import rdrecord, wrsamp
from wfdb.io.archive import WFDBArchive

np.random.seed(1234)


@pytest.fixture
def temp_record():
    """
    Create a temporary WFDB record and archive for testing.

    This fixture generates a synthetic 2-channel signal, writes it to a temporary
    directory using `wrsamp`, then creates an uncompressed `.wfdb` archive (ZIP container)
    containing the `.hea` and `.dat` files. The archive is used to test read/write
    round-trip support for WFDB archives.

    Yields
    ------
    dict
        A dictionary containing:
        - 'record_name': Path to the record base name (without extension).
        - 'archive_path': Full path to the created `.wfdb` archive.
        - 'original_signal': The original NumPy array of the signal.
        - 'fs': The sampling frequency.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        record_basename = "testrecord"
        fs = 250
        sig_len = 1000
        sig = (np.random.randn(sig_len, 2) * 1000).astype(np.float32)

        # Write into tmpdir with record name only
        wrsamp(
            record_name=record_basename,
            fs=fs,
            units=["mV", "mV"],
            sig_name=["I", "II"],
            p_signal=sig,
            fmt=["24", "24"],
            adc_gain=[200.0, 200.0],
            baseline=[0, 0],
            write_dir=tmpdir,
        )

        # Construct full paths for archive creation
        hea_path = os.path.join(tmpdir, record_basename + ".hea")
        dat_path = os.path.join(tmpdir, record_basename + ".dat")
        archive_path = os.path.join(tmpdir, record_basename + ".wfdb")

        with WFDBArchive(record_name=record_basename, mode="w") as archive:
            archive.create_archive(
                file_list=[hea_path, dat_path],
                output_path=archive_path,
            )

        yield {
            "record_name": os.path.join(tmpdir, record_basename),
            "archive_path": archive_path,
            "original_signal": sig,
            "fs": fs,
        }


def test_wfdb_archive_inline_round_trip():
    """
    There are two ways of creating an archive:

    1. Inline archive creation via wrsamp(..., wfdb_archive=...)
    This creates the .hea and .dat files directly inside the archive as part of the record writing step.

    2. Two-step creation via wrsamp(...) followed by WFDBArchive.create_archive(...)
    This writes regular WFDB files to disk, which are then added to an archive container afterward.

    Test round-trip read/write using inline archive creation via `wrsamp(..., wfdb_archive=...)`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        record_basename = "testrecord"
        record_path = os.path.join(tmpdir, record_basename)
        archive_path = record_path + ".wfdb"
        fs = 250
        sig_len = 1000
        sig = (np.random.randn(sig_len, 2) * 1000).astype(np.float32)

        # Create archive inline
        wfdb_archive = WFDBArchive(record_basename, mode="w")
        wrsamp(
            record_name=record_basename,
            fs=fs,
            units=["mV", "mV"],
            sig_name=["I", "II"],
            p_signal=sig,
            fmt=["24", "24"],
            adc_gain=[200.0, 200.0],
            baseline=[0, 0],
            write_dir=tmpdir,
            wfdb_archive=wfdb_archive,
        )
        wfdb_archive.close()

        assert os.path.exists(archive_path), "Archive was not created"

        # Read back from archive
        record = rdrecord(archive_path)

        assert record.fs == fs
        assert record.n_sig == 2
        assert record.p_signal.shape == sig.shape

        # Add tolerance to account for loss of precision during archive round-trip
        np.testing.assert_allclose(record.p_signal, sig, rtol=1e-2, atol=3e-3)


def test_wfdb_archive_round_trip(temp_record):
    record_name = temp_record["record_name"]
    archive_path = temp_record["archive_path"]
    original_signal = temp_record["original_signal"]
    fs = temp_record["fs"]

    assert os.path.exists(archive_path), "Archive was not created"

    record = rdrecord(archive_path)

    assert record.fs == fs
    assert record.n_sig == 2
    assert record.p_signal.shape == original_signal.shape

    # Add tolerance to account for loss of precision during archive round-trip
    np.testing.assert_allclose(
        record.p_signal, original_signal, rtol=1e-2, atol=3e-3
    )


def test_archive_read_subset_channels(temp_record):
    """
    Test reading a subset of channels from an archive.
    """
    archive_path = temp_record["archive_path"]
    original_signal = temp_record["original_signal"]

    record = rdrecord(archive_path, channels=[1])

    assert record.n_sig == 1
    assert record.p_signal.shape[0] == original_signal.shape[0]

    # Add tolerance to account for loss of precision during archive round-trip
    np.testing.assert_allclose(
        record.p_signal[:, 0], original_signal[:, 1], rtol=1e-2, atol=3e-3
    )


def test_archive_read_partial_samples(temp_record):
    """
    Test reading a sample range from the archive.
    """
    archive_path = temp_record["archive_path"]
    original_signal = temp_record["original_signal"]

    start, stop = 100, 200
    record = rdrecord(archive_path, sampfrom=start, sampto=stop)

    assert record.p_signal.shape == (stop - start, original_signal.shape[1])
    np.testing.assert_allclose(
        record.p_signal, original_signal[start:stop], rtol=1e-2, atol=1e-3
    )


def test_archive_missing_file_error(temp_record):
    """
    Ensure appropriate error is raised when expected files are missing from the archive.
    """
    archive_path = temp_record["archive_path"]

    # Remove one file from archive (e.g. the .dat file)
    with zipfile.ZipFile(archive_path, "a") as zf:
        zf_name = [name for name in zf.namelist() if name.endswith(".dat")][0]
        zf.fp = None  # Prevent auto-close bug in some zipfile implementations
    os.rename(archive_path, archive_path + ".bak")
    with (
        zipfile.ZipFile(archive_path + ".bak", "r") as zin,
        zipfile.ZipFile(archive_path, "w") as zout,
    ):
        for item in zin.infolist():
            if not item.filename.endswith(".dat"):
                zout.writestr(item, zin.read(item.filename))
    os.remove(archive_path + ".bak")

    with pytest.raises(FileNotFoundError, match=r".*\.dat.*"):
        rdrecord(archive_path)
