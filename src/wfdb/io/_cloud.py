"""Write WFDB files to cloud filesystems through fsspec."""

import os
import shutil
import tempfile
from contextlib import contextmanager

from wfdb.io._coreio import CLOUD_PROTOCOLS, _fsspec_open, _join_path
from wfdb.io.archive import WFDBArchive


def is_cloud_path(path):
    """Return whether a path uses one of the supported cloud protocols."""
    return os.fspath(path).startswith(tuple(CLOUD_PROTOCOLS))


def check_cloud_write(write_dir, wfdb_archive=None):
    """Reject unsupported cloud destinations before any remote lookup."""
    if os.fspath(write_dir).startswith("physionet://"):
        raise ValueError("Writing to physionet:// is not supported")
    if isinstance(wfdb_archive, (str, os.PathLike)) and is_cloud_path(wfdb_archive):
        raise ValueError("Writing .wfdb archives to cloud storage is unsupported")
    if is_cloud_path(write_dir) and wfdb_archive is not None:
        if isinstance(wfdb_archive, WFDBArchive):
            raise ValueError("A cloud write_dir is unused with an open WFDBArchive")
        raise ValueError("Writing .wfdb archives to cloud storage is unsupported")


@contextmanager
def staged_cloud_write(write_dir, header_name=None):
    """Stage files locally and upload them to a cloud directory on success.

    Signal and annotation files are uploaded before headers. The main record
    header, if supplied, is uploaded last so readers do not discover an
    incomplete new record while its files are still being uploaded.

    """
    with tempfile.TemporaryDirectory(prefix="wfdb-cloud-") as staging:
        yield staging
        paths = []
        for root, _, names in os.walk(staging):
            for name in names:
                local_path = os.path.join(root, name)
                relative_path = os.path.relpath(local_path, staging).replace(
                    os.sep, "/"
                )
                paths.append((relative_path, local_path))

        paths.sort(
            key=lambda item: (
                item[0] == header_name,
                item[0].endswith(".hea"),
                item[0],
            )
        )
        for relative_path, local_path in paths:
            remote_path = _join_path(os.fspath(write_dir), relative_path)
            with open(local_path, "rb") as source:
                with _fsspec_open(remote_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
