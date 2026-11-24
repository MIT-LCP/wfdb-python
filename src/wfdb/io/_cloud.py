"""Write WFDB files to cloud filesystems through fsspec."""

import os
import shutil
import tempfile
from contextlib import contextmanager

from wfdb.io._coreio import CLOUD_PROTOCOLS, _fsspec_open, _join_path


def is_cloud_path(path):
    """Return whether a path uses one of the supported cloud protocols."""
    return os.fspath(path).startswith(tuple(CLOUD_PROTOCOLS))


@contextmanager
def staged_cloud_write(write_dir, header_name=None):
    """Stage files locally and upload them to a cloud directory on success.

    Signal and annotation files are uploaded before headers. The main record
    header, if supplied, is uploaded last so readers do not discover an
    incomplete new record while its files are still being uploaded.

    """
    if os.fspath(write_dir).startswith("physionet://"):
        raise ValueError("Writing to physionet:// is not supported")

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
