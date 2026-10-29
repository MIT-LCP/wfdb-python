"""Support for .wfdb archive files.

A .wfdb archive is a ZIP file containing the files of a single WFDB
record: its header file, the signal files listed in the header, and
optionally annotation files.
Header files are compressed, while signal files are stored without
compression so that a range of samples can be read without
decompressing the whole file.
"""

import contextlib
import os
import secrets
import stat
import weakref
import zipfile
from contextlib import contextmanager

ARCHIVE_EXTENSION = ".wfdb"

# Largest header file that will be read from an archive. Headers are
# small, so anything bigger is treated as a damaged or malicious file.
MAX_HEADER_SIZE = 1024 * 1024


class WFDBArchive:
    """A .wfdb archive opened for reading or writing.

    In read mode the archive must already exist. In write mode a new
    archive is built in a temporary file next to the target path and
    only moved into place when the archive is closed without an error,
    so an existing archive is either fully replaced or left untouched.
    The new archive gets the permissions of the archive it replaces, or
    the default permissions for new files.

    An archive holds a single record, so only one header file can be
    written to it. Annotation files and other files can be added
    alongside the record's header and signal files.

    Use the object as a context manager so that it is always closed. An
    archive in write mode that is discarded without being closed is not
    saved, and its temporary file is removed.

    Parameters
    ----------
    path : str or os.PathLike
        The path of the archive. The .wfdb extension is added if it is
        missing (the extension is not case-sensitive).
    mode : str, optional
        'r' to read an existing archive (default), or 'w' to write a new
        one.

    Examples
    --------
    >>> with WFDBArchive("record.wfdb", mode="w") as archive:
    ...     wfdb.wrsamp("record", fs=250, units=["mV"], sig_name=["I"],
    ...                 p_signal=signal, fmt=["16"], wfdb_archive=archive)

    """

    def __init__(self, path, mode="r"):
        if mode not in ("r", "w"):
            raise ValueError("mode must be 'r' or 'w'")

        path = os.fspath(path)
        if not is_archive_path(path):
            path += ARCHIVE_EXTENSION
        self.path = path
        self.mode = mode
        self._tmp_path = None

        if mode == "r":
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Archive not found: {path}")
            try:
                self._zipfile = zipfile.ZipFile(path, mode="r")
            except zipfile.BadZipFile as e:
                raise ValueError(f"Not a valid WFDB archive: {path}") from e
        else:
            self._tmp_path = _create_temp_file(path)
            try:
                # Files dated before 1980 (which ZIP can't represent) get the
                # earliest possible date instead of raising an error
                self._zipfile = zipfile.ZipFile(
                    self._tmp_path, mode="w", strict_timestamps=False
                )
            except BaseException:
                os.remove(self._tmp_path)
                raise

        # Close the archive, and remove an unsaved temporary file, if the
        # object is discarded without being closed
        self._finalizer = weakref.finalize(
            self, _discard, self._zipfile, self._tmp_path
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()

    @property
    def record_name(self):
        """The record name implied by the archive file name."""
        return os.path.basename(self.path)[: -len(ARCHIVE_EXTENSION)]

    def namelist(self):
        """Return the names of the files in the archive."""
        self._require_open()
        return self._zipfile.namelist()

    def exists(self, file_name):
        """Check whether a file is in the archive."""
        return file_name in self.namelist()

    def file_size(self, file_name):
        """Return the uncompressed size of a file in the archive, in bytes."""
        return self._getinfo(file_name).file_size

    def open(self, file_name):
        """Open a file in the archive for reading as a binary file object."""
        self._require_mode("r")
        return self._zipfile.open(self._getinfo(file_name), mode="r")

    def header_file_name(self):
        """Find the header file of the record in the archive.

        The header is expected to be named after the archive (for
        example 'record.hea' in 'record.wfdb'). If there is no such
        file but the archive contains exactly one header file, that
        file is used.

        Returns
        -------
        str
            The name of the header file in the archive.

        """
        file_name = self.record_name + ".hea"
        if not self.exists(file_name):
            headers = [n for n in self.namelist() if n.endswith(".hea")]
            if len(headers) != 1:
                raise FileNotFoundError(
                    f"Could not find '{file_name}' in archive '{self.path}'"
                    + (
                        f" (it contains several header files: {headers})"
                        if headers
                        else ""
                    )
                )
            file_name = headers[0]
        return file_name

    def read_header(self):
        """Return the content of the record's header file.

        See `header_file_name` for how the header file is found.

        Returns
        -------
        str
            The content of the header file.

        """
        file_name = self.header_file_name()
        if self.file_size(file_name) > MAX_HEADER_SIZE:
            raise ValueError(
                f"Header file '{file_name}' in archive '{self.path}' is "
                f"larger than {MAX_HEADER_SIZE} bytes"
            )
        with self.open(file_name) as f:
            # Don't trust the size stored in the archive
            content = f.read(MAX_HEADER_SIZE + 1)
        if len(content) > MAX_HEADER_SIZE:
            raise ValueError(
                f"Header file '{file_name}' in archive '{self.path}' is "
                f"larger than {MAX_HEADER_SIZE} bytes"
            )
        return content.decode("ascii", errors="ignore")

    def write(self, file_name, data):
        """Add a file to an archive opened for writing.

        Parameters
        ----------
        file_name : str
            The name of the file in the archive.
        data : bytes
            The content of the file.

        """
        self._check_new_file(file_name)
        self._zipfile.writestr(file_name, data, compress_type=_compress_type(file_name))

    def add_file(self, path):
        """Copy a file from disk into an archive opened for writing.

        Parameters
        ----------
        path : str or os.PathLike
            The path of the file. It is stored under its base name.

        """
        file_name = os.path.basename(path)
        self._check_new_file(file_name)
        self._zipfile.write(
            path, arcname=file_name, compress_type=_compress_type(file_name)
        )

    def close(self):
        """Close the archive.

        In write mode, this moves the new archive into place.
        """
        if self._zipfile is None:
            return
        self._finalizer.detach()
        try:
            self._zipfile.close()
            if self._tmp_path is not None:
                # A new archive has the default permissions for new files. A
                # replaced archive keeps its permissions.
                try:
                    mode = stat.S_IMODE(os.stat(self.path).st_mode)
                except FileNotFoundError:
                    pass
                else:
                    os.chmod(self._tmp_path, mode)
                os.replace(self._tmp_path, self.path)
        except BaseException:
            _discard(None, self._tmp_path)
            raise
        finally:
            self._zipfile = None
            self._tmp_path = None

    def abort(self):
        """Close the archive without saving it.

        In read mode, this is the same as `close`.
        """
        if self._zipfile is None:
            return
        self._finalizer()
        self._zipfile = None
        self._tmp_path = None

    def _getinfo(self, file_name):
        self._require_open()
        try:
            return self._zipfile.getinfo(file_name)
        except KeyError:
            raise FileNotFoundError(
                f"Could not find '{file_name}' in archive '{self.path}'"
            ) from None

    def _check_new_file(self, file_name):
        """Check that a file can be added to an archive opened for writing."""
        self._require_mode("w")
        if self.exists(file_name):
            raise ValueError(f"'{file_name}' was already written to the archive")
        if file_name.endswith(".hea"):
            headers = [n for n in self.namelist() if n.endswith(".hea")]
            if headers:
                raise ValueError(
                    f"Cannot add '{file_name}' to '{self.path}': an archive "
                    f"holds a single record, and it already contains the "
                    f"header '{headers[0]}'"
                )

    def _require_open(self):
        if self._zipfile is None:
            raise ValueError(f"Archive '{self.path}' is closed")

    def _require_mode(self, mode):
        self._require_open()
        if self.mode != mode:
            action = "read from" if mode == "r" else "written to"
            raise ValueError(
                f"Archive '{self.path}' was opened in mode '{self.mode}' "
                f"and cannot be {action}"
            )


def _discard(zip_file, tmp_path):
    """Close a zip file and remove its temporary file, if there is one.

    The temporary file is removed even if closing the zip file fails.
    """
    try:
        if zip_file is not None:
            zip_file.close()
    finally:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)


def _create_temp_file(path):
    """Create an empty temporary file next to `path`, and return its path.

    The file is named after `path`, with a random part. It is created with the default
    permissions for new files (the umask is applied by the operating system), unlike
    `tempfile.mkstemp`, which makes files readable only by their owner.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    for _ in range(100):
        tmp_path = os.path.join(
            directory, f".{os.path.basename(path)}.{secrets.token_hex(4)}.tmp"
        )
        try:
            fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return tmp_path
    raise FileExistsError(f"Could not create a temporary file next to '{path}'")


def _compress_type(file_name):
    """Return the compression type for a file in an archive.

    Header files are compressed, but signal files are stored as they are, so that
    reading part of a signal doesn't require decompressing all of it.
    """
    if file_name.endswith(".hea"):
        return zipfile.ZIP_DEFLATED
    return zipfile.ZIP_STORED


def create_archive(file_paths, archive_path):
    """Create a .wfdb archive from the files of a record on disk.

    Parameters
    ----------
    file_paths : list of str
        The paths of the files to add: the header file and its signal
        files. Each file is stored under its base name.
    archive_path : str or os.PathLike
        The path of the archive to create. An existing archive at this
        path is replaced.

    Examples
    --------
    >>> create_archive(["100.hea", "100.dat"], "100.wfdb")

    """
    with WFDBArchive(archive_path, mode="w") as archive:
        for path in file_paths:
            archive.add_file(path)


def is_archive_path(record_name):
    """Check whether a record name refers to a .wfdb archive.

    The extension is not case-sensitive, so 'record.WFDB' is an archive too.
    """
    return os.fspath(record_name).lower().endswith(ARCHIVE_EXTENSION)


@contextmanager
def archive_for_writing(wfdb_archive, write_dir=""):
    """Get a WFDBArchive to write a record to.

    Parameters
    ----------
    wfdb_archive : str, os.PathLike, WFDBArchive or None
        A path opens a new archive that is saved when the context exits,
        or discarded if an error occurs. A WFDBArchive opened in write
        mode is used as it is and left open. None gives None.
    write_dir : str, optional
        The directory that a relative path is relative to, as for the
        files of a record written without an archive.

    """
    if wfdb_archive is None:
        yield None
    elif isinstance(wfdb_archive, WFDBArchive):
        wfdb_archive._require_mode("w")
        yield wfdb_archive
    elif isinstance(wfdb_archive, (str, os.PathLike)):
        path = os.path.join(write_dir, wfdb_archive)
        with WFDBArchive(path, mode="w") as archive:
            yield archive
    else:
        raise TypeError(
            "wfdb_archive must be a path or a WFDBArchive opened in write mode"
        )
