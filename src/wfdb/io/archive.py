"""Support for .wfdb archive files.

A .wfdb archive is a ZIP file containing the files of a single WFDB
record: its header file, the signal files listed in the header, and
optionally annotation files. The archive of a multi-segment record also
contains the header and signal files of its segments.
Files are stored without compression, so that a range of samples can
be read without decompressing the whole signal file, and so that
reading an archive doesn't require a decompression library. Archives
made by other tools may compress their files, and can still be read.
"""

import contextlib
import hashlib
import os
import re
import secrets
import stat
import time
import weakref
import zipfile
from contextlib import contextmanager

from wfdb.io import header

ARCHIVE_EXTENSION = ".wfdb"

# Largest header file that will be read from an archive. Headers are
# small, so anything bigger is treated as a damaged or malicious file.
MAX_HEADER_SIZE = 1024 * 1024

# The name of a file at the top level of an archive: an ASCII WFDB name made of
# letters, digits, hyphens and underscores, with an extension
_FILE_NAME = re.compile(r"[-\w]+(\.[-\w]+)*", re.ASCII)

# The name of a header file at the top level of an archive
_HEADER_NAME = re.compile(r"[-\w]+\.hea", re.ASCII)


class WFDBArchive:
    """A .wfdb archive opened for reading or writing.

    In read mode the archive must already exist. In write mode a new
    archive is built in a temporary file next to the target path and
    only moved into place when the archive is closed without an error,
    so an existing archive is either fully replaced or left untouched.
    The new archive gets the permissions of the archive it replaces, or
    the default permissions for new files.

    An archive holds a single record. It can contain several header
    files only if the record is a multi-segment record: then the other
    headers must be the segments (or layout header) listed in the main
    header. The main header must be named after the archive (for example
    'record.hea' in 'record.wfdb'). This is checked when the archive is
    closed. Annotation files and other files can be added alongside the
    record's files. All files are at the top level of the archive, and
    their names are made of ASCII letters, digits, hyphens and
    underscores, with an extension.

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
        # In write mode: the content of the header files written, and a digest of
        # every file written
        self._headers = {}
        self._digests = {}

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
        self._require_open()
        try:
            self._zipfile.getinfo(file_name)
        except KeyError:
            return False
        return True

    def file_size(self, file_name):
        """Return the uncompressed size of a file in the archive, in bytes."""
        return self._getinfo(file_name).file_size

    def open(self, file_name):
        """Open a file in the archive for reading as a binary file object."""
        self._require_mode("r")
        return self._zipfile.open(self._getinfo(file_name), mode="r")

    def header_file_name(self):
        """Find the main header file of the record in the archive.

        The header is expected to be named after the archive (for
        example 'record.hea' in 'record.wfdb'). If there is no such
        file, the header used is the only header file in the archive,
        or, for a multi-segment record, the header that lists all the
        other headers as its segments.

        Returns
        -------
        str
            The name of the header file in the archive.

        """
        file_name = self.record_name + ".hea"
        if self.exists(file_name):
            return file_name

        headers = self._header_names()
        if len(headers) == 1:
            return headers[0]
        main_header = None
        if headers:
            main_header = _main_header(
                {name: self._read_header_file(name) for name in headers}
            )
        if main_header is None:
            raise FileNotFoundError(
                f"Could not find '{file_name}' in archive '{self.path}'"
                + (f" (it contains several header files: {headers})" if headers else "")
            )
        return main_header

    def read_header(self, record_name=None):
        """Return the content of a header file in the archive.

        Parameters
        ----------
        record_name : str, optional
            The name of the record (or segment) whose header to read. By
            default, the main header of the archive's record is read; see
            `header_file_name` for how it is found.

        Returns
        -------
        str
            The content of the header file.

        """
        if record_name is None:
            file_name = self.header_file_name()
        else:
            file_name = record_name + ".hea"
        return self._read_header_file(file_name)

    def _read_header_file(self, file_name):
        """Read a header file from the archive, refusing very large files."""
        with self.open(file_name) as f:
            # Read at most one byte more than allowed, whatever size the archive
            # declares, so that a damaged or malicious header isn't read in full
            content = f.read(MAX_HEADER_SIZE + 1)
        if len(content) > MAX_HEADER_SIZE:
            raise ValueError(
                f"Header file '{file_name}' in archive '{self.path}' is "
                f"larger than {MAX_HEADER_SIZE} bytes"
            )
        return content.decode("ascii", errors="ignore")

    def write(self, file_name, data):
        """Add a file to an archive opened for writing.

        The data is streamed into the archive, so a large signal is not copied in
        memory. Writing a file that is already in the archive is allowed only if the
        content is the same (segments of a multi-segment record can share a signal
        file), and then the file is not written again.

        Parameters
        ----------
        file_name : str
            The name of the file in the archive.
        data : bytes-like object
            The content of the file, such as `bytes`, or a `memoryview` of a
            contiguous array.

        """
        self._require_mode("w")
        data = memoryview(data).cast("B")
        if not self._register_file(file_name, hashlib.sha256(data).digest()):
            return
        if _HEADER_NAME.fullmatch(file_name):
            self._headers[file_name] = bytes(data).decode("ascii", errors="ignore")
        # The same file information that ZipFile.writestr uses
        info = zipfile.ZipInfo(file_name, date_time=time.localtime()[:6])
        info.external_attr = 0o600 << 16
        # ZIP64 is needed for files of 2 GiB or more, and has to be chosen up front
        with self._zipfile.open(
            info, mode="w", force_zip64=data.nbytes >= zipfile.ZIP64_LIMIT
        ) as f:
            f.write(data)

    def add_file(self, path):
        """Copy a file from disk into an archive opened for writing.

        As with `write`, a file that is already in the archive is only allowed if the
        content is the same.

        Parameters
        ----------
        path : str or os.PathLike
            The path of the file. It is stored under its base name.

        """
        self._require_mode("w")
        file_name = os.path.basename(path)
        if self._register_file(file_name, _file_digest(path)):
            if _HEADER_NAME.fullmatch(file_name):
                with open(path, encoding="ascii", errors="ignore") as f:
                    self._headers[file_name] = f.read()
            self._zipfile.write(path, arcname=file_name)

    def close(self):
        """Close the archive.

        In write mode, this checks that the archive holds a single record, and moves the
        new archive into place. If the check fails, the archive is not saved.
        """
        if self._zipfile is None:
            return
        if self.mode == "w":
            try:
                self._check_single_record()
            except BaseException:
                self.abort()
                raise
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
            _discard(self._zipfile, self._tmp_path)
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

    def _register_file(self, file_name, digest):
        """Record that a file is about to be written.

        Return False if the same content was already written under that name.
        """
        if not _FILE_NAME.fullmatch(file_name):
            raise ValueError(
                f"Cannot add '{file_name}' to '{self.path}': files in an archive "
                "are at the top level, and their names are made of ASCII letters, "
                "digits, hyphens and underscores, with an extension"
            )
        if file_name in self._digests:
            if self._digests[file_name] != digest:
                raise ValueError(
                    f"'{file_name}' was already written to the archive "
                    "with different content"
                )
            return False
        self._digests[file_name] = digest
        return True

    def _check_single_record(self):
        """Check that the header files written make up a single record.

        This is one header, or a multi-segment header and the segments it lists.
        """
        if not self._headers:
            return
        main_header = _main_header(self._headers)
        if main_header is None:
            raise ValueError(
                f"Cannot save '{self.path}': an archive holds a single "
                f"record, but the header files {sorted(self._headers)} are "
                "not one multi-segment record and its segments"
            )
        expected = self.record_name + ".hea"
        if main_header != expected:
            raise ValueError(
                f"Cannot save '{self.path}': the header file of the record in an "
                f"archive must be named after the archive ('{expected}'), but it "
                f"is '{main_header}'"
            )

    def _header_names(self):
        """Return the names of the header files in the archive.

        Only files at the top level that are named after a valid record name count,
        so metadata files that macOS adds, such as '__MACOSX/._100.hea' (added by
        Finder when it zips '100.hea') or '._100.hea' (written next to '100.hea' on
        some drives), are ignored.
        """
        return [n for n in self.namelist() if _HEADER_NAME.fullmatch(n)]

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


def _segment_header_names(header_content):
    """Return the header file names of the segments listed in a header.

    This includes the layout header of a multi-segment header. A single-segment header
    lists no segments.
    """
    header_lines, _ = header.parse_header_content(header_content)
    if not header_lines or "/" not in header_lines[0].split()[0]:
        return set()
    names = {line.split()[0] for line in header_lines[1:]}
    names.discard("~")
    return {name + ".hea" for name in names}


def _main_header(headers):
    """Find the main header of a record, given the content of all its headers.

    Parameters
    ----------
    headers : dict
        The content of each header file, keyed by file name.

    Returns
    -------
    str or None
        The name of the only header, or of the multi-segment header that
        lists all the other headers as its segments. None if there is no
        such header.

    """
    if len(headers) == 1:
        return next(iter(headers))
    names = set(headers)
    for name, content in headers.items():
        segments = _segment_header_names(content)
        # Most headers list no segments, so skip them without building a set of
        # the other names (which would make this quadratic in the number of headers)
        if len(segments) >= len(names) - 1 and names - {name} <= segments:
            return name
    return None


def _file_digest(path):
    """Return the SHA-256 digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


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


def create_archive(file_paths, archive_path):
    """Create a .wfdb archive from the files of a record on disk.

    Parameters
    ----------
    file_paths : list of str
        The paths of the files to add: the header file and its signal
        files. Each file is stored under its base name.
    archive_path : str or os.PathLike
        The path of the archive to create, which must be named after the
        record. An existing archive at this path is replaced.

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
