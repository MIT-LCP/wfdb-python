import io
import os
import shutil
import zipfile
from contextlib import contextmanager

_archive_cache = {}


class WFDBArchive:
    """
    Helper class for working with WFDB .wfdb ZIP archives.

    If used for reading, the archive must already exist.
    If used for writing, use mode='w' and call `write(...)` or `create_archive(...)`.

    Used only if:
      - .wfdb is included in the record_name explicitly, or
      - .wfdb is passed directly to the file loading function.
    """

    def __init__(self, record_name, mode="r"):
        """
        Initialize a WFDBArchive for a given record name (without extension).

        Parameters
        ----------
        record_name : str
            The base name of the archive, without the .wfdb extension.
        mode : str
            'r' for read (default), 'w' for write.
        """
        self.record_name = record_name
        # Only append .wfdb if it's not already there
        if record_name.endswith(".wfdb"):
            self.archive_path = record_name
        else:
            self.archive_path = f"{record_name}.wfdb"
        self.zipfile = None
        self.mode = mode

        if mode == "r":
            if not os.path.exists(self.archive_path):
                raise FileNotFoundError(
                    f"Archive not found: {self.archive_path}"
                )
            if not zipfile.is_zipfile(self.archive_path):
                raise ValueError(f"Invalid WFDB archive: {self.archive_path}")
            self.zipfile = zipfile.ZipFile(self.archive_path, mode="r")

        elif mode == "w":
            # Create archive file if needed
            if not os.path.exists(self.archive_path):
                # Create the directory if it doesn't exist
                os.makedirs(
                    os.path.dirname(os.path.abspath(self.archive_path)),
                    exist_ok=True,
                )
                WFDBArchive.make_archive_file([], self.archive_path)
            self.zipfile = zipfile.ZipFile(self.archive_path, mode="a")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def exists(self, filename):
        """
        Check if a file exists in the archive.
        """
        return self.zipfile and filename in self.zipfile.namelist()

    @staticmethod
    def make_archive_file(file_list, output_path):
        with zipfile.ZipFile(output_path, mode="w") as zf:
            for file in file_list:
                compress = zipfile.ZIP_DEFLATED
                zf.write(
                    file, arcname=os.path.basename(file), compress_type=compress
                )

    @contextmanager
    def open(self, filename, mode="r"):
        """
        Open a file, either from disk or from the archive.
        Mode 'r' (text) or 'rb' (binary) supported.
        """
        if self.zipfile and filename in self.zipfile.namelist():
            with self.zipfile.open(filename, "r") as f:
                if "b" in mode:
                    yield f
                else:
                    yield io.TextIOWrapper(f)
        else:
            raise FileNotFoundError(
                f"Could not find '{filename}' as loose file or inside "
                f"'{self.archive_path}'."
            )

    def close(self):
        """
        Close the archive if open.
        """
        if self.zipfile:
            self.zipfile.close()

    def write(self, filename, data):
        """
        Write binary data to the archive (replaces if already exists).
        """
        if self.zipfile is None:
            self.zipfile = zipfile.ZipFile(self.archive_path, mode="w")
            self.zipfile.writestr(filename, data)
            return

        # If already opened in read or append mode, use replace-then-move
        tmp_path = self.archive_path + ".tmp"
        with zipfile.ZipFile(self.archive_path, mode="r") as zin:
            with zipfile.ZipFile(tmp_path, mode="w") as zout:
                for item in zin.infolist():
                    if item.filename != filename:
                        zout.writestr(item, zin.read(item.filename))
                zout.writestr(filename, data)
        shutil.move(tmp_path, self.archive_path)
        self.zipfile = zipfile.ZipFile(self.archive_path, mode="a")

    def create_archive(self, file_list, output_path=None):
        """
        Create a .wfdb archive containing the specified list of files.
        If output_path is not specified, uses self.archive_path.
        """
        output_path = output_path or self.archive_path
        WFDBArchive.make_archive_file(file_list, output_path)

        # If this archive object points to the archive, reload the zipfile in append mode
        if output_path == self.archive_path:
            if self.zipfile:
                self.zipfile.close()
            self.zipfile = zipfile.ZipFile(self.archive_path, mode="a")


def get_archive(record_base_name, mode="r"):
    """
    Get or create a WFDBArchive for the given record base name.
    """
    if record_base_name not in _archive_cache:
        _archive_cache[record_base_name] = WFDBArchive(
            record_base_name, mode=mode
        )
    return _archive_cache[record_base_name]
