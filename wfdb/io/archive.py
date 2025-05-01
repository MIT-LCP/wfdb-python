import os
import zipfile
from contextlib import contextmanager

_archive_cache = {}


class WFDBArchive:
    """
    Helper class for working with WFDB .wfdb ZIP archives.

    Used only if:
      - .wfdb is included in the record_name explicitly, or
      - .wfdb is passed directly to the file loading function.
    """
    def __init__(self, record_name):
        """
        Initialize a WFDBArchive for a given record name (without extension).

        record_name : str
          The base name of the archive, without the .wfdb extension.
        """
        self.record_name = record_name
        self.archive_path = f"{record_name}.wfdb"

        if not os.path.exists(self.archive_path):
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")
        if not zipfile.is_zipfile(self.archive_path):
            raise ValueError(f"Invalid WFDB archive: {self.archive_path}")
        self.zipfile = zipfile.ZipFile(self.archive_path, mode="r")

    def exists(self, filename):
        """
        Check if a file exists in the archive.
        """
        return self.zipfile and filename in self.zipfile.namelist()

    @contextmanager
    def open(self, filename, mode="r"):
        """
        Open a file, either from disk or from the archive.
        Mode 'r' (text) or 'rb' (binary) supported.
        """
        if self.zipfile and filename in self.zipfile.namelist():
            with self.zipfile.open(filename, 'r') as f:
                if "b" in mode:
                    yield f
                else:
                    import io
                    yield io.TextIOWrapper(f)
        else:
            raise FileNotFoundError(
                f"Could not find '{filename}' as loose file or inside '{self.archive_path}'."
                )

    def close(self):
        """
        Close the archive if open.
        """
        if self.zipfile:
            self.zipfile.close()

    def create_archive(self, file_list, output_path=None):
        """
        Create a .wfdb archive containing the specified list of files.
        If output_path is not specified, uses self.archive_path.
        """
        output_path = output_path or self.archive_path
        with zipfile.ZipFile(output_path, mode="w") as zf:
            for file in file_list:
                compress = (
                    zipfile.ZIP_STORED
                    if file.endswith((".hea", ".hea.json", ".hea.yml"))
                    else zipfile.ZIP_DEFLATED
                )
                zf.write(file, arcname=os.path.basename(file), compress_type=compress)


def get_archive(record_base_name):
    """
    Get or create a WFDBArchive for the given record base name.
    """
    if record_base_name not in _archive_cache:
        _archive_cache[record_base_name] = WFDBArchive(record_base_name)
    return _archive_cache[record_base_name]
