import io
import os
import posixpath

import fsspec

from wfdb.io import download

# Cloud protocols
CLOUD_PROTOCOLS = ["az://", "azureml://", "s3://", "gs://", "memory://", "physionet://"]


def _fsspec_open(path, mode="rb", **kwargs):
    """Open a local path or cloud URI with `fsspec.open`.

    Files with a `physionet://` URI are read in parts as they are needed,
    with HTTP range requests, instead of being downloaded in full, and
    without progress bars.

    Parameters
    ----------
    path : str
        The local path or cloud URI.
    mode : str, optional
        The mode to open the file in.
    kwargs : dict, optional
        Passed to `fsspec.open`.

    Returns
    -------
    fsspec.core.OpenFile
        The file to open, as a context manager.

    """
    if os.fspath(path).startswith("physionet://"):
        kwargs = {**download.PHYSIONET_OPEN_OPTIONS, **kwargs}
    return fsspec.open(path, mode, **kwargs)


def _split_path(record_name):
    """Split a local path or cloud URI into its directory and base name.

    Unlike `os.path.split`, this keeps the protocol of a URI intact if the URI
    has no directory, so that "memory://100" gives "memory://" and "100".

    Parameters
    ----------
    record_name : str or os.PathLike
        The local path or cloud URI.

    Returns
    -------
    dir_name : str
        The local directory or cloud URI of the directory.
    base_name : str
        The last component of the path.

    """
    path = os.fspath(record_name)
    if any(path.startswith(proto) for proto in CLOUD_PROTOCOLS):
        protocol, separator, location = path.partition("://")
        dir_name, base_name = posixpath.split(location)
        return protocol + separator + dir_name, base_name
    return os.path.split(path)


def _open_input_file(path):
    """Open a local file, or a file at a cloud URI, to read in binary mode.

    A file at a cloud URI, such as a `physionet://` URI, is read into memory,
    so that it can be read, sought and sized like a local file.

    Parameters
    ----------
    path : str or os.PathLike
        The path of a local file, or a cloud URI.

    Returns
    -------
    file
        The open file.

    """
    path = os.fspath(path)
    if any(path.startswith(proto) for proto in CLOUD_PROTOCOLS):
        with _fsspec_open(path, "rb") as f:
            return io.BytesIO(f.read())
    return open(path, "rb")


def _join_path(dir_name, file_name):
    """Join a local directory or cloud URI with a file name.

    URIs always use forward slashes, regardless of the operating system.

    Parameters
    ----------
    dir_name : str
        The local directory or cloud URI.
    file_name : str
        The name of the file.

    Returns
    -------
    str
        The path or URI of the file.

    """
    if any(dir_name.startswith(proto) for proto in CLOUD_PROTOCOLS):
        return posixpath.join(dir_name, file_name)
    return os.path.join(dir_name, file_name)


def _open_file(
    pn_dir,
    file_name,
    mode="r",
    *,
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    check_access=False,
):
    """Open a data file as a random-access file object.

    See the documentation of `open` and `wfdb.io._url.openurl` for details
    about the `mode`, `buffering`, `encoding`, `errors`, and `newline`
    parameters.

    Parameters
    ----------
    pn_dir : str or None
        The PhysioNet database directory where the file is stored, or None
        if file_name is a local or cloud path.
    file_name : str
        The name of the file, either as a local filesystem path or cloud
        URL (if `pn_dir` is None) or a PhysioNet URL path
        (if `pn_dir` is a string.)
    mode : str, optional
        The standard I/O mode for the file ("r" by default).  If `pn_dir`
        is not None, this must be "r", "rt", or "rb".
    buffering : int, optional
        Buffering policy.
    encoding : str, optional
        Name of character encoding used in text mode.
    errors : str, optional
        Error handling strategy used in text mode.
    newline : str, optional
        Newline translation mode used in text mode.
    check_access : bool, optional
        If true, raise an exception immediately if the file does not
        exist or is not accessible.

    """
    if pn_dir is None:
        return _fsspec_open(
            file_name,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    else:
        # check to make sure a cloud path isn't being passed under pn_dir
        if any(pn_dir.startswith(proto) for proto in CLOUD_PROTOCOLS):
            raise ValueError(
                "Cloud paths should be passed under record_name, not under pn_dir"
            )

        return download._open_pn_file(
            pn_dir,
            file_name,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            check_access=check_access,
        )
