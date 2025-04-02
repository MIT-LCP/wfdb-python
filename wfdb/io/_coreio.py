import posixpath

import fsspec

from wfdb.io import _url
from wfdb.io.download import config


# Cloud protocols
CLOUD_PROTOCOLS = ["az://", "azureml://", "s3://", "gs://"]


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
    """
    Open a data file as a random-access file object.

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
        return fsspec.open(
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

        url = posixpath.join(config.db_index_url, pn_dir, file_name)
        return _url.openurl(
            url,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            check_access=check_access,
        )
