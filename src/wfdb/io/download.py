import functools
import json
import multiprocessing.dummy
import os
import posixpath
import time

import fsspec
import numpy as np

from wfdb.io import _url

# The PhysioNet index url
PN_INDEX_URL = "https://physionet.org/files/"
PN_CONTENT_URL = "https://physionet.org/content/"


class Config:
    """General class structure for the PhysioNet database.

    Attributes
    ----------
    N/A

    """

    db_index_url: str


# The configuration database index url. Uses PhysioNet index by default.
config = Config()
config.db_index_url = PN_INDEX_URL


def set_db_index_url(db_index_url=PN_INDEX_URL):
    """Set the database index url to a custom location for streaming.

    Parameters
    ----------
    db_index_url : str, optional
        The desired new database index url. Leave as default to reset
        to the PhysioNet index url.

    Returns
    -------
    N/A

    """
    config.db_index_url = db_index_url


# Options for opening physionet:// files: read only the parts of a file that
# are needed, with HTTP range requests, and show no progress bars
PHYSIONET_OPEN_OPTIONS = {"stream": True, "show_progress": False}


def _physionet_uri(pn_dir, file_name):
    """Return the physionet:// URI of a file in a PhysioNet database.

    Parameters
    ----------
    pn_dir : str
        The PhysioNet database directory, including its version, such as
        'mitdb/1.0.0'.
    file_name : str
        The name of the file in that directory.

    Returns
    -------
    str or None
        The URI, or None if the database index URL has been set to a server
        other than PhysioNet (with `set_db_index_url`).

    """
    if config.db_index_url.rstrip("/") != PN_INDEX_URL.rstrip("/"):
        return None
    return "physionet://" + posixpath.join(pn_dir, file_name)


def _open_pn_file(pn_dir, file_name, mode="rb", **kwargs):
    """Open a file in a PhysioNet database directory.

    Files on PhysioNet are read through the physionet package, which reads
    only the parts of a file that are needed, and reads restricted databases
    with the sign-in saved by `physionet login`. Files on another server, set
    with `set_db_index_url`, are read with `wfdb.io._url.openurl`.

    Parameters
    ----------
    pn_dir : str
        The PhysioNet database directory, including its version.
    file_name : str
        The name of the file in that directory.
    mode : str, optional
        The mode to open the file in: "rb" (the default) or "r".
    kwargs : dict, optional
        Passed to `wfdb.io._url.openurl` for another server. For PhysioNet,
        only `encoding`, `errors` and `newline` are used.

    Returns
    -------
    file
        The file to open, as a context manager.

    """
    uri = _physionet_uri(pn_dir, file_name)
    if uri is None:
        url = posixpath.join(config.db_index_url, pn_dir, file_name)
        return _url.openurl(url, mode, **kwargs)
    # Other options would make fsspec create a separate filesystem, without
    # the connection and lookups that are shared by every physionet:// file
    text_options = {
        name: kwargs[name]
        for name in ("encoding", "errors", "newline")
        if kwargs.get(name) is not None
    }
    return fsspec.open(uri, mode, **PHYSIONET_OPEN_OPTIONS, **text_options)


def _remote_file_size(url=None, file_name=None, pn_dir=None):
    """Get the remote file size in bytes.

    Parameters
    ----------
    url : str, optional
        The full url of the file. Use this option to explicitly
        state the full url.
    file_name : str, optional
        The base file name. Use this argument along with pn_dir if you
        want the full url to be constructed.
    pn_dir : str, optional
        The base file name. Use this argument along with file_name if
        you want the full url to be constructed.

    Returns
    -------
    remote_file_size : int
        Size of the file in bytes.

    """
    if file_name and pn_dir:
        with _open_pn_file(pn_dir, file_name) as f:
            return f.seek(0, os.SEEK_END)

    with _url.openurl(url, "rb") as f:
        remote_file_size = f.seek(0, os.SEEK_END)

    return remote_file_size


def _stream_header(file_name: str, pn_dir: str) -> str:
    """Stream the text of a remote header file.

    Parameters
    ----------
    file_name : str
        The name of the headerr file to be read.
    pn_dir : str
        The PhysioNet database directory from which to find the
        required header file. eg. For file '100.hea' in
        'http://physionet.org/content/mitdb', pn_dir='mitdb'.

    Returns
    -------
    N/A : str
        The text contained in the header file

    """
    # Get the content of the remote file
    with _open_pn_file(pn_dir, file_name) as f:
        content = f.read()

    return content.decode("iso-8859-1")


def _stream_dat(file_name, pn_dir, byte_count, start_byte, dtype):
    """Stream data from a remote dat file into a 1d numpy array.

    Parameters
    ----------
    file_name : str
        The name of the dat file to be read.
    pn_dir : str
        The PhysioNet directory where the dat file is located.
    byte_count : int
        The number of bytes to be read.
    start_byte : int
        The starting byte number to read from.
    dtype : str
        The numpy dtype to load the data into.

    Returns
    -------
    sig_data : ndarray
        The data read from the dat file.

    """
    # Get the content
    with _open_pn_file(pn_dir, file_name, buffering=0) as f:
        f.seek(start_byte)
        content = f.read(byte_count)

    # Convert to numpy array
    sig_data = np.frombuffer(content, dtype=dtype)

    return sig_data


def _stream_annotation(file_name, pn_dir):
    """Stream an entire remote annotation file from Physionet.

    Parameters
    ----------
    file_name : str
        The name of the annotation file to be read.
    pn_dir : str
        The PhysioNet directory where the annotation file is located.

    Returns
    -------
    ann_data : ndarray
        The resulting data stream in numpy array format.

    """
    # Get the content
    with _open_pn_file(pn_dir, file_name) as f:
        content = f.read()

    # Convert to numpy array
    ann_data = np.frombuffer(content, dtype=np.dtype("<u1"))

    return ann_data


def get_dbs():
    """Get a list of all the PhysioNet databases available.

    Parameters
    ----------
    N/A

    Returns
    -------
    dbs : list
        All of the databases currently available for analysis.

    Examples
    --------
    >>> dbs = wfdb.get_dbs()
    >>> dbs
    [
     ['aami-ec13', 'ANSI/AAMI EC13 Test Waveforms'],
     ['adfecgdb', 'Abdominal and Direct Fetal ECG Database'],
     ...
     ['wrist', 'Wrist PPG During Exercise']
    ]

    """
    with _url.openurl("https://physionet.org/rest/database-list/", "rb") as f:
        content = f.read()
    dbs = json.loads(content)
    dbs = [[d["slug"], d["title"]] for d in dbs]
    dbs.sort()

    return dbs


# ---- Helper functions for downloading PhysioNet files ------- #


def get_version(pn_dir):
    """Get the version number of the desired project.

    Parameters
    ----------
    pn_dir : str
        The PhysioNet database directory from which to find the
        required version number. eg. For the project 'mitdb' in
        'http://physionet.org/content/mitdb', pn_dir='mitdb'.

    Returns
    -------
    version_number : str
        The version number of the most recent database. It is looked up
        once per database, and remembered until the program ends.

    """
    return _latest_version(pn_dir.split("/")[0])


# How many times to try looking up a version, if PhysioNet is unavailable
VERSION_ATTEMPTS = 3


@functools.cache
def _latest_version(db_dir):
    """Look up the latest version of a PhysioNet database.

    Connection errors, server errors and rate limiting are retried, after 1
    and then 2 seconds.

    """
    import physionet
    from physionet.api.exceptions import NotFoundError, PhysioNetAPIError

    for attempt in range(VERSION_ATTEMPTS):
        try:
            return physionet.latest_version(db_dir)
        except NotFoundError as exc:
            raise FileNotFoundError(f"PhysioNet has no database {db_dir!r}") from exc
        except (ConnectionError, PhysioNetAPIError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            unavailable = isinstance(exc, ConnectionError) or (
                status is not None and (status >= 500 or status == 429)
            )
            if not unavailable or attempt == VERSION_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)


def get_record_list(db_dir, records="all"):
    """Get a list of records belonging to a database.

    Parameters
    ----------
    db_dir : str
        The database directory, usually the same as the database slug.
        The location to look for a RECORDS file.
    records : list, optional
        An option used when this function acts as a helper function.
        Leave as default 'all' to get all records.

    Returns
    -------
    record_list : list
        All of the possible record names for the input database.

    Examples
    --------
    >>> wfdb.get_record_list('mitdb')

    """
    if "/" not in db_dir:
        db_dir = posixpath.join(db_dir, get_version(db_dir))

    # Check for a RECORDS file
    if records == "all":
        try:
            with _open_pn_file(db_dir, "RECORDS") as f:
                content = f.read()
        except FileNotFoundError:
            db_url = posixpath.join(config.db_index_url, db_dir)
            raise ValueError("The database %s has no WFDB files to download" % db_url)

        # Get each line as a string
        record_list = content.decode("ascii").splitlines()
    # Otherwise the records are input manually
    else:
        record_list = records

    return record_list


def get_annotators(db_dir, annotators):
    """Get a list of annotators belonging to a database.

    Parameters
    ----------
    db_dir : str
        The database directory, usually the same as the database slug.
        The location to look for a ANNOTATORS file.
    annotators : list, str
        Determines from which records to get the annotators from. Leave as
        default 'all' to get all annotators.

    Returns
    -------
    annotators : list
        All of the possible annotators for the input database.

    Examples
    --------
    >>> wfdb.get_annotators('mitdb')

    """
    # Full url PhysioNet database
    db_url = posixpath.join(config.db_index_url, db_dir)

    if annotators is not None:
        # Check for an ANNOTATORS file
        try:
            with _open_pn_file(db_dir, "ANNOTATORS") as f:
                content = f.read()
        except FileNotFoundError:
            if annotators == "all":
                return
            else:
                raise ValueError(
                    "The database %s has no annotation files to download" % db_url
                )

        # Make sure the input annotators are present in the database
        ann_list = content.decode("ascii").splitlines()
        ann_list = [a.split("\t")[0] for a in ann_list]

        # Get the annotation file types required
        if annotators == "all":
            # all possible ones
            annotators = ann_list
        else:
            # In case they didn't input a list
            if type(annotators) is str:
                annotators = [annotators]
            # user input ones. Check validity.
            for a in annotators:
                if a not in ann_list:
                    raise ValueError(
                        "The database contains no annotators with extension: %s" % a
                    )

    return annotators


def _project_files(db_dir):
    """List the files in a PhysioNet database directory, from its checksum list.

    Parameters
    ----------
    db_dir : str
        The PhysioNet database directory, including its version, such as
        'mitdb/1.0.0'.

    Returns
    -------
    set of str or None
        The paths of the files, relative to `db_dir`, or None if the database
        index URL has been set to a server other than PhysioNet.

    """
    if _physionet_uri(db_dir, "") is None:
        return None
    fs = fsspec.filesystem("physionet", **PHYSIONET_OPEN_OPTIONS)
    prefix = db_dir.rstrip("/") + "/"
    return {path[len(prefix) :] for path in fs.find(db_dir) if path.startswith(prefix)}


def _download_files(db_dir, dl_dir, files, keep_subdirs, overwrite):
    """Download files from a PhysioNet database with the physionet package.

    Each file is checked against the database's checksum list. A file that is
    already saved with the right content is skipped, and one that was partly
    downloaded is completed.

    Parameters
    ----------
    db_dir : str
        The PhysioNet database directory, including its version, such as
        'mitdb/1.0.0' or 'mimic3wdb/1.0/30'.
    dl_dir : str
        The local directory to save the files in.
    files : list of str
        The paths of the files to download, relative to `db_dir`.
    keep_subdirs : bool
        Whether to save the files in the same subdirectories as on PhysioNet
        (True), or all in `dl_dir` (False).
    overwrite : bool
        Whether to download files again even if they are already saved.

    Returns
    -------
    N/A

    """
    import physionet

    slug, version, *subdirs = db_dir.strip("/").split("/")
    saved_as = {}
    for file in files:
        path = posixpath.join(*subdirs, file)
        saved_as[path] = file if keep_subdirs else posixpath.basename(file)

    if not os.path.isdir(dl_dir):
        os.makedirs(dl_dir)
        print("Created local base download directory: %s" % dl_dir)
    # Two files at a time, as before, to avoid overloading the server. With
    # overwrite, physionet replaces each saved file only after checking every
    # requested file and destination, and once the new copy is complete.
    physionet.download(
        slug,
        version,
        output_dir=dl_dir,
        files=saved_as,
        verbose=False,
        show_progress=False,
        jobs=2,
        overwrite=overwrite,
    )


def make_local_dirs(dl_dir, dl_inputs, keep_subdirs):
    """Make any required local directories to prepare for downloading.

    Parameters
    ----------
    dl_dir : str
        The full local directory path in which to download the files.
    dl_inputs : list
        The desired input names for creating the directories.
    keep_subdirs : bool
        Whether to keep the relative subdirectories of downloaded files as they
        are organized in PhysioNet (True), or to download all files into the
        same base directory (False).

    Returns
    -------
    N/A

    """
    # Make the local download dir if it doesn't exist
    if not os.path.isdir(dl_dir):
        os.makedirs(dl_dir)
        print("Created local base download directory: %s" % dl_dir)
    # Create all required local subdirectories
    # This must be out of dl_pn_file to
    # avoid clash in multiprocessing
    if keep_subdirs:
        dl_dirs = set([os.path.join(dl_dir, d[1]) for d in dl_inputs])
        for d in dl_dirs:
            if not os.path.isdir(d):
                os.makedirs(d)
    return


def dl_pn_file(inputs):
    """Download a file from Physionet.

    The input args are to be unpacked for the use of multiprocessing map, because
    python2 doesn't have starmap.

    Parameters
    ----------
    inputs : list
        All of the required information needed to download a file
        from Physionet:
        [basefile, subdir, db, dl_dir, keep_subdirs, overwrite].

    Returns
    -------
    N/A

    """
    basefile, subdir, db, dl_dir, keep_subdirs, overwrite = inputs

    # Full url of file
    url = posixpath.join(config.db_index_url, db, subdir, basefile)

    # Figure out where the file should be locally
    if keep_subdirs:
        dldir = os.path.join(dl_dir, subdir)
    else:
        dldir = dl_dir

    local_file = os.path.join(dldir, basefile)

    # The file exists locally.
    if os.path.isfile(local_file):
        # Redownload regardless
        if overwrite:
            dl_full_file(url, local_file)
        # Process accordingly.
        else:
            local_file_size = os.path.getsize(local_file)
            with _url.openurl(url, "rb") as f:
                remote_file_size = f.seek(0, os.SEEK_END)
                # Local file is smaller than it should be. Append it.
                if local_file_size < remote_file_size:
                    print(
                        "Detected partially downloaded file: %s Appending file..."
                        % local_file
                    )
                    f.seek(local_file_size, os.SEEK_SET)
                    with open(local_file, "ba") as writefile:
                        writefile.write(f.read())
                    print("Done appending.")
                # Local file is larger than it should be. Redownload.
                elif local_file_size > remote_file_size:
                    dl_full_file(url, local_file)
                # If they're the same size, do nothing.

    # The file doesn't exist. Download it.
    else:
        dl_full_file(url, local_file)

    return


def dl_full_file(url, save_file_name):
    """Download a file. No checks are performed.

    Parameters
    ----------
    url : str
        The url of the file to download.
    save_file_name : str
        The name to save the file as.

    Returns
    -------
    N/A

    """
    with _url.openurl(url, "rb") as readfile:
        content = readfile.read()
    with open(save_file_name, "wb") as writefile:
        writefile.write(content)

    return


def dl_files(db, dl_dir, files, keep_subdirs=True, overwrite=False):
    """Download specified files from a PhysioNet database.

    Parameters
    ----------
    db : str
        The PhysioNet database directory to download. eg. For database:
        'http://physionet.org/content/mitdb', db='mitdb'.
    dl_dir : str
        The full local directory path in which to download the files.
    files : list
        A list of strings specifying the file names to download relative to the
        database base directory.
    keep_subdirs : bool, optional
        Whether to keep the relative subdirectories of downloaded files as they
        are organized in PhysioNet (True), or to download all files into the
        same base directory (False).
    overwrite : bool, optional
        If True, all files are downloaded again. If False, a file that is
        already saved with the right content (checked against the database's
        checksum list) is skipped, a partly downloaded file is completed, and
        any other saved file is downloaded again. A saved file is only
        replaced once its new copy is complete and checked.

    Returns
    -------
    N/A

    Notes
    -----
    Files are downloaded with the `physionet` package, two at a time, from the
    AWS open data mirror for open databases, and from PhysioNet for restricted
    ones, which can be downloaded after signing in with `physionet login`.
    Each file is checked against the database's checksum list. If the database
    index URL has been set to another server with `set_db_index_url`, files are
    downloaded from that server instead, comparing only their sizes when they
    are already saved.

    Examples
    --------
    >>> wfdb.dl_files('ahadb', os.getcwd(),
                      ['STAFF-Studies-bibliography-2016.pdf', 'data/001a.hea',
                      'data/001a.dat'])

    """
    # Full url PhysioNet database
    db_dir = posixpath.join(db, get_version(db))

    # The physionet package reports a missing database, version or file
    # before downloading anything
    if _physionet_uri(db_dir, "") is not None:
        print("Downloading files...")
        _download_files(db_dir, dl_dir, files, keep_subdirs, overwrite)
        print("Finished downloading files")
        return

    # Check if the database is valid
    db_url = posixpath.join(PN_CONTENT_URL, db_dir) + "/"
    _url.openurl(db_url, check_access=True)

    # Construct the urls to download
    dl_inputs = [
        (
            os.path.split(file)[1],
            os.path.split(file)[0],
            db_dir,
            dl_dir,
            keep_subdirs,
            overwrite,
        )
        for file in files
    ]

    # Make any required local directories
    make_local_dirs(dl_dir, dl_inputs, keep_subdirs)

    print("Downloading files...")
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    with multiprocessing.dummy.Pool(processes=2) as pool:
        pool.map(dl_pn_file, dl_inputs)
    print("Finished downloading files")
