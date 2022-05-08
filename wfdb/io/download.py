import json
import multiprocessing.dummy
import os
import posixpath

import numpy as np

from wfdb.io import _url


# The PhysioNet index url
PN_INDEX_URL = "https://physionet.org/files/"
PN_CONTENT_URL = "https://physionet.org/content/"


class Config(object):
    """
    General class structure for the PhysioNet database.

    Attributes
    ----------
    N/A

    """

    pass


# The configuration database index url. Uses PhysioNet index by default.
config = Config()
config.db_index_url = PN_INDEX_URL


def set_db_index_url(db_index_url=PN_INDEX_URL):
    """
    Set the database index url to a custom value, to stream remote
    files from another location.

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


def _remote_file_size(url=None, file_name=None, pn_dir=None):
    """
    Get the remote file size in bytes.

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
    # Option to construct the url
    if file_name and pn_dir:
        url = posixpath.join(config.db_index_url, pn_dir, file_name)

    with _url.openurl(url, "rb") as f:
        remote_file_size = f.seek(0, os.SEEK_END)

    return remote_file_size


def _stream_header(file_name: str, pn_dir: str) -> str:
    """
    Stream the text of a remote header file.

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
    # Full url of header location
    url = posixpath.join(config.db_index_url, pn_dir, file_name)

    # Get the content of the remote file
    with _url.openurl(url, "rb") as f:
        content = f.read()

    return content.decode("iso-8859-1")


def _stream_dat(file_name, pn_dir, byte_count, start_byte, dtype):
    """
    Stream data from a remote dat file into a 1d numpy array.

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
    # Full url of dat file
    url = posixpath.join(config.db_index_url, pn_dir, file_name)

    # Get the content
    with _url.openurl(url, "rb", buffering=0) as f:
        f.seek(start_byte)
        content = f.read(byte_count)

    # Convert to numpy array
    sig_data = np.fromstring(content, dtype=dtype)

    return sig_data


def _stream_annotation(file_name, pn_dir):
    """
    Stream an entire remote annotation file from Physionet.

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
    # Full url of annotation file
    url = posixpath.join(config.db_index_url, pn_dir, file_name)

    # Get the content
    with _url.openurl(url, "rb") as f:
        content = f.read()

    # Convert to numpy array
    ann_data = np.fromstring(content, dtype=np.dtype("<u1"))

    return ann_data


def get_dbs():
    """
    Get a list of all the PhysioNet databases available.

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
    """
    Get the version number of the desired project.

    Parameters
    ----------
    pn_dir : str
        The PhysioNet database directory from which to find the
        required version number. eg. For the project 'mitdb' in
        'http://physionet.org/content/mitdb', pn_dir='mitdb'.

    Returns
    -------
    version_number : str
        The version number of the most recent database.

    """
    db_dir = pn_dir.split("/")[0]
    url = posixpath.join(PN_CONTENT_URL, db_dir) + "/"
    with _url.openurl(url, "rb") as f:
        content = f.read()
    contents = [line.decode("utf-8").strip() for line in content.splitlines()]
    version_number = [v for v in contents if "Version:" in v]
    version_number = version_number[0].split(":")[-1].strip().split("<")[0]

    return version_number


def get_record_list(db_dir, records="all"):
    """
    Get a list of records belonging to a database.

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
    # Full url PhysioNet database
    if "/" not in db_dir:
        db_url = posixpath.join(
            config.db_index_url, db_dir, get_version(db_dir)
        )
    else:
        db_url = posixpath.join(config.db_index_url, db_dir)

    # Check for a RECORDS file
    if records == "all":
        try:
            with _url.openurl(posixpath.join(db_url, "RECORDS"), "rb") as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(
                "The database %s has no WFDB files to download" % db_url
            )

        # Get each line as a string
        record_list = content.decode("ascii").splitlines()
    # Otherwise the records are input manually
    else:
        record_list = records

    return record_list


def get_annotators(db_dir, annotators):
    """
    Get a list of annotators belonging to a database.

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
            with _url.openurl(posixpath.join(db_url, "ANNOTATORS"), "rb") as f:
                content = f.read()
        except FileNotFoundError:
            if annotators == "all":
                return
            else:
                raise ValueError(
                    "The database %s has no annotation files to download"
                    % db_url
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
            if type(annotators) == str:
                annotators = [annotators]
            # user input ones. Check validity.
            for a in annotators:
                if a not in ann_list:
                    raise ValueError(
                        "The database contains no annotators with extension: %s"
                        % a
                    )

    return annotators


def make_local_dirs(dl_dir, dl_inputs, keep_subdirs):
    """
    Make any required local directories to prepare for downloading.

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
    """
    Download a file from Physionet. The input args are to be unpacked
    for the use of multiprocessing map, because python2 doesn't have starmap.

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
    """
    Download a file. No checks are performed.

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
    """
    Download specified files from a PhysioNet database.

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
        If True, all files will be redownloaded regardless. If False, existing
        files with the same name and relative subdirectory will be checked.
        If the local file is the same size as the online file, the download is
        skipped. If the local file is larger, it will be deleted and the file
        will be redownloaded. If the local file is smaller, the file will be
        assumed to be partially downloaded and the remaining bytes will be
        downloaded and appended.

    Returns
    -------
    N/A

    Examples
    --------
    >>> wfdb.dl_files('ahadb', os.getcwd(),
                      ['STAFF-Studies-bibliography-2016.pdf', 'data/001a.hea',
                      'data/001a.dat'])

    """
    # Full url PhysioNet database
    db_dir = posixpath.join(db, get_version(db))
    db_url = posixpath.join(PN_CONTENT_URL, db_dir) + "/"

    # Check if the database is valid
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
    pool = multiprocessing.dummy.Pool(processes=2)
    pool.map(dl_pn_file, dl_inputs)
    print("Finished downloading files")

    return
