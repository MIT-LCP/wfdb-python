import multiprocessing
import numpy as np
import re
import os
import posixpath
import requests


# The physiobank index url
PB_INDEX_URL = 'http://physionet.org/physiobank/database/'

class Config(object):
    pass

# The configuration database index url. Uses physiobank index by default.
config = Config()
config.db_index_url = PB_INDEX_URL


def set_db_index_url(db_index_url=PB_INDEX_URL):
    """
    Set the database index url to a custom value, to stream remote
    files from another location.

    Parameters
    ----------
    db_index_url : str, optional
        The desired new database index url. Leave as default to reset
        to the physiobank index url.

    """
    config.db_index_url = db_index_url


def _remote_file_size(url=None, file_name=None, pb_dir=None):
    """
    Get the remote file size in bytes

    Parameters
    ----------
    url : str, optional
        The full url of the file. Use this option to explicitly
        state the full url.
    file_name : str, optional
        The base file name. Use this argument along with pb_dir if you
        want the full url to be constructed.
    pb_dir : str, optional
        The base file name. Use this argument along with file_name if
        you want the full url to be constructed.

    Returns
    -------
    remote_file_size : int
        Size of the file in bytes

    """

    # Option to construct the url
    if file_name and pb_dir:
        url = posixpath.join(config.db_index_url, pb_dir, file_name)

    response = requests.head(url, headers={'Accept-Encoding': 'identity'})
    # Raise HTTPError if invalid url
    response.raise_for_status()

    # Supposed size of the file
    remote_file_size = int(response.headers['content-length'])

    return remote_file_size

def _stream_header(file_name, pb_dir):
    """
    Stream the lines of a remote header file.

    Parameters
    ----------
    file_name : str

    pb_dir : str
        The Physiobank database directory from which to find the
        required header file. eg. For file '100.hea' in
        'http://physionet.org/physiobank/database/mitdb', pb_dir='mitdb'.

    """
    # Full url of header location
    url = posixpath.join(config.db_index_url, pb_dir, file_name)
    response = requests.get(url)

    # Raise HTTPError if invalid url
    response.raise_for_status()

    # Get each line as a string
    filelines = response.content.decode('iso-8859-1').splitlines()

    # Separate content into header and comment lines
    header_lines = []
    comment_lines = []

    for line in filelines:
        line = str(line.strip())
        # Comment line
        if line.startswith('#'):
            comment_lines.append(line)
        # Non-empty non-comment line = header line.
        elif line:
            # Look for a comment in the line
            ci = line.find('#')
            if ci > 0:
                header_lines.append(line[:ci])
                # comment on same line as header line
                comment_lines.append(line[ci:])
            else:
                header_lines.append(line)

    return (header_lines, comment_lines)


def _stream_dat(file_name, pb_dir, byte_count, start_byte, dtype):
    """
    Stream data from a remote dat file, into a 1d numpy array.

    Parameters
    ----------
    file_name : str
        The name of the dat file to be read.
    pb_dir : str
        The physiobank directory where the dat file is located.
    byte_count : int
        The number of bytes to be read.
    start_byte : int
        The starting byte number to read from.
    dtype : str
        The numpy dtype to load the data into.

    Returns
    -------
    sig_data : numpy array
        The data read from the dat file.

    """

    # Full url of dat file
    url = posixpath.join(config.db_index_url, pb_dir, file_name)

    # Specify the byte range
    end_byte = start_byte + byte_count - 1
    headers = {"Range":"bytes=%d-%d" % (start_byte, end_byte),
               'Accept-Encoding': '*/*'}

    # Get the content
    response = requests.get(url, headers=headers, stream=True)

    # Raise HTTPError if invalid url
    response.raise_for_status()

    # Convert to numpy array
    sig_data = np.fromstring(response.content, dtype=dtype)

    return sig_data


def _stream_annotation(file_name, pb_dir):
    """
    Stream an entire remote annotation file from physiobank

    Parameters
    ----------
    file_name : str
        The name of the annotation file to be read.
    pb_dir : str
        The physiobank directory where the annotation file is located.

    """
    # Full url of annotation file
    url = posixpath.join(config.db_index_url, pb_dir, file_name)

    # Get the content
    response = requests.get(url)
    # Raise HTTPError if invalid url
    response.raise_for_status()

    # Convert to numpy array
    ann_data = np.fromstring(response.content, dtype=np.dtype('<u1'))

    return ann_data


def get_dbs():
    """
    Get a list of all the Physiobank databases available.

    Examples
    --------
    >>> dbs = get_dbs()

    """
    url = posixpath.join(config.db_index_url, 'DBS')
    response = requests.get(url)

    dbs = response.content.decode('ascii').splitlines()
    dbs = [re.sub('\t{2,}', '\t', line).split('\t') for line in dbs]

    return dbs


# ---- Helper functions for downloading physiobank files ------- #

def get_record_list(db_dir, records='all'):
    """
    Get a list of records belonging to a database.

    Parameters
    ----------
    db_dir : str
        The database directory, usually the same as the database slug.
        The location to look for a RECORDS file.
    records : list, optional
        A Option used when this function acts as a helper function.
        Leave as default 'all' to get all records.

    Examples
    --------
    >>> wfdb.get_record_list('mitdb')

    """
    # Full url physiobank database
    db_url = posixpath.join(config.db_index_url, db_dir)

    # Check for a RECORDS file
    if records == 'all':
        response = requests.get(posixpath.join(db_url, 'RECORDS'))
        if response.status_code == 404:
            raise ValueError('The database %s has no WFDB files to download' % db_url)

        # Get each line as a string
        record_list = response.content.decode('ascii').splitlines()
    # Otherwise the records are input manually
    else:
        record_list = records

    return record_list


def get_annotators(db_dir, annotators):

    # Full url physiobank database
    db_url = posixpath.join(config.db_index_url, db_dir)

    if annotators is not None:
        # Check for an ANNOTATORS file
        r = requests.get(posixpath.join(db_url, 'ANNOTATORS'))
        if r.status_code == 404:
            if annotators == 'all':
                return
            else:
                raise ValueError('The database %s has no annotation files to download' % db_url)
        # Make sure the input annotators are present in the database
        ann_list = r.content.decode('ascii').splitlines()
        ann_list = [a.split('\t')[0] for a in ann_list]

        # Get the annotation file types required
        if annotators == 'all':
            # all possible ones
            annotators = ann_list
        else:
            # In case they didn't input a list
            if type(annotators) == str:
                annotators = [annotators]
            # user input ones. Check validity.
            for a in annotators:
                if a not in ann_list:
                    raise ValueError('The database contains no annotators with extension: %s' % a)

    return annotators


def make_local_dirs(dl_dir, dl_inputs, keep_subdirs):
    """
    Make any required local directories to prepare for downloading
    """

    # Make the local download dir if it doesn't exist
    if not os.path.isdir(dl_dir):
        os.makedirs(dl_dir)
        print('Created local base download directory: %s' % dl_dir)
    # Create all required local subdirectories
    # This must be out of dl_pb_file to
    # avoid clash in multiprocessing
    if keep_subdirs:
        dl_dirs = set([os.path.join(dl_dir, d[1]) for d in dl_inputs])
        for d in dl_dirs:
            if not os.path.isdir(d):
                os.makedirs(d)
    return


def dl_pb_file(inputs):
    """
    Download a file from physiobank.

    The input args are to be unpacked for the use of multiprocessing
    map, because python2 doesn't have starmap...

    """

    basefile, subdir, db, dl_dir, keep_subdirs, overwrite = inputs

    # Full url of file
    url = posixpath.join(config.db_index_url, db, subdir, basefile)

    # Supposed size of the file
    remote_file_size = _remote_file_size(url)

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
            # Local file is smaller than it should be. Append it.
            if local_file_size < remote_file_size:
                print('Detected partially downloaded file: %s Appending file...' % local_file)
                headers = {"Range": "bytes="+str(local_file_size)+"-", 'Accept-Encoding': '*/*'}
                r = requests.get(url, headers=headers, stream=True)
                print('headers: ', headers)
                print('r content length: ', len(r.content))
                with open(local_file, 'ba') as writefile:
                    writefile.write(r.content)
                print('Done appending.')
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
        The url of the file to download
    save_file_name : str
        The name to save the file as

    """
    response = requests.get(url)
    with open(save_file_name, 'wb') as writefile:
        writefile.write(response.content)

    return


def dl_files(db, dl_dir, files, keep_subdirs=True, overwrite=False):
    """
    Download specified files from a Physiobank database.

    Parameters
    ----------
    db : str
        The Physiobank database directory to download. eg. For database:
        'http://physionet.org/physiobank/database/mitdb', db='mitdb'.
    dl_dir : str
        The full local directory path in which to download the files.
    files : list
        A list of strings specifying the file names to download relative to the
        database base directory.
    keep_subdirs : bool, optional
        Whether to keep the relative subdirectories of downloaded files as they
        are organized in Physiobank (True), or to download all files into the
        same base directory (False).
    overwrite : bool, optional
        If True, all files will be redownloaded regardless. If False, existing
        files with the same name and relative subdirectory will be checked.
        If the local file is the same size as the online file, the download is
        skipped. If the local file is larger, it will be deleted and the file
        will be redownloaded. If the local file is smaller, the file will be
        assumed to be partially downloaded and the remaining bytes will be
        downloaded and appended.

    Examples
    --------
    >>> wfdb.dl_files('ahadb', os.getcwd(),
                      ['STAFF-Studies-bibliography-2016.pdf', 'data/001a.hea',
                      'data/001a.dat'])

    """

    # Full url physiobank database
    db_url = posixpath.join(config.db_index_url, db)
    # Check if the database is valid
    response = requests.get(db_url)
    response.raise_for_status()

    # Construct the urls to download
    dl_inputs = [(os.path.split(file)[1], os.path.split(file)[0], db, dl_dir, keep_subdirs, overwrite) for file in files]

    # Make any required local directories
    make_local_dirs(dl_dir, dl_inputs, keep_subdirs)

    print('Downloading files...')
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.Pool(processes=2)
    pool.map(dl_pb_file, dl_inputs)
    print('Finished downloading files')

    return
