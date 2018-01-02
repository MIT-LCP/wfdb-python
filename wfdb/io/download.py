import numpy as np
import re
import os
import posixpath
import requests


# Read a header file from physiobank
def stream_header(record_name, pb_dir):

    # Full url of header location
    url = posixpath.join(db_index_url, pb_dir, record_name+'.hea')
    r = requests.get(url)

    # Raise HTTPError if invalid url
    r.raise_for_status()

    # Get each line as a string
    filelines = r.content.decode('iso-8859-1').splitlines()

    # Separate content into header and comment lines
    headerlines = []
    commentlines = []

    for line in filelines:
        line = str(line.strip())
        # Comment line
        if line.startswith('#'):
            commentlines.append(line)
        # Non-empty non-comment line = header line.
        elif line:
            # Look for a comment in the line
            ci = line.find('#')
            if ci > 0:
                headerlines.append(line[:ci])
                # comment on same line as header line
                commentlines.append(line[ci:])
            else:
                headerlines.append(line)

    return (headerlines, commentlines)


# Read certain bytes from a dat file from physiobank
def stream_dat(file_name, pb_dir, fmt, bytecount, startbyte, datatypes):

    # Full url of dat file
    url = posixpath.join(db_index_url, pb_dir, file_name)

    # Specify the byte range
    endbyte = startbyte + bytecount-1
    headers = {"Range": "bytes="+str(startbyte)+"-"+str(endbyte), 'Accept-Encoding': '*/*'}

    # Get the content
    r = requests.get(url, headers=headers, stream=True)

    # Raise HTTPError if invalid url
    r.raise_for_status()

    sigbytes = r.content

    # Convert to numpy array
    sigbytes = np.fromstring(sigbytes, dtype = np.dtype(datatypes[fmt]))

    return sigbytes

# Read an entire annotation file from physiobank
def streamannotation(file_name, pb_dir):

    # Full url of annotation file
    url = posixpath.join(db_index_url, pb_dir, file_name)

    # Get the content
    r = requests.get(url)
    # Raise HTTPError if invalid url
    r.raise_for_status()

    annbytes = r.content

    # Convert to numpy array
    annbytes = np.fromstring(annbytes, dtype = np.dtype('<u1'))

    return annbytes


def get_dbs():
    """
    Return a list of all the physiobank databases available.

    Usage:
    dbs = get_dbs()
    """
    url = posixpath.join(db_index_url, 'DBS')
    r = requests.get(url)

    dbs = r.content.decode('ascii').splitlines()
    dbs = [re.sub('\t{2,}', '\t', line).split('\t') for line in dbs]

    return dbs


# ---- Helper functions for downloading physiobank files ------- #

def getrecordlist(dburl, records):
    # Check for a RECORDS file
    if records == 'all':
        r = requests.get(posixpath.join(dburl, 'RECORDS'))
        if r.status_code == 404:
            raise ValueError('The database '+dburl+' has no WFDB files to download')

        # Get each line as a string
        recordlist = r.content.decode('ascii').splitlines()
    # Otherwise the records are input manually
    else:
        recordlist = records

    return recordlist

def getannotators(dburl, annotators):

    if annotators is not None:
        # Check for an ANNOTATORS file
        r = requests.get(posixpath.join(dburl, 'ANNOTATORS'))
        if r.status_code == 404:
            if annotators == 'all':
                return
            else:
                raise ValueError('The database '+dburl+' has no annotation files to download')
        # Make sure the input annotators are present in the database
        annlist = r.content.decode('ascii').splitlines()
        annlist = [a.split('\t')[0] for a in annlist]

        # Get the annotation file types required
        if annotators == 'all':
            # all possible ones
            annotators = annlist
        else:
            # In case they didn't input a list
            if type(annotators) == str:
                annotators = [annotators]
            # user input ones. Check validity.
            for a in annotators:
                if a not in annlist:
                    raise ValueError('The database contains no annotators with extension: '+a)

    return annotators

# Make any required local directories
def makelocaldirs(dl_dir, dlinputs, keep_subdirs):

    # Make the local download dir if it doesn't exist
    if not os.path.isdir(dl_dir):
        os.makedirs(dl_dir)
        print("Created local base download directory: ", dl_dir)
    # Create all required local subdirectories
    # This must be out of dlpbfile to
    # avoid clash in multiprocessing
    if keep_subdirs:
        dldirs = set([os.path.join(dl_dir, d[1]) for d in dlinputs])
        for d in dldirs:
            if not os.path.isdir(d):
                os.makedirs(d)
    return


# Download a file from physiobank
# The input args are to be unpacked for the use of multiprocessing
def dlpbfile(inputs):

    basefile, subdir, db, dl_dir, keep_subdirs, overwrite = inputs

    # Full url of file
    url = posixpath.join(db_index_url, db, subdir, basefile)

    # Get the request header
    rh = requests.head(url, headers={'Accept-Encoding': 'identity'})
    # Raise HTTPError if invalid url
    rh.raise_for_status()

    # Supposed size of the file
    onlinefilesize = int(rh.headers['content-length'])

    # Figure out where the file should be locally
    if keep_subdirs:
        dldir = os.path.join(dl_dir, subdir)
    else:
        dldir = dl_dir

    localfile = os.path.join(dldir, basefile)

    # The file exists locally.
    if os.path.isfile(localfile):
        # Redownload regardless
        if overwrite:
            dlfullfile(url, localfile)
        # Process accordingly.
        else:
            localfilesize = os.path.getsize(localfile)
            # Local file is smaller than it should be. Append it.
            if localfilesize < onlinefilesize:
                print('Detected partially downloaded file: '+localfile+' Appending file...')
                headers = {"Range": "bytes="+str(localfilesize)+"-", 'Accept-Encoding': '*/*'}
                r = requests.get(url, headers=headers, stream=True)
                print('headers: ', headers)
                print('r content length: ', len(r.content))
                with open(localfile, "ba") as writefile:
                    writefile.write(r.content)
                print('Done appending.')
            # Local file is larger than it should be. Redownload.
            elif localfilesize > onlinefilesize:
                dlfullfile(url, localfile)
            # If they're the same size, do nothing.

    # The file doesn't exist. Download it.
    else:
        dlfullfile(url, localfile)

    return

# Download a file. No checks.
def dlfullfile(url, localfile):
    r = requests.get(url)
    with open(localfile, "wb") as writefile:
        writefile.write(r.content)

    return




db_index_url = 'http://physionet.org/physiobank/database/'
