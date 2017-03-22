import numpy as np
import re
import os
import sys
import requests
import multiprocessing
from . import records
        
# Read a header file from physiobank
def streamheader(recordname, pbdir):

    # Full url of header location
    url = os.path.join(dbindexurl, pbdir, recordname+'.hea')
    r = requests.get(url)
    
    # Raise HTTPError if invalid url
    r.raise_for_status()
    
    # Get each line as a string
    filelines = r.content.decode('ascii').splitlines()
    
    # Separate content into header and comment lines
    headerlines = []
    commentlines = []
    
    for line in filelines:
        line = line.strip()
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
def streamdat(filename, pbdir, fmt, bytecount, startbyte, datatypes):
    
    # Full url of dat file
    url = os.path.join(dbindexurl, pbdir, filename)

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

    # For special formats that were read as unsigned 1 byte blocks to be further processed,
    # convert dtype from uint8 to uint64
    if fmt == ['212', '310', '311']:
        sigbytes = sigbytes.astype('uint')

    return sigbytes

# Read an entire annotation file from physiobank
def streamannotation(filename, pbdir):

    # Full url of annotation file
    url = os.path.join(dbindexurl, pbdir, filename)

    # Get the content
    r = requests.get(url)
    # Raise HTTPError if invalid url
    r.raise_for_status()
    
    annbytes = r.content

    # Convert to numpy array
    annbytes = np.fromstring(annbytes, dtype = np.dtype('<u1'))

    return annbytes


# Download all the WFDB files from a physiobank database
def dldatabase(pbdb, dlbasedir, keepsubdirs = True, overwrite = False): 

    # Full url physiobank database
    dburl = os.path.join(dbindexurl, pbdb)

    # Check if the database is valid
    r = requests.get(dburl)
    r.raise_for_status()

    # Check for a RECORDS file
    recordsurl = os.path.join(dburl, 'RECORDS')
    r = requests.get(recordsurl)
    if r.status_code == 404:
        sys.exit('The database '+dburl+' has no WFDB files to download')

    # Get each line as a string
    recordlist = r.content.decode('ascii').splitlines()

    # All files to download (relative to the database's home directory)
    allfiles = []

    for rec in recordlist:
        # Check out whether each record is in MIT or EDF format
        if rec.endswith('.edf'):
            allfiles.append(rec)
        else:
            # If MIT format, have to figure out all associated files
            allfiles.append(rec+'.hea')
            
            dirname, baserecname = os.path.split(rec)

            record = records.rdheader(baserecname, pbdir = os.path.join(pbdb, dirname))

            # Single segment record
            if type(record) == records.Record:
                # Add all dat files of the segment
                for file in record.filename:
                    allfiles.append(os.path.join(dirname, file))

            # Multi segment record
            else:
                for seg in record.segname:
                    # Skip empty segments
                    if seg == '~':
                        continue
                    # Add the header
                    allfiles.append(os.path.join(dirname, seg+'.hea'))
                    # Layout specifier has no dat files
                    if seg.endswith('_layout'):
                        continue
                    # Add all dat files of the segment
                    recseg = records.rdheader(seg, pbdir = os.path.join(pbdb, dirname))
                    for file in recseg.filename:
                        allfiles.append(os.path.join(dirname, file))

    dlinputs = [(os.path.split(file)[1], os.path.split(file)[0], pbdb, dlbasedir, keepsubdirs, overwrite) for file in allfiles]

    # Make the local download dir if it doesn't exist
    if not os.path.isdir(dlbasedir):
        os.makedirs(dlbasedir)
        print("Created local base download directory: ", dlbasedir)

    print('Download files...')

    # Create multiple processes to download files. 
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.Pool(processes=2)
    pool.map(dlpbfile, dlinputs)

    print('Finished downloading files')

    return


# Download selected WFDB files from a physiobank database
# def dldatabaserecords(pbdb, dlbasedir, keepsubirs = True, overwrite = False): 



# Download a file from physiobank
def dlpbfile(inputs):

    basefile, subdir, pbdb, dlbasedir, keepsubdirs, overwrite = inputs

    # Full url of file
    url = os.path.join(dbindexurl, pbdb, subdir, basefile)
    
    # Get the request header
    rh = requests.head(url, headers={'Accept-Encoding': 'identity'})
    # Raise HTTPError if invalid url
    rh.raise_for_status()

    # Supposed size of the file
    onlinefilesize = int(rh.headers['content-length'])
    
    # Figure out where the file should be locally
    if keepsubdirs:
        dldir = os.path.join(dlbasedir, subdir)
        # Make the local download subdirectory if it doesn't exist
        if not os.path.isdir(dldir):  
            os.makedirs(dldir)
            print("Created local download subdirectory: ", dldir)
    else:
        dldir = dlbasedir
    
    localfile = os.path.join(dldir, basefile)
    
    # The file exists. Process accordingly.
    if os.path.isfile(localfile):
        # Redownload regardless
        if overwrite:
            dlfullfile(url, localfile)
        else:
            localfilesize = os.path.getsize(localfile)
            # Local file is smaller than it should be. Append it.
            if localfilesize < onlinefilesize:
                print('Detected partially downloaded file: '+localfile+' Appending file...')
                headers = {"Range": "bytes="+str(localfilesize)+"-", 'Accept-Encoding': '*/*'} 
                r = requests.get(url, headers=headers, stream=True)
                with open(localfile, "wb") as writefile:
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



dbindexurl = 'http://physionet.org/physiobank/database/'