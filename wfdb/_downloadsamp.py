import numpy as np
import re
import os
import sys
import requests
from ._rdheader import rdheader

def downloadsamp(pbrecname, targetdir):
    """Check a specified local directory for all necessary files required to read a Physiobank
       record, and download any missing files into the same directory. Returns a list of files
       downloaded, or exits with error if an invalid Physiobank record is specified.

    Usage: dledfiles = dlrecordfiles(pbrecname, targetdir)

    Input arguments:
    - pbrecname (required): The name of the MIT format Physiobank record to be read, prepended
      with the Physiobank subdirectory the file is contained in (without any file extensions).
      eg. pbrecname=prcp/12726 to download files http://physionet.org/physiobank/database/prcp/12726.hea
      and 12727.dat
    - targetdir (required): The local directory to check for files required to read the record,
      in which missing files are also downloaded.

    Output arguments:
    - dledfiles:  The list of files downloaded from PhysioBank.

    """

    physioneturl = "http://physionet.org/physiobank/database/"
    pbdir, baserecname = os.path.split(pbrecname)
    displaydlmsg=1
    dledfiles = [] 
    
    if not os.path.isdir(targetdir):  # Make the target dir if it doesn't exist
        os.makedirs(targetdir)
        print("Created local directory: ", targetdir)
    
    # For any missing file, check if the input physiobank record name is
    # valid, ie whether the file exists on physionet. Download if valid, exit
    # if invalid.
    dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+baserecname+".hea", os.path.join(targetdir, 
        baserecname+".hea"), dledfiles, displaydlmsg, targetdir)
        
    fields = rdheader(os.path.join(targetdir, baserecname))

    # Need to check validity of link if ANY file is missing.
    if fields["nseg"] == 1:  # Single segment. Check for all the required dat files
        for f in set(fields["filename"]):
            # Missing dat file
            dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+f, os.path.join(targetdir, f), 
                dledfiles, displaydlmsg, targetdir)
    else:  # Multi segment. Check for all segment headers and their dat files
        for segment in fields["filename"]:
            if segment != '~':
                # Check the segment header
                dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+segment+".hea", 
                    os.path.join(targetdir, segment+".hea"), dledfiles, displaydlmsg, targetdir)    
                segfields = rdheader(os.path.join(targetdir, segment))
                for f in set(segfields["filename"]):
                    if f != '~':
                        # Check the segment's dat file
                        dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+f, 
                            os.path.join(targetdir, f), dledfiles, displaydlmsg, targetdir)
                            
    if dledfiles:
        print('Downloaded all missing files for record.')
    return dledfiles  # downloaded files


# Download a file if it is missing. Also error check 0 byte files.  
def dlifmissing(url, filename, dledfiles, displaydlmsg, targetdir):
    fileexists = os.path.isfile(filename)  
    if fileexists:
        # Likely interrupted download
        if os.path.getsize(filename)==0:
            try:
                input = raw_input
            except NameError:
                pass
            userresponse=input("Warning: File "+filename+" is 0 bytes.\n"
                "Likely interrupted download. Remove file and redownload? [y/n]: ")
            # override input for python 2 compatibility
            while userresponse not in ['y','n']:
                userresponse=input("Remove file and redownload? [y/n]: ")
            if userresponse=='y':
                os.remove(filename)
                dledfiles.append(dlorexit(url, filename, displaydlmsg, targetdir))
                displaydlmsg=0
            else:
                print("Skipping download.")
        # File is already present.
        else:
            print("File "+filename+" is already present.")
    else:
        dledfiles.append(dlorexit(url, filename, displaydlmsg, targetdir))
        displaydlmsg=0
    
    # If a file gets downloaded, displaydlmsg is set to 0. No need to print the message more than once. 
    return dledfiles, displaydlmsg
                 
    
# Download the file from the specified 'url' as the 'filename', or exit with warning.
def dlorexit(url, filename, displaydlmsg=0, targetdir=[]):
    if displaydlmsg: # We want this message to be called once for all files downloaded.
        print('Downloading missing file(s) into directory: {}'.format(targetdir))
    try:
        r = requests.get(url)
        with open(filename, "wb") as writefile:
            writefile.write(r.content)
        return filename
    except requests.HTTPError:
        sys.exit("Attempted to download invalid target file: " + url)


# Download files required to read a wfdb annotation.
def dlannfiles():
    return dledfiles


# Download all the records in a physiobank database.
def dlPBdatabase(database, targetdir):
    return dledfiles



