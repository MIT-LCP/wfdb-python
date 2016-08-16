### Written by - Chen Xie 2016 ### 
### Please report bugs and suggestions to https://github.com/MIT-LCP/wfdb-python or cx1111@mit.edu ###

import urllib.request as ul
import configparser 
import os
import sys
from . import readsignal

   
def checkrecordfiles(recordname, filedirectory):
    """Check a local directory along with the database cache directory specified in 'config.ini' for all necessary files required to read a WFDB record. Calls pbdownload.dlrecordfiles to download any missing files into the database cache directory. Returns the base record name if all files were present, or a full path record name specifying where the downloaded files are to be read, and a list of files downloaded. 
    
    *If you wish to directly download files for a record, it highly recommended to call 'pbdownload.dlrecordfiles' directly. This is a helper function for readsignal.rdsamp which tries to parse the 'recordname' input to deduce whether it contains a local directory, physiobank database, or both. Its usage format is different and more complex than that of 'dlrecordfiles'. 
    
    Usage: readrecordname, downloadedfiles = checkrecordfiles(recordname, filedirectory)
    
    Input arguments: 
    - recordname (required): The name of the WFDB record to be read (without any file extensions). Can be prepended with a local directory, or a physiobank subdirectory (or both if the relative local directory exists and takes the same name as the physiobank subdirectory). eg: recordname=mitdb/100 
    - filedirectory (required): The local directory to check for the files required to read the record before checking the database cache directory. If the 'recordname' argument is prepended with a directory, this function will assume that it is a local directory and prepend that to this 'filedirectory' argument and check the resulting directory instead.
    
    Output arguments:
    - readrecordname: The record name prepended with the path the files are to be read from.
    - downloadedfiles:  The list of files downloaded from PhysioBank. 
     
    """
    
    config=configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.ini"))
    dbcachedir=config['PBDOWNLOAD']['dbcachedir'] # Base directory to store downloaded physiobank files, read from config.ini
    basedir, baserecname = os.path.split(recordname) 
    
    # At this point we do not know whether basedir is a local directory, a physiobank directory, or both! 
   
    if not basedir: # if there is no base directory then: 1. The files must be located in the current working directory. 
        #  2. There will be no target files to download. So just directly return. If files are missing we cannot download them anyway.
        return recordname, []
          
    # If this is reached, basedir is defined. Check if there is a directory called 'basedir':
    if os.path.isdir(basedir): # the 'basedir' directory exists. Check it for files.
        # It is possible that basedir is also a physiobank database. Therefore if any files are missing, ,try to download files  assuming basedir is the physiobank database directory. If it turns out that basedir is not a pb database, an error will be triggered. The record would not be readable without the missing file(s) anyway.  
        
        downloaddir=os.path.join(dbcachedir, basedir)
        
        if not os.path.isfile(os.path.join(basedir, baserecname+".hea")): # The basedir directory is missing the header file.
            dledfiles=dlrecordfiles(recordname, downloaddir) # If invalid pb database, function would exit.
            return os.path.join(downloaddir, baserecname), dledfiles # Files downloaded, confirmed valid pb database.
        
        # Header is present in basedir
        fields=readsignal.readheader(recordname)
        
        if fields["nseg"]==1: # Single segment. Check for all the required dat files 
            for f in fields["filename"]:
                if not os.path.isfile(os.path.join(basedir, f)): # Missing a dat file. Download in db cache dir. 
                    dledfiles=dlrecordfiles(recordname, downloaddir)    
                    return os.path.join(downloaddir, baserecname), dledfiles
        else: # Multi segment. Check for all segment headers and their dat files
            for segment in fields["filename"]:
                if segment!='~':
                    if not os.path.isfile(os.path.join(basedir, segment+".hea")): # Missing a segment header
                        dledfiles=dlrecordfiles(recordname, downloaddir)    
                        return os.path.join(downloaddir, baserecname), dledfiles
                    segfields=readsignal.readheader(os.path.join(basedir, segment))
                    for f in segfields["filename"]:
                        if f!='~':
                            if not os.path.isfile(os.path.join(basedir, f)): # Missing a segment's dat file     
                                dledfiles=dlrecordfiles(recordname, downloaddir)    
                                return os.path.join(downloaddir, baserecname), dledfiles
        
        return recordname, [] # All files were already present in the 'basedir' directory. 
        
       
    else: # there is no 'basedir' directory. Therefore basedir must be a physiobank database directory. check the current working directory for files. If any are missing, check the cache directory for files and download missing files from physiobank. 
        
        pbdir=basedir # physiobank directory
        downloaddir=os.path.join(dbcachedir, pbdir)
        
        if not os.path.isfile(baserecname+".hea"):
            dledfiles=dlrecordfiles(recordname, downloaddir)    
            return os.path.join(downloaddir, baserecname), dledfiles 
        
        # Header is present in current working dir. 
        fields=readsignal.readheader(baserecname)
        
        if fields["nseg"]==1: # Single segment. Check for all the required dat files 
            for f in fields["filename"]:
                if not os.path.isfile(f): # Missing a dat file. Download in db cache dir. 
                    dledfiles=dlrecordfiles(recordname, downloaddir)  
                    return os.path.join(downloaddir, baserecname), dledfiles
        else: # Multi segment. Check for all segment headers and their dat files
            for segment in fields["filename"]:
                if segment!='~':
                    if not os.path.isfile(os.path.join(targetdir, segment+".hea")): # Missing a segment header
                        dledfiles=dlrecordfiles(recordname, downloaddir)    
                        return os.path.join(downloaddir, baserecname), dledfiles
                    segfields=readsignal.readheader(os.path.join(targetdir, segment))
                    for f in segfields["filename"]:
                        if f!='~':
                            if not os.path.isfile(os.path.join(targetdir, f)): # Missing a segment's dat file     
                                dledfiles=dlrecordfiles(recordname, downloaddir)    
                                return os.path.join(downloaddir, baserecname), dledfiles

        # All files are present in current directory. Return base record name and no dled files.
        return baserecname, [] 
        
    
def dlrecordfiles(pbrecname, targetdir):
    
    """Check a specified local directory for all necessary files required to read a Physiobank record, and download any missing files into the same directory. Returns a list of files downloaded, or exits with error if an invalid Physiobank record is specified.  
    
    Usage: dledfiles = dlrecordfiles(pbrecname, targetdir)
    
    Input arguments: 
    - pbrecname (required): The name of the MIT format Physiobank record to be read, prepended with the Physiobank subdirectory the file is contain in (without any file extensions). eg. pbrecname=prcp/12726 to download files http://physionet.org/physiobank/database/prcp/12726.hea and 12727.dat  
    - targetdir (required): The local directory to check for files required to read the record, in which missing files are also downloaded. 
    
    Output arguments:
    - dledfiles:  The list of files downloaded from PhysioBank. 
     
    """
    pbdir, baserecname=os.path.split(pbrecname) 
    print('Downloading missing file(s) into directory: ', targetdir)
    
    if not os.path.isdir(targetdir): # Make the target directory if it doesn't already exist
        os.makedirs(targetdir)
        madetargetdir=1  
    else:
        madetargetdir=0 
    dledfiles=[] # List of downloaded files 
  
    # For any missing file, check if the input physiobank record name is valid, ie whether the file exists on physionet. Download if valid, exit if invalid. 
    
    if not os.path.isfile(os.path.join(targetdir, baserecname+".hea")):
        # Not calling dlorexit here. Extra instruction of removing the faulty created directory.
        try:
            ul.urlretrieve("http://physionet.org/physiobank/database/"+pbrecname+".hea", os.path.join(targetdir, baserecname+".hea")) 
            dledfiles.append(os.path.join(targetdir, baserecname+".hea"))
        except ul.HTTPError:
            if madetargetdir:
                os.rmdir(targetdir) # Remove the recently created faulty directory. 
            sys.exit("Attempted to download invalid target file: http://physionet.org/physiobank/database/"+pbrecname+".hea")
     
    fields=readsignal.readheader(os.path.join(targetdir, baserecname))
    
    # Even if the header file exists, it could have been downloaded prior. Need to check validity of link if ANY file is missing.     
    if fields["nseg"]==1: # Single segment. Check for all the required dat files 
        for f in fields["filename"]:
            if not os.path.isfile(os.path.join(targetdir, f)): # Missing a dat file
                dledfiles=dlorexit("http://physionet.org/physiobank/database/"+pbdir+"/"+f, os.path.join(targetdir, f), dledfiles)
                
    else: # Multi segment. Check for all segment headers and their dat files
        for segment in fields["filename"]:
            if segment!='~':
                if not os.path.isfile(os.path.join(targetdir, segment+".hea")): # Missing a segment header
                    dledfiles=dlorexit("http://physionet.org/physiobank/database/"+pbdir+"/"+segment+".hea", os.path.join(targetdir, segment+".hea"), dledfiles)
                segfields=readsignal.readheader(os.path.join(targetdir, segment))
                for f in segfields["filename"]:
                    if f!='~':
                        if not os.path.isfile(os.path.join(targetdir, f)): # Missing a segment's dat file     
                            dledfiles=dlorexit("http://physionet.org/physiobank/database/"+pbdir+"/"+f, os.path.join(targetdir, f), dledfiles)
    
    print('Download complete')
    return dledfiles # downloaded files
    
    

# Helper function for dlrecordfiles. Download the file from the specified 'url' as the 'filename', or exit with warning. 
def dlorexit(url, filename, dledfiles):
    try:
        ul.urlretrieve(url, filename)
        dledfiles.append(filename)
        return dledfiles
    except ul.HTTPError:
        sys.exit("Attempted to download invalid target file: "+url)

    
    
    
# Download files required to read a wfdb annotation. 
def dlannfiles():
    return dledfiles

    
    
# Download all the records in a physiobank database. 
def dlPBdatabase(database, targetdir):
    return dledfiles
    
    
