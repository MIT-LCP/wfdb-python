import urllib.request as ul
import configparser 
import os
import sys
from . import readsignal

   
# Check for all files required to read a physiobank record. Call dlrecordfiles if any missing. Return the base record name prepended with the target directory where the files are to be read from and a list of files downloaded. All missing files will be downloaded in the database cache directory specified in the config.ini file.  
def checkrecordfiles(recordname, filedirectory):
    
    config=configparser.ConfigParser()
    config.read("config.ini")
    dbcachedir=config['PBDOWNLOAD']['dbcachedir'] # Base directory to store downloaded physiobank files, read from config.ini
    basedir, baserecname = os.path.split(recordname) 
    
    # At this point we do not know whether basedir is a local directory, a physiobank directory, or both! 
   
    if not basedir: # if there is no base directory then: 1. The files must be located in the current working directory. 
        #  2. There will be no target files to download. So just directly return. If files are missing we cannot download them anyway.
        return recordname, []
          
    # If this is reached, basedir is defined. Check if there is a directory called 'basedir':
    if os.path.isdir(basedir): # the 'basedir' directory exists. Check it for files.
        # It is possible that basedir is also a physiobank database. Therefore if any files are missing, ,try to download files (into the cache dir) assuming basedir is the physiobank database directory. If it turns out that basedir is not a pb database, an error will be triggered. The record would not be readable without the missing file(s) anyway.  
        
        downloaddir=os.path.join(dbcachedir, basedir)
        
        if not os.path.isfile(os.path.join(basedir, baserecname+".hea")): # The basedir directory is missing the header file.
            dledfiles=dlrecordfiles(recordname, basedir) # If invalid pb database, function would exit.
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
        
    
# http://stackoverflow.com/questions/3256576/catching-http-errors
# Download all the required files for a record into a target folder. Files already present in the target folder will be omitted. 
# checkrecordfiles only calls dlrecordfiles with targetdir==downloaddir  
# Exits if invalid physiobank record. 
def dlrecordfiles(pbrecname, targetdir):
    
    pbdir, baserecname=os.path.split(pbrecname) 
    
    if not os.path.isdir(targetdir): # Make the target directory if it doesn't already exist
        os.mkdir(targetdir)
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
            sys.exit("Error: Attempted to download invalid target file: "+url)
     
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
                    dledfilesdlorexit("http://physionet.org/physiobank/database/"+pbdir+"/"+segment+".hea", os.path.join(targetdir, segment+".hea"), dledfiles)
                segfields=readsignal.readheader(os.path.join(targetdir, segment))
                for f in segfields["filename"]:
                    if f!='~':
                        if not os.path.isfile(os.path.join(targetdir, f)): # Missing a segment's dat file     
                            dledfiles=dlorexit("http://physionet.org/physiobank/database/"+pbdir+"/"+f, os.path.join(targetdir, f), dledfiles)
    
    return dledfiles # downloaded files
    
    

# Helper function for dlrecordfiles. Download the file from the specified 'url' as the 'filename', or exit with warning. 
def dlorexit(url, filename, dledfiles):
    try:
        ul.urlretrieve(url, filename)
        dledfiles.append(filename)
        return dledfiles
    except ul.HTTPError:
        sys.exit("Error: Attempted to download invalid target file: "+url)

    
    
    
    
    
    
    
    
    

def dlannfiles():
    return dledfiles

    
    
# Download all the records in a physiobank database. 
def dlPBdatabase(database, fulltargetfolder):
    
    
    
    return dledfiles
    
    
