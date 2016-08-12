import urllib.request as ul
import configparser 
import os
from . import readsignal






# Check for all files required to read a physiobank record and download all missing files from physiobank. Return the record name prepended with target directory where the files are to be read from. Called from rdsamp.     

# Does not work for downloading files in the physiobank index, but does work for files in databases. 

# filedirectory = directory to search for files. 
def getrecordfiles(recordname, filedirectory):
    
    print("we are here....")
    
    config=configparser.ConfigParser()
    config.read("config.ini")
    dbcachedir=config['PBDOWNLOAD']['dbcachedir'] # Location to store downloaded physiobank files read from config.ini
    filestoremove=[] # List of files that should be deleted after being processed.  
  


    # Actually we don't know if this is pbsubdir. It could just be a local directory.... eg. recordname='testfolder/100'
    
    # At this point it's impossible to tell if it is a local directory. Logic: need to test this first. See if it exists. 
    
    pbsubdir, baserecordname = os.path.split(recordname) # mitdb, 100 = split(mitdb/100)
    
    downloaddir=os.path.join(dbcachedir, pbsubdir) # Final directory to download and read files. /usr/local/database/mitdb
   
    


    if filedirectory==downloaddir: # Target file directory is cachedirectory/pbsubdirectory. Look for all required files there and download any missing ones into it. 
        printdlmessage=1 
        removefiles=int(config['PBDOWNLOAD']['removefiles']) # Specifier of whether to keep the downloaded files. 
        if not os.path.exists(downloaddir): # Create the cache subdirectory if necessary
            print("Creating database cache subdirectory: ", downloaddir)
            os.makedirs(downloaddir)
           
        # recordname=mitdb/100       filedirectory=/home/cx1111
        if not os.path.isfile(os.path.join(downloaddir, baserecordname+".hea")): # Base header not present. Download. 
            print("cant find",os.path.join(downloaddir, recordname+".hea"))
            print("Missing file(s)1. Downloading from: http://physionet.org/physiobank/database/"+pbsubdir+" into cache directory: "+downloaddir)
            printdlmessage=0
            ul.urlretrieve("http://physionet.org/physiobank/database/"+recordname+".hea", os.path.join(downloaddir, baserecordname+".hea"))
            if removefiles:
                filestoremove.append(os.path.join(downloaddir, os.path.join(downloaddir, baserecordname+".hea")))
                           
        #print("downloaddir: ", downloaddir)
        #print("baserecordname: ", baserecordname)
        
        
        fields=readsignal.readheader(os.path.join(downloaddir, baserecordname))
        if fields["nseg"]==1: # Single segment. Check for all the required dat files 
            for f in fields["filename"]:
                if not os.path.isfile(os.path.join(downloaddir, f)): # Missing a dat file
                    if printdlmessage:
                        print("Missing file(s)2. Downloading from: http://physionet.org/physiobank/database/"+pbsubdir+" into cache directory: "+downloaddir)
                        printdlmessage=0
                    ul.urlretrieve("http://physionet.org/physiobank/database/"+pbsubdir+"/"+f, os.path.join(downloaddir, f))
                    if removefiles:
                        filestoremove.append(os.path.join(downloaddir, f))
                    
        else: # Multi segment. Check for all segment headers and their dat files
            for segment in fields["filename"]:
                if segment!='~':
                    if not os.path.isfile(os.path.join(filedirectory, segment+".hea")): # Missing a segment header
                        if printdlmessage:
                            print("Missing file(s)3. Downloading from: http://physionet.org/physiobank/database/"+pbsubdir+" into cache directory: "+downloaddir)
                        printdlmessage=0
                        ul.urlretrieve("http://physionet.org/physiobank/database/"+pbsubdir+"/"+segment+".hea", os.path.join(downloaddir, segment+".hea"))
                        if removefiles:
                            filestoremove.append(os.path.join(downloaddir, segment+".hea"))
                
                    segfields=readsignal.readheader(os.path.join(downloaddir, segment))
                    for f in segfields["filename"]:
                        if f!='~':
                            if not os.path.isfile(os.path.join(downloaddir, f)): # Missing a segment's dat file 
                                if printdlmessage:
                                    print("Missing file(s)4. Downloading from: http://physionet.org/physiobank/database/"+pbsubdir+" into cache directory: "+downloaddir)
                                    printdlmessage=0
                                ul.urlretrieve("http://physionet.org/physiobank/database/"+pbsubdir+"/"+f, os.path.join(downloaddir, f))
                                if removefiles:
                                    filestoremove.append(os.path.join(downloaddir, f))
        
         # Put this before every urlget? print("Missing a file. Attempting to download from PhysioBank into cache directory...") 
        
        return os.path.join(downloaddir, baserecordname), filestoremove # return the record name with the db cache directory. 
        

    else: # Target file directory is not download directory. Look for all required files. If anything is missing, recall the 
        # function with the download directory as input. Nothing is ever downloaded outside the download directory. 
        
        # recordname=mitdb/100       filedirectory=/home/cx1111
        if not os.path.isfile(os.path.join(filedirectory, recordname+".hea")): # Missing base header. 
            return getrecordfiles(recordname, downloaddir)
        
        fields=readsignal.readheader(os.path.join(filedirectory, recordname))
        if fields["nseg"]==1: # Single segment. Check for all the required dat files 
            for f in fields["filename"]:
                if not os.path.isfile(os.path.join(filedirectory, f)): # Missing a dat file
                    return getrecordfiles(recordname, downloaddir)
        else: # Multi segment. Check for all segment headers and their dat files
            for segment in fields["filename"]:
                if segment!='~':
                    if not os.path.isfile(os.path.join(filedirectory, segment+".hea")): # Missing a segment header
                        return getrecordfiles(recordname, downloaddir)
                    else:
                        segfields=readsignal.readheader(os.path.join(filedirectory, segment))
                        for f in segfields["filename"]:
                            if f!='~':
                                if not os.path.isfile(os.path.join(filedirectory, f)): # Missing a segment's dat file 
                                    return getrecordfiles(recordname, downloaddir)
                    
        return recordname, filestoremove # All files present, nothing downloaded. Just return base recordname.
       
        
        
        #################################################### /old ###################################
        
        
        
        
        
        
        
        
        
# Make sure it works for 8 scenarios of input arguments and cases:

# 1. recordname=mitdb/100 
# 2. recordname=testfolder/100

# a. filedirectory=dbcachedir
# b. filedirectory!=dbcachedir

# i. mitdb or testfolder exists
# ii. mitdb or testfolder does not exist. 

        
# 1ai
# 1aii
# 1bi
# 1bii
# 2ai
# 2aii
# 2bi
# 2bii
        
        
        
        
# check filedirectory for necessary files. 
# If certain conditions are met, call dlrecfiles to download missing files into the cache directory + database subdirectory
# return the full record name of where to read the files, and a list of fill directory files to delete. 
            
# Initial input arguments: testfolder/100, pwd.          This function should NOT be called directly. But dlrecfiles can be. 
def checkrecordfiles(recordname, filedirectory):
    
    
    
    config=configparser.ConfigParser()
    config.read("config.ini")
    dbcachedir=config['PBDOWNLOAD']['dbcachedir'] # Base directory to store downloaded physiobank files, read from config.ini
    
      

    # Actually we don't know if this is pbsubdir. It could just be a local directory.... eg. recordname='testfolder/100'
    
    # At this point it's impossible to tell if it is a local directory. Logic: need to test this first. See if it exists. 
    
    basedir, baserecordname = os.path.split(recordname) # mitdb, 100 = split(mitdb/100)
    
    # At this point we do not know whether basedir is a local directory or a physiobank directory, or both! So figure it out...
    # No sneaky shortcut reading DBS file either since it doesn't specify subdirectories within databases.... 
    
    # downloaddir=os.path.join(dbcachedir, pbsubdir) 
   

    if not basedir: # if there is no base directory then: 1. The files must be located in the current working directory. 
        #  2. There will be no target files to download. So just directly return. If files are missing we cannot download them anyway.
        return recordname, filestoremove
    
    
    # If we reach here, it means basedir is defined. Check if the basedir directory exists:
    
    
    
    
    
    if os.path.isdir(basedir): # the basedir directory exists. Check it for files.  it may also be a physionet database directory....
        
        # if the basedir directory is missing a file. It is possible that basedir is also a physiobank database. What to do now?
        if not os.path.isfile(os.path.join(basedir, baserecordname+".hea")):
            
                
            # If it is not also a database then too bad, just return and let it fail. No target to download. 
            
            # So we somehow need to check if it is a database. We'd have to create the directory: dbcachedir/basedir and try to 
            # Download files. 
            # If it happens that basedir is a physiobank directory, great! Download away into the cache dir. 
            # If it is not a physiobank directory, then the original rdsamp would fail anyway so who cares. Only issue is the 
            # created folder in dbcachedir/basedir that shouldn't exist. Raise an error and remove it!!! 

            
            print("Missing file(s) in directory: "+basedir+" Attempting to download from: http://physionet.org/physiobank/database/"+basedir)
            dledfiles=dlrecordfiles(recordname, basedir)
            if dledfiles==-1: # Invalid return. This means basedir is not a physiobank database. 
                # Missing file and no valid pb database provided. Just return. The error will be triggered in rdsamp.   
                return recordname, [] 
        
        # Header either already existed or was downloaded. 
          
            
                
            
            
            
            
            
        
        
        return recordname, dledfiles
        
    else: # there is no 'basedir' directory. conclude that basedir must be a physiobank database directory. 
        
        pbdir=basedir # physiobank directory
        downloaddir=os.path.join(dbcachedir, pbdir)
        
        # check the current directory for files. If any are missing, check the cache directory for files and download missing 
        # files there from physiobank. 
        
        
        
        
        if not os.path.isfile(baserecordname):
            dledfiles=dlrecordfiles(baserecordname, downloaddir)
            
            return downloaddir+baserecordname, dledfiles # Downloaded files into the cache. Return that as the location to read. 
            
            
            
        
        
        
        # All files are present in current directory. Return base record name and no dled files.
        return baserecordname, [] 
        
    
    
    
    
# http://stackoverflow.com/questions/3256576/catching-http-errors
# Download all the required files for a record into a target folder. Files already present in the target folder will be omitted. 

# Returns an error if invalid database directory.
def dlrecordfiles(pbrecname, targetdir):
    
    pbdir, baserecname=os.path.split(pbrecname) 
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)
        madetargetdir=1 # Keep track of making targetdir in case it should be deleted. 
    else:
        madetargetdir=0 
    dledfiles=[] # List of downloaded files 
   
    # Before checking directory, check if the physiobank record name provided is valid. If not, then just return -1. 
    # Don't worry about checking and downloading header. Negligible overhead compared to needing to download other files. 
    # If they already has all files, then why the hell would they call this anyway? rdsamp only calls this if a file is missing. 
    
    # Is there a situation where this would prevent useful action?
    # This function is either called directly by a user or by checkrecords in rdsamp. If the link is invalid, there is nothing to 
    # dl anyway so this function is useless. ALso if they already had the files, why the hell would they call it anyway? 
    
    
    # Check if the physiobank record name is valid, ie that the file exists on physionet. Download the header if valid, or exit. 
    try:
        ul.urlretrieve("http://physionet.org/physiobank/database/"+pbrecname+".hea", os.path.join(targetdir, baserecname+".hea")) 
    except ul.HTTPError:
        print("Invalid PhysioBank record name provided - ", pbrecname)
        return -1
    
    # If this is reached, the physiobank record is valid and the header file has been downloaded. 
 
    fields=readheader(os.path.join(targetdir, baserecname))
    
    if fields["nseg"]==1: # Single segment. Check for all the required dat files 
        for f in fields["filename"]:
            if not os.path.isfile(os.path.join(targetdir, f)): # Missing a dat file
                ul.urlretrieve("http://physionet.org/physiobank/database/"+pbdir+"/"+f, os.path.join(targetdir, f))
                dledfiles.append(os.path.join(targetdir, f))
    else: # Multi segment. Check for all segment headers and their dat files
        for segment in fields["filename"]:
            if segment!='~':
                if not os.path.isfile(os.path.join(targetdir, segment+".hea")): # Missing a segment header
                    ul.urlretrieve("http://physionet.org/physiobank/database/"+pbdir+"/"+segment+".hea", os.path.join(targetdir, segment+".hea"))
                    dledfiles.append(os.path.join(targetdir, segment+".hea"))
                                
                segfields=readsignal.readheader(os.path.join(targetdir, segment))
                for f in segfields["filename"]:
                    if f!='~':
                        if not os.path.isfile(os.path.join(targetdir, f)): # Missing a segment's dat file     
                            ul.urlretrieve("http://physionet.org/physiobank/database/"+pbdir+"/"+f, os.path.join(targetdir, f))
                            dledfiles.append(os.path.join(targetdir, f))
    
    return dledfiles # downloaded files
    
    
    

    
    
    
    
    
    
    
    
    

def dlannfiles():
    return dledfiles

    
    
# Download all the records in a physiobank database. 
def dlPBdatabase(database, fulltargetfolder):
    
    
    
