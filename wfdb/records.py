# Actions:
# For wrheader(), all fields must be already filled in and consistent with each other. The signals field will not be used.
# For wrsamp(), the field to use will be d_signals (which may be empty). 
# There will be a function to call on the object, with usesignals = digital or physical, which fills in the respective fields. setsignalfeatures( usesignals = 'digital' or 'physical'). 
# This is separate from another method 'setdefaults' which they will have to call too. 
# The checkfieldcohesion() function will be called in wrheader and check all the header fields, excluding the signals. 
# The checkfieldcohesion() function will be called in wrsamp after wrhea and before wrdat to check the signals against each field.

import numpy as np
import re
import os
import sys
from collections import OrderedDict
from calendar import monthrange
import ._headers
import ._signals

# The specifications of a WFDB fields.
class WFDBfieldspecs():
    
    def __init__(self, speclist):
    
        # Data types the field (or its elements) can be
        self.allowedtypes = speclist[0]
        # The text delimiter that preceeds the field if it is a field that gets written to header files.
        self.delimiter = speclist[1]
        # The required/dependent field which must also be present
        self.dependency = speclist[2]
        # Whether the field is always required for writing a header (WFDB requirements + extra rules enforced by this library).
        self.write_req = speclist[3]

# The following dictionaries hold WFDB field specifications, separated by category. 

# The physical and digital signals.
signalspecs = OrderedDict([('p_signals', WFDBfieldspecs([[np.ndarray], None, None, False])),
                          ('d_signals', WFDBfieldspecs([[np.ndarray], None, None, False]))])

# The segment field. A list of WFDBrecord objects?
segmentspecs = OrderedDict([('segment', WFDBfieldspecs([[WFDBrecord], None, None, True]))])

# Record specification fields  
# Note: nseg is essential for multi but not a field in single. 
# getwritefields defined in _headers.Headers_Mixin will remove it.           
recfieldspecs = OrderedDict([('recordname', WFDBfieldspecs([[str], '', None, True])),
                         ('nseg', WFDBfieldspecs([[int], '/', 'recordname', True])), 
                         ('nsig', WFDBfieldspecs([[int], ' ', 'recordname', True])),
                         ('fs', WFDBfieldspecs([[int, float], ' ', 'nsig', True])),
                         ('counterfreq', WFDBfieldspecs([[int, float], '/', 'fs', False])),
                         ('basecounter', WFDBfieldspecs([[int, float], '(', 'counterfreq', False])),
                         ('siglen', WFDBfieldspecs([[int], ' ', 'fs', True])),
                         ('basetime', WFDBfieldspecs([[str], ' ', 'siglen', False])),
                         ('basedate', WFDBfieldspecs([[str], ' ', 'basetime', False]))])
 
# Signal specification fields. Type will be list.
sigfieldspecs = OrderedDict([('filename', WFDBfieldspecs([[str], '', None, True])),
                         ('fmt', WFDBfieldspecs([[int, str], ' ', 'filename', True])),
                         ('sampsperframe', WFDBfieldspecs([[int], 'x', 'fmt', False])),
                         ('skew', WFDBfieldspecs([[int], ':', 'fmt', False])),
                         ('byteoffset', WFDBfieldspecs([[int], '+', 'fmt', False])),
                         ('adcgain', WFDBfieldspecs([[int, float], ' ', 'fmt', True])),
                         ('baseline', WFDBfieldspecs([[int], '(', 'adcgain', True])),
                         ('units', WFDBfieldspecs([[str], '/', 'adcgain', True])),
                         ('adcres', WFDBfieldspecs([[int], ' ', 'adcgain', False])),
                         ('adczero', WFDBfieldspecs([[int], ' ', 'adcres', False])),
                         ('initvalue', WFDBfieldspecs([[int], ' ', 'adczero', False])),
                         ('checksum', WFDBfieldspecs([[int], ' ', 'initvalue', False])),
                         ('blocksize', WFDBfieldspecs([[int], ' ', 'checksum', False])),
                         ('signame', WFDBfieldspecs([[str], ' ', 'blocksize', False]))])
    
# Segment specification fields. Type will be list. 
segfieldspecs = OrderedDict([('segname', WFDBfieldspecs([[str], '', None, True, 0])),
                         ('seglen', WFDBfieldspecs([[int], ' ', 'segname', True, 0]))])

# Comment field. Type will be list.
comfieldspecs = OrderedDict([('comments', WFDBfieldspecs([[int], '', None, False, False]))])





# The base WFDB class to extend to create WFDBrecord and WFDBmultirecord. Contains shared helper functions and fields.             
class WFDBbaserecord():
    # Constructor
    
    def __init__(self, recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter = None, 
                 siglen = None, basetime = None,basedate = None, 
                 comments = None)

    # Check whether a single field is valid in its basic form. Does not check compatibility with other fields. 
    def checkfield(self, field): 
        
        # Check that the field is present
        if self.field == None:
            sys.exit("Missing field required: "+field)
           
        # Check the type of the field (and of its elements if it is to be a list) 
        self.checkfieldtype(field)
        
        # Individual specific field checks:
        if field == 'd_signals':
            # Check shape
            if self.d_signals.ndim != 2:
                errormsg = "signals must be a 2d numpy array"
                return errormsg
            # Check dtype
            if self.d_signals.dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
                errormsg = 'd_signals must be a 2d numpy array with dtype == int64, int32, int16, or int8.'
                return errormsg     
        elif field =='p_signals':        
            # Check shape
            if self.p_signals.ndim != 2:
                errormsg = "signals must be a 2d numpy array"
            
        # Add this later. 
        #elif field == 'segment':
            
        # Record specification fields
        elif field == 'recordname':       
            # Allow letters, digits, and underscores.
            if re.match('\w+', self.recordname).string != self.recordname:
                errormsg = 'recordname must only comprise of letters, digits, and underscores.'
        elif field == 'nseg':
            if self.nseg <=0:
                errormsg = 'nseg must be a positive integer'
        elif field == 'nsig':
            if self.nsig <=0:
                errormsg = 'nsig must be a positive integer'
        elif field == 'fs':
            if self.fs<=0:
                errormsg = 'fs must be a positive number'
        elif field == 'counterfreq':
            if self.counterfreq <=0:
                errormsg = 'counterfreq must be a positive number'
        elif field == 'basecounter':
            if self.basecounter <=0:
                errormsg = 'basecounter must be a positive number' 
        elif field == 'siglen':
            if self.siglen <=0:
                errormsg = 'siglen must be a positive integer'
        elif field == 'basetime':
            _ = parsetimestring(self.basetime)
        elif field == 'basedate':
            _ = parsetimestring(self.basedate)
        
        # Signal specification fields. Lists of elements. 
        elif field == 'filename':
            # Check for filename characters
            for f in self.filename:
                acceptedstring = re.match('[\w]+\.?[\w]+',f)
                if not acceptedstring or acceptedstring != f:
                    sys.exit('File names should only contain alphanumerics and an extension. eg. record_100.dat')
            # Check that dat files are grouped together 
            if orderedsetlist(self.filename)[0] != orderednoconseclist(self.filename):
                sys.exit('filename error: all entries for signals that share a given file must be consecutive')
        elif field == 'fmt':
            for f in self.fmt:
                if f not in datformats:
                    sys.exit('File formats must be valid WFDB dat formats: '+' , '.join(datformats))    
        elif field == 'sampsperframe':
            for f in self.sampsperframe:
                if f < 1:
                    sys.exit(errormsg = 'sampsperframe values must be positive integers')
                sys.exit('Sorry, I have not implemented multiple samples per frame into wrsamp yet')
        elif field == 'skew':
            for f in self.skew:
                if f < 1:
                    sys.exit('skew values must be non-negative integers')
                sys.exit('Sorry, I have not implemented skew into wrsamp yet')
        elif field == 'byteoffset':
            for f in self.byteoffset:
                if f < 0:
                    sys.exit('byteoffset values must be non-negative integers')
        elif field == 'adcgain':
            for f in self.adcgain:
                if f <= 0:
                    sys.exit('adcgain values must be positive numbers')
        elif field == 'baseline':
            # Currently original WFDB library only has 4 bytes for baseline. 
            for f in self.baseline:
                if f < -2147483648 or f> 2147483648:
                    sys.exit('baseline values must be between -2147483648 (-2^31) and 2147483647 (2^31 -1)')
        elif field == 'units':
            for f in self.units:
                if re.search('\s', f):
                    sys.exit('units strings may not contain whitespaces.')
        elif field == 'adcres':
            for f in self.adcres:
                if f < 1:
                    sys.exit('adcres values must be positive integers')
        # elif field == 'adczero': nothing to check here
        # elif field == 'initvalue': nothing to check here
        # elif field == 'checksum': nothing to check here
        elif field == 'blocksize': 
            for f in self.blocksize:
                if f < 1:
                    sys.exit('blocksize values must be positive integers')
        elif field == 'signame':
            for f in self.signame:
                if re.search('\s', f):
                    sys.exit('signame strings may not contain whitespaces.')
            
        # Segment specification fields
        elif field == 'segname':
            for f in self.segname:
                acceptedstring = re.match('[\w]+',f)
                if not acceptedstring or acceptedstring != f:
                    sys.exit('Segment record names should only contain alphanumerics')
        elif field == 'seglen':
            for f in self.seglen:
                if f < 1:
                    sys.exit('seglen values must be positive integers')
                    
        # Comment field
        elif field == 'comments':
            for f in self.comments:
                if f=='': # Allow empty string comment lines
                    continue
                if f[0] == '#':
                    print("Note: comment strings do not need to begin with '#'. This library adds them automatically.")
                if re.search('[\t\n\r\f\v]', f):
                    sys.exit('comments may not contain tabs or newlines (they may contain spaces and underscores).')
                    


    # Check the data type of the specified field.
    def checkfieldtype(self, field):
        
        # signal, segment, and comment specification fields are lists. Check their elements.    
        if field in sigfieldspecs:
            listcheck = 1
            allowedtypes = sigfieldspecs[field].allowedtypes
        elif field in segfieldspecs:
            listcheck = 1
            allowedtypes = segfieldspecs[field].allowedtypes
        elif field in comfieldspecs:
            listcheck = 1
            allowedtypes = comfieldspecs[field].allowedtypes
        elif field in signalspecs:
            listcheck = 1
            allowedtypes = signalspecs[field].allowedtypes
        elif field in segmentspecs:
            listcheck = 1
            allowedtypes = segmentspecs[field].allowedtypes
        elif field in recfieldspecs:
            listcheck = 1
            allowedtypes = recfieldspecs[field].allowedtypes

        item = getattr(self, field)

        # List fields and their elements
        if listcheck:
            if type(item)!=list:
                sys.exit('Field: '+field+' must be a list')
            for i in item:
                if type(i) not in allowedtypes:
                    print('Each element in field: '+field+' must be one of the following types:', allowedtypes)
                    sys.exit()
        # Non-list fields  
        else:
            if type(item) not in allowedtypes:
                print('Field: '+field+' must be one of the following types:', allowedtypes)
                sys.exit()


# Class for single segment WFDB records.
class WFDBrecord(WFDBbaserecord, _headers.Headers_Mixin, _signals.Signals_Mixin):
    
    # Constructor
    def __init__(self, p_signals=None, d_signals=None,
                 recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter=None, 
                 siglen=None, basetime=None, basedate=None, 
                 filename=None, fmt=None, sampsperframe=None, 
                 skew=None, byteoffset=None, adcgain=None, 
                 baseline=None, units=None, adcres=None, 
                 adczero=None, initvalue=None, checksum=None, 
                 blocksize=None, signame=None, comments=None):
        
        # Note the lack of 'nseg' field. Single segment records cannot have this field. Even nseg = 1 makes 
        # the header a multi-segment header. 
        
        super(self, recordname=recordname, nsig=nsig, 
              fs=fs, counterfreq=counterfreq, basecounter =basecounter,
              siglen = siglen, basetime = basetime, basedate = basedate, 
              comments = comments)
        
        self.p_signals = p_signals
        self.d_signals = d_signals
        
        self.filename=filename
        self.fmt=fmt
        self.sampsperframe=sampsperframe
        self.skew=skew
        self.byteoffset=byteoffset
        self.adcgain=adcgain
        self.baseline=baseline
        self.units=units
        self.adcres=adcres
        self.adczero=adczero
        self.initvalue=initvalue
        self.checksum=checksum
        self.blocksize=blocksize
        self.signame=signame
        
        
    # Write a wfdb header file and associated dat files if any.  
    def wrsamp(self):

        # Perform field validity and cohesion checks, and write the header file.
        self.wrheader()

        if self.nsig>0:
            # Perform signal validity and cohesion checks, and write the associated dat files.
            self.wrdats()
            
    



        

            


# Time string parser for WFDB header - H(H):M(M):S(S(.sss)) format. 
def parsetimestring(timestring):
    times = re.findall("(?P<hours>\d{1,2}):(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}[.\d+]*)", timestring)
        
    if not times:
        sys.exit("Invalid time string: "+timestring+". Acceptable format is: 'Hours:Minutes:Seconds'")
    else:
        hours, minutes, seconds = times[0]
        
    if not hours or not minutes or not seconds:
        sys.exit("Invalid time string: "+timestring+". Acceptable format is: 'Hours:Minutes:Seconds'")
    
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)
    
    if int(hours) >23:
        sys.exit('hours must be < 24')
    elif hours<0:
        sys.exit('hours must be positive')
    if minutes>59:
        sys.exit('minutes must be < 60') 
    elif minutes<0:
        sys.exit('minutes must be positive')  
    if seconds>59:
        sys.exit('seconds must be < 60')
    elif seconds<0:
        sys.exit('seconds must be positive')
        
    return (hours, minutes, seconds)

# Date string parser for WFDB header - DD/MM/YYYY   
def parsedatestring(datestring):
    dates = re.findall(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})", datestring)
        
    if not dates:
        sys.exit("Invalid date string. Acceptable format is: 'DD/MM/YYYY'")
    else:
        day, month, year = dates[0]
    
    day = int(day)
    month = int(month)
    year = int(year)
    
    if year<1:
        sys.exit('year must be positive')
    if month<1 or month>12:
        sys.exit('month must be between 1 and 12')
    if day not in range(monthrange(year, month)[0], monthrange(year, month)[1]):
        sys.exit('day does not exist for specified year and month')
    
    return (day, month, year)
    
                    
                
                
                
  

# Write each line in a list of strings to a text file
def linestofile(filename, lines):
    f = open(headername,'w')
    for l in lines:
        f.write("%s\n" % l)
    f.close()              
                
                
                