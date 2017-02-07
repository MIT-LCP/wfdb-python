# For wrheader(), all fields must be already filled in and cohesive with one another other. The signals field will not be used.
# For wrsamp(), the field to use will be d_signals (which is allowed to be empty for 0 channel records). 
# set_p_features and set_d_features use characteristics of the p_signals or d_signals field to fill in other header fields. 
# These are separate from another method 'setdefaults' which the user may call to  
# The checkfieldcohesion() function will be called in wrheader which checks all the header fields.
# The checksignalcohesion() function will be called in wrsamp in wrdat to check the d_signal against the header fields.

import numpy as np
import re
import os
import sys
from collections import OrderedDict
from calendar import monthrange
from . import _headers
from . import _signals


# The base WFDB class to extend to create WFDBrecord and WFDBmultirecord. Contains shared helper functions and fields.             
class WFDBbaserecord():
    # Constructor
    
    def __init__(self, recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter = None, 
                 siglen = None, basetime = None, basedate = None, 
                 comments = None):
        self.recordname = recordname
        self.nsig = nsig
        self.fs = fs
        self.counterfreq = counterfreq
        self.basecounter = basecounter
        self.siglen = siglen
        self.basetime = basetime
        self.basedate = basedate
        self.comments = comments




    # Check whether a single field is valid in its basic form. Does not check compatibility with other fields. 
    def checkfield(self, field): 
        # Check that the field is present
        if getattr(self, field) is None:
            sys.exit("Missing field required: "+field)
           
        # Check the type of the field (and of its elements if it is to be a list) 
        self.checkfieldtype(field)
        
        # Individual specific field checks:
        if field == 'd_signals':
            # Check shape
            if self.d_signals.ndim != 2:
                sys.exit("signals must be a 2d numpy array")
            # Check dtype
            if self.d_signals.dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
                sys.exit('d_signals must be a 2d numpy array with dtype == int64, int32, int16, or int8.')   
        elif field =='p_signals':        
            # Check shape
            if self.p_signals.ndim != 2:
                sys.exit("signals must be a 2d numpy array")
            
        # Add this later. 
        #elif field == 'segment':
            
        # Record specification fields
        elif field == 'recordname':       
            # Allow letters, digits, and underscores.
            acceptedstring = re.match('\w+', self.recordname)
            if not acceptedstring or acceptedstring.string != self.recordname:
                sys.exit('recordname must only comprise of letters, digits, and underscores.')
        elif field == 'nseg':
            if self.nseg <=0:
                sys.exit('nseg must be a positive integer')
        elif field == 'nsig':
            if self.nsig <=0:
                sys.exit('nsig must be a positive integer')
        elif field == 'fs':
            if self.fs<=0:
                sys.exit('fs must be a positive number')
        elif field == 'counterfreq':
            if self.counterfreq <=0:
                sys.exit('counterfreq must be a positive number')
        elif field == 'basecounter':
            if self.basecounter <=0:
                sys.exit('basecounter must be a positive number') 
        elif field == 'siglen':
            if self.siglen <0:
                sys.exit('siglen must be a non-negative integer')
        elif field == 'basetime':
            _ = parsetimestring(self.basetime)
        elif field == 'basedate':
            _ = parsetimestring(self.basedate)
        
        # Signal specification fields. Lists of elements. 
        elif field == 'filename':
            # Check for filename characters
            for f in self.filename:
                acceptedstring = re.match('[\w]+\.?[\w]+',f)
                if not acceptedstring or acceptedstring.string != f:
                    sys.exit('File names should only contain alphanumerics and an extension. eg. record_100.dat')
            # Check that dat files are grouped together 
            if orderedsetlist(self.filename)[0] != orderednoconseclist(self.filename):
                sys.exit('filename error: all entries for signals that share a given file must be consecutive')
        elif field == 'fmt':
            for f in self.fmt:
                if f not in _signals.datformats:
                    sys.exit('File formats must be valid WFDB dat formats: '+' , '.join(_signals.datformats))    
        elif field == 'sampsperframe':
            for f in self.sampsperframe:
                if f < 1:
                    sys.exit('sampsperframe values must be positive integers')
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
                if f < 0:
                    sys.exit('adcres values must be non-negative integers')
        # elif field == 'adczero': nothing to check here
        # elif field == 'initvalue': nothing to check here
        # elif field == 'checksum': nothing to check here
        elif field == 'blocksize': 
            for f in self.blocksize:
                if f < 0:
                    sys.exit('blocksize values must be non-negative integers')
        elif field == 'signame':
            for f in self.signame:
                if re.search('\s', f):
                    sys.exit('signame strings may not contain whitespaces.')
            
        # Segment specification fields
        elif field == 'segname':
            # Segment names must be alphanumerics or just a single '~'
            for f in self.segname:
                if f == '~':
                    continue
                acceptedstring = re.match('[\w]+',f)
                if not acceptedstring or acceptedstring.string != f:
                    sys.exit("Non-null segment names may only contain alphanumerics. Null segment names must be equal to '~'")
        elif field == 'seglen':
            # For records with more than 1 segment, the first segment may be 
            # the layout specification segment with a length of 0
            if len(self.seglen)>1:
                if self.seglen[0] < 0:
                    sys.exit('seglen values must be positive integers. Only seglen[0] may be 0 to indicate a layout segment')
                sl = self.seglen[1:]
            else:
                sl = self.seglen 
            for f in sl:
                if f < 1:
                    sys.exit('seglen values must be positive integers. Only seglen[0] may be 0 to indicate a layout segment')
                    
        # Comment field
        elif field == 'comments':
            for f in self.comments:
                if f=='': # Allow empty string comment lines
                    continue
                if f[0] == '#':
                    print("Note: comment strings do not need to begin with '#'. This library adds them automatically.")
                if re.search('[\t\n\r\f\v]', f):
                    sys.exit('comments may not contain tabs or newlines (they may contain spaces and underscores).')
                    



# Class for single segment WFDB records.
class WFDBrecord(WFDBbaserecord, _headers.HeadersMixin, _signals.SignalsMixin):
    
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
        
        super(WFDBrecord, self).__init__(recordname, nsig,
                    fs, counterfreq, basecounter, siglen,
                    basetime, basedate, comments)
        
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
    

    # Example sequence for a user to call wrsamp on a physical ecg signal x with 3 channels:
    # 1. Initiate the WFDBrecord object with essential information. 
    # >> record1 = WFDBrecord(recordname = 'record1', p_signals = x, fs = 125, units = ['mV','mV','mV'], signame = ['I','II','V'])
    # 2. Compute optimal fields to store the digital signal, carry out adc, and set the fields.
    # >> record1.set_d_features(do_adc = 1)
    # 3. Set default values of any missing field dependencies
    # >> record1.setdefaults()
    # 4. Write the record files - header and associated dat
    # >> record1.wrsamp()

    # Example sequence for a user to call wrsamp on a digital ecg signal x with 3 channels:
    # 1. Initiate the WFDBrecord object with essential information. 
    # >> record1 = WFDBrecord(recordname = 'record1', p_signals = x, fs = 125, units = ['mV','mV','mV'], signame = ['I','II','V'])
    # 2. Compute optimal fields to store the digital signal, carry out adc, and set the fields.
    # >> record1.set_d_features(do_adc = 1)
    # 3. Set default values of any missing field dependencies
    # >> record1.setdefaults()
    # 4. Write the record files - header and associated dat
    # >> record1.wrsamp()

    # Check the data type of the specified field.
    def checkfieldtype(self, field):
        
        # signal and comment specification fields are lists. Check their elements.  

        # Record specification field  
        if field in _headers.recfieldspecs:
            listcheck = 0
            allowedtypes = _headers.recfieldspecs[field].allowedtypes
        # Signal specification field
        elif field in _headers.sigfieldspecs:
            listcheck = 1
            allowedtypes = _headers.sigfieldspecs[field].allowedtypes
        # Comments field
        elif field == 'comments':
            listcheck = 1
            allowedtypes = [str]
        # Signals field
        elif field in ['p_signals','d_signals']:
            listcheck = 0
            allowedtypes = [np.ndarray]

        item = getattr(self, field)

        # List fields and their elements
        if listcheck:
            if type(item)!=list:
                sys.exit('Field: '+field+' must be a list with length equal to nsig')
            for i in item:
                if type(i) not in allowedtypes:
                    print('Each element in field: '+field+' must be one of the following types:', allowedtypes)
                    sys.exit()
        # Non-list fields  
        else:
            if type(item) not in allowedtypes:
                print('Field: '+field+' must be one of the following types:', allowedtypes)
                sys.exit()




# Class for multi segment WFDB records.
class WFDBmultirecord(WFDBbaserecord, _headers.MultiHeadersMixin):
    
    # Constructor
    def __init__(self, segments = None, recordname=None, 
                 nsig=None, fs=None, counterfreq=None, 
                 basecounter=None, siglen=None,
                 basetime=None, basedate=None, segname = None,
                 seglen = None, comments=None):


        super(WFDBmultirecord, self).__init__(recordname, nsig,
                    fs, counterfreq, basecounter, siglen,
                    basetime, basedate, comments)

        self.segments = segments
        self.segname = segname
        self.seglen = seglen

    # Write a multi-segment header, along with headers and dat files for all segments
    def wrsamp():
        print('to do')


    # Check the data type of the specified field.
    def checkfieldtype(self, field):
        
        # segment and comment specification, and segment fields are lists. Check their elements.    
        
        # Record specification field  
        if field in _headers.recfieldspecs:
            listcheck = 0
            allowedtypes = _headers.recfieldspecs[field].allowedtypes
        # Segment specification field
        elif field in _headers.segfieldspecs:
            listcheck = 1
            allowedtypes = _headers.segfieldspecs[field].allowedtypes
        # Comments field
        elif field == 'comments':
            listcheck = 1
            allowedtypes = [str]
        # Segment field
        elif field == 'segment':
            listcheck = 1
            allowedtypes = [WFDBrecord]
        
        item = getattr(self, field)

        # List fields and their elements
        if listcheck:
            if type(item)!=list:
                sys.exit('Field: '+field+' must be a list with length equal to nseg')

            for i in item:
                if type(i) not in allowedtypes:
                    print('Each element in field: '+field+' must be one of the following types:', allowedtypes)
                    sys.exit()
        # Non-list fields  
        else:
            if type(item) not in allowedtypes:
                print('Field: '+field+' must be one of the following types:', allowedtypes)
                sys.exit()

    
    # Check the cohesion of the segments field with other fields used to write the record
    def checksegmentcohesion(self):

        # Check that nseg is equal to the length of the segments field
        if self.nseg != len(self.segments):
            sys.exit('Length of segments must match the nseg field')

        for i in range(0, nseg):
            s = self.segments[i]

            # If segment 0 is a layout specification record, check that its file names are all == '~'' 
            if i==0 and self.seglen[0] == 0:
                for filename in s.filename:
                    if filename != '~':
                        sys.exit("Layout specification records must have all filenames named '~'")

            # Check that sampling frequencies all match the one in the master header
            if s.fs != self.fs:
                sys.exit("The fs in each segment must match the overall record's fs")

            # Check the signal length of the segment against the corresponding seglen field
            if s.siglen != self.seglen[i]:
                sys.exit('The signal length of segment '+str(i)+' does not match the corresponding segment length')

            totalsiglen = totalsiglen + getattr(s, 'siglen')

        # No need to check the sum of siglens from each segment object against siglen
        # Already effectively done it when checking sum(seglen) against siglen


# Shortcut functions for wrsamp

#def s_wrsamp()


# Read a WFDB single or multi segment record. Return a WFDBrecord object or WFDBmultirecord object
def rdsamp(recordname, sampfrom=0, sampto=None, channels = None):  

    record = rdheader(recordname)

    # A single segment record
    if type(record) == WFDBrecord:

        # Read signals from the associated dat files that contain wanted channels
        record.d_signals = rddatfiles(record.filename, record.fmt, record.sampfrom, record.sampto, record.channels)
        # Perform dac to get physical signal
        record.p_signals = record.dac()

        # Should we do the channel selection and edit the object fields here? 
        record.alterfields(channels = channels, sampfrom = sampfrom, sampto = sampto)

    # A multi segment record
    else:
        print('on it')

    return record


# Read a WFDB header. Return a WFDBrecord object or WFDBmultirecord object
def rdheader(recordname):  

    # Read the header file. Separate comment and non-comment lines
    headerlines, commentlines = _headers.getheaderlines(recordname)

    # Get fields from record line
    d_rec = _headers.read_rec_line(headerlines[0])

    # Processing according to whether the header is single or multi segment

    # Single segment header - Process signal specification lines
    if d_rec['nseg'] is None:
        # Create a single-segment WFDB record object
        record = WFDBrecord()
        # Read the fields from the signal lines
        d_sig = _headers.read_sig_lines(headerlines[1:])
        # Set the object's fields
        for field in _headers.sigfieldspecs:
            setattr(record, field, d_sig[field])    
    # Multi segment header - Process segment specification lines
    else:
        # Create a multi-segment WFDB record object
        record = WFDBmultirecord()
        # Read the fields from the segment lines
        d_seg = _headers.read_seg_lines(headerlines[1:])    
        # Set the object's fields
        for field in _headers.segfieldspecs:
            setattr(record, field, d_seg[field])  

    # Set the comments field
    record.comments = []
    for line in commentlines:
        record.comments.append(line.strip('\s#'))

    # Set the record line fields
    for field in _headers.recfieldspecs:
        setattr(record, field, d_rec[field])

    return record


# Read the dat files associated with a record that carry wanted channels
def rddatfiles(filenames, fmt, sampfrom, sampto, channels):

    # Get the set of dat files to be read, and
    # the channels that belong to each file. 
    filenames, datchannels = orderedsetlist(filenames)

    # Remove dat files that do not contain wanted channels
    for filename in filenames:
        if datchannels[filename] not in channels:
            filenames.remove(filename)
            del(datchannels[filename])

    # Get the fmt corresponding to each remaining dat file

    # Allocate signal array

    # Read the relevant dat files

    # 

    #for i in range(0, len(filenames)):
            
    #    rddatfile(filenames[i], self.fmt[min(datchannels[filenames[i]])],
    #        self.d_signals[:, min(datchannels[filenames[i]]):max(datchannels[filenames[i]])+1])

# Read a single dat file
def rddatfile(filename, fmt, sampfrom, sampto):



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
    
                
# Display a message to the user, asking whether they would like to continue. 
def request_approval(message):
    
    pyversion = sys.version_info[0]
    if pyversion not in [2, 3]:
        # Exit before printing the message if python is unsupported
        sys.exit('This package is only supported for python 2 and 3')

    print(message)
    answer=[]
    if sys.version_info[0] == 2:
        while answer not in ['y', 'n']:
            answer = raw_input('[y/n]: ')

    else:
        while answer not in ['y', 'n']:
            answer = input('[y/n]: ')


    if answer == 'y':
        return
    else:
        sys.exit('Exiting')

# Returns the unique elements in a list in the order that they appear. 
# Also returns the indices of the original list that correspond to each output element. 
def orderedsetlist(fulllist):
    uniquelist = []
    original_inds = {}

    for i in range(0, len(fulllist)):
        item = fulllist[i]
        # new item
        if item not in uniquelist:
            uniquelist.append(item)
            original_inds[item] = [i]
        # previously seen item
        else:
            original_inds[item].append(i)
    return uniquelist, original_inds

# Returns elements in a list without consecutive repeated values.  
def orderednoconseclist(fulllist):
    noconseclist = [fulllist[0]]
    if len(fulllist) == 1:
        return noconseclist
    for i in fulllist:
        if i!= noconseclist[-1]:
            noconseclist.append(i)
    return noconseclist