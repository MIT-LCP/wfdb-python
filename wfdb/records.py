# For wrheader(), all fields must be already filled in and cohesive with one another other. The signals field will not be used.
# For wrsamp(), the field to use will be d_signals (which is allowed to be empty for 0 channel records). 
# set_p_features and set_d_features use characteristics of the p_signals or d_signals field to fill in other header fields. 
# These are separate from another method 'setdefaults' which the user may call to set default header fields 
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


# The base WFDB class to extend to create Record and MultiRecord. Contains shared helper functions and fields.             
class BaseRecord():
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
            acceptedstring = re.match('-\w+', self.recordname)
            if not acceptedstring or acceptedstring.string != self.recordname:
                sys.exit('recordname must only comprise of letters, digits, dashes, and underscores.')
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
                    
    # Ensure that input read parameters are valid for the record
    def checkreadinputs(self, sampfrom, sampto, channels):

        # Data Type Check
        if type(sampfrom) not in _headers.inttypes:
            sys.exit('sampfrom must be an integer')
        if type(sampto) not in _headers.inttypes:
            sys.exit('sampto must be an integer')
        if type(channels) != list:
            sys.exit('channels must be a list of integers')

        # Duration Ranges
        if sampfrom<0:
            sys.exit('sampfrom must be a non-negative integer')
        if sampfrom>self.siglen:
            sys.exit('sampfrom must be shorter than the signal length')
        if sampto<0:
            sys.exit('sampto must be a non-negative integer')
        if sampto>self.siglen:
            sys.exit('sampto must be shorter than the signal length')
        if sampto<=sampfrom:
            sys.exit('sampto must be greater than sampfrom')   

        # Channel Ranges
        for c in channels:
            if c<0:
                sys.exit('Input channels must all be non-negative integers')
            if c>self.nsig-1:
                sys.exit('Input channels must all be lower than the total number of channels')


# Class for single segment WFDB records.
class Record(BaseRecord, _headers.HeadersMixin, _signals.SignalsMixin):
    
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
        
        super(Record, self).__init__(recordname, nsig,
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


    # Arrange/edit object fields to reflect user channel and/or signal range input
    def arrangefields(self, channels):
        
        # Rearrange signal specification fields
        for field in _headers.sigfieldspecs:
            item = getattr(self, field)
            if item is not self.nsig*[None]:
                setattr(self, field, [item[c] for c in channels]) 

        # Update record specification parameters
        # Important that these get updated after ^^
        self.nsig = len(channels)
        self.siglen = self.d_signals.shape[0]

        # Checksum and initvalue to be updated if present
        if self.checksum is not None:
            self.checksum = self.calc_checksum()
        if self.initvalue is not None:
            self.initvalue = list(self.d_signals[0, :])


# Class for multi segment WFDB records.
class MultiRecord(BaseRecord, _headers.MultiHeadersMixin):
    
    # Constructor
    def __init__(self, segments = None, layout = None,
                 recordname=None, nsig=None, fs=None, 
                 counterfreq=None, basecounter=None, 
                 siglen=None, basetime=None, basedate=None, 
                 segname = None, seglen = None, comments=None):


        super(MultiRecord, self).__init__(recordname, nsig,
                    fs, counterfreq, basecounter, siglen,
                    basetime, basedate, comments)

        self.segments = segments
        self.segname = segname
        self.seglen = seglen

    # Write a multi-segment header, along with headers and dat files for all segments
    def wrsamp(self):
        # Perform field validity and cohesion checks, and write the header file.
        self.wrheader()
        # Perform record validity and cohesion checks, and write the associated segments.
        for seg in self.segments:
            seg.wrsamp()


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
            allowedtypes = [Record]
        
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


    # Determine the segments and the samples 
    # within each segment that have to be read in a 
    # multi-segment record. Called during rdsamp. 
    def requiredsegments(self, sampfrom, sampto, channels):

        # The starting segment with actual samples
        if self.layout == 'Fixed':
            startseg = 0
        else:
            print('startseg is')
            startseg = 1
            print(startseg)
        # Cumulative sum of segment lengths
        cumsumlengths = list(np.cumsum(self.seglen[startseg:]))
        print('cumsumlengths: ', cumsumlengths)
        # First segment
        readsegs = [[sampfrom < cs for cs in cumsumlengths].index(True)]
        # Final segment
        if sampto == cumsumlengths[len(cumsumlengths) - 1]:
            readsegs.append(len(cumsumlengths) - 1)
        else:
            readsegs.append([sampto < cs for cs in cumsumlengths].index(True))

        print('readsegs: ', readsegs)

        # Obtain the sampfrom and sampto to read for each segment
        if readsegs[1] == readsegs[0]:  
            # Only one segment to read
            readsegs = [readsegs[0]]
            readsamps = [[sampfrom, sampto]] # This is wrong?
        else:
            # More than one segment to read
            readsegs = list(range(readsegs[0], readsegs[1]+1)) 
            readsamps = [[0, self.seglen[s + startseg]]
                         for s in readsegs]
            # Starting sample for first segment
            readsamps[0][0] = sampfrom - ([0] + cumsumlengths)[readsegs[0]]
            # End sample for last segment
            readsamps[-1][1] = sampto - ([0] + cumsumlengths)[readsegs[-1]]  

        # Add 1 for variable layout records
        readsegs = list(np.add(readsegs,startseg))

        return (readsegs, readsamps)

    # Get the channel numbers to be read from each segment
    def requiredsignals(self, readsegs, channels, dirname):

        print('self.layout: ', self.layout)
        # Fixed layout. All channels are the same.
        if self.layout == 'Fixed':
            print('in fixed???')
            # Should we bother here with skipping empty segments? 
            # They won't be read anyway. 
            readsigs = [channels]*len(readsegs)
        # Variable layout: figure out channels by matching record names
        else:
            print('in variable')
            readsigs = []
            # The overall layout signal names
            l_signames = self.segments[0].signame
            # The wanted signals
            w_signames = [l_signames[c] for c in channels]

            # For each segment ... 
            for i in range(0, len(readsegs)):
                # Skip empty segments
                if self.segname[readsegs[i]] == '~':
                    readsigs.append(None)
                else:
                    # Get the signal names of the current segment
                    s_signames = rdheader(os.path.join(dirname, self.segname[readsegs[i]])).signame
                    readsigs.append(wanted_siginds(w_signames, s_signames))

        return readsigs

    # Arrange/edit object fields to reflect user channel and/or signal range input
    def arrangefields(self, readsegs, segranges, channels):
        
        # Update seglen values for relevant segments
        for i in range(0, len(readsegs)):
            self.seglen[readsegs[i]] = segranges[i][1] - segranges[i][0]
        
        # Update record specification parameters
        self.nsig = len(channels)
        self.siglen = sum([sr[1]-sr[0] for sr in segranges])

        # Get rid of the segments and segment line parameters
        # outside the desired segment range
        if self.layout == 'Fixed':
            self.segments = self.segments[readsegs[0]:readsegs[-1]+1]
            self.segname = self.segname[readsegs[0]:readsegs[-1]+1]
            self.seglen = self.seglen[readsegs[0]:readsegs[-1]+1]
        else:
            # Keep the layout specifier
            self.segments = [self.segments[0]] + self.segments[readsegs[0]:readsegs[-1]+1]
            self.segname = [self.segname[0]] + self.segname[readsegs[0]:readsegs[-1]+1]
            self.seglen = [self.seglen[0]] + self.seglen[readsegs[0]:readsegs[-1]+1]

    # Convert a MultiRecord object to a Record object
    def multi_to_single(self):

        # The fields to transfer to the new object
        fields = self.__dict__.copy()

        # Remove multirecord fields
        del(fields['segments'])
        del(fields['segname'])
        del(fields['seglen'])
        del(fields['nseg'])

        # The output physical signals
        p_signals = np.zeros([self.siglen, self.nsig])

        # Get the physical samples from each segment

        # Start and end samples in the overall array
        # to place the segment samples into
        startsamps = [0] + list(np.cumsum(self.seglen)[0:-1])
        endsamps = list(np.cumsum(self.seglen))
        
        if self.layout == 'Fixed':
            # Figure out the signal names and units from one of the segments
            for seg in self.segments:
                if seg is not None:
                    fields['signame'] = seg.signame
                    fields['units'] = seg.units
                    break 

            for i in range(0, nseg):
                seg = self.segments[i]

                # Empty segment
                if seg is None:
                    p_signals[startsamps[i]:endsamps[i],:] = np.nan
                # Non-empty segment
                else:
                    if not hasattr(seg, 'p_signals'):
                        seg.p_signals = seg.dac()
                    p_signals[startsamps[i]:endsamps[i],:] = seg.p_signals                 
        # For variable layout, have to get channels by name
        else:
            # Get the signal names from the layout segment
            fields['signame'] = self.segments[0].signame
            fields['units'] = self.segments[0].units

            for i in range(1, self.nseg):
                seg = self.segments[i]

                # Empty segment
                if seg is None:
                    p_signals[startsamps[i]:endsamps[i],:] = np.nan
                # Non-empty segment
                else:
                    # Figure out if there are any channels wanted and 
                    # the output channels they are to be stored in
                    inchannels = []
                    outchannels = []
                    for s in fields['signame']:
                        if s in seg.signame:
                            inchannels.append(seg.signame.index(s))
                            outchannels.append(fields['signame'].index(s))

                    # Segment contains no wanted channels. Fill with nans. 
                    if inchannels == []:
                        p_signals[startsamps[i]:endsamps[i],:] = np.nan
                    # Segment contains wanted channel(s). Transfer samples. 
                    else:
                        if not hasattr(seg, 'p_signals'):
                            seg.p_signals = seg.dac()
                        for ch in range(0, fields['nsig']):
                            if ch not in outchannels:
                                p_signals[startsamps[i]:endsamps[i],ch] = np.nan
                            else:
                                p_signals[startsamps[i]:endsamps[i],ch] = seg.p_signals[:, inchannels[outchannels.index(ch)]]

        # Create the single segment Record object and set attributes
        record = Record()
        for field in fields:
            setattr(record, field, fields[field])
        record.p_signals = p_signals

        return record        

        
#------------------- Reading Records -------------------#

# Read a WFDB single or multi segment record. Return a Record or MultiRecord object
def rdsamp(recordname, sampfrom=0, sampto=None, channels = None, physical = True, stackmulti = True):  

    # If the user specifies a sample or signal range, some fields 
    # of the output object will be updated from the fields directly
    # read from the header, which represent the entire record.

    # The goal of this function is to create an object, or objects, 
    # with description fields that reflect the state its contents 
    # when created. 

    dirname, baserecordname = os.path.split(recordname)

    # Read the header fields into the appropriate record object
    record = rdheader(recordname)

    # Set defaults for sampto and channels
    if sampto is None:
        sampto = record.siglen
    if channels is None:
        channels = list(range(record.nsig))

    # Ensure that input fields are valid for the record
    record.checkreadinputs(sampfrom, sampto, channels)

    # A single segment record
    if type(record) == Record:
        # Read signals from the associated dat files that contain wanted channels
        record.d_signals = _signals.rdsegment(record.filename, record.nsig, record.fmt, record.siglen, 
            record.byteoffset, record.sampsperframe, record.skew,
            sampfrom, sampto, channels, dirname)

        # Arrange/edit the object fields to reflect user channel and/or signal range input
        record.arrangefields(channels)

        if physical == 1:
            # Perform dac to get physical signal
            record.p_signals = record.dac()

    # A multi segment record

    # We can make another rdsamp function (called rdsamp_segment) to call
    # for individual segments to deal with the skews. 
    else:
        # Strategy: 
        # 1. Read the required segments and store them in 
        # Record objects. 
        # 2. Update the parameters of the objects to reflect
        # the state of the sections read.
        # 3. Update the parameters of the overall MultiRecord
        # object to reflect the state of the individual segments.
        # 4. If specified, convert the MultiRecord object
        # into a single Record object.

        # Segments field is a list of Record objects
        # Empty segments store None.

        # If stackmulti == True, Physical must be true. There is no 
        # meaningful representation of digital signals transferred 
        # from individual segments. 
        if stackmulti == True and physical != True:
            sys.exit('If stackmulti is True, physical must also be True.')

        record.segments = [None]*record.nseg

        # Variable layout
        if record.seglen[0] == 0:
            record.layout = 'Variable'
            # Read the layout specification header
            record.segments[0] = rdheader(os.path.join(dirname, record.segname[0]))
        # Fixed layout
        else:
            record.layout = 'Fixed'
            
        # Get the segments numbers, samples, and 
        # channel indices within each segment to read.
        readsegs, segranges  = record.requiredsegments(sampfrom, sampto, channels)
        segsigs = record.requiredsignals(readsegs, channels, dirname) 

        print('readsegs: ', readsegs)
        print('segranges: ', segranges)
        print('segsigs: ', segsigs)

        # Read the desired samples in the relevant segments
        for i in range(0, len(readsegs)):
            segnum = readsegs[i]
            # Empty segment or segment with no relevant channels
            if record.segname[segnum] == '~' or segsigs[i] is None:
                record.segments[segnum] = None 
            else:
                print('recordname: ', os.path.join(dirname, record.segname[segnum]))
                print('sampfrom: ', segranges[i][0])
                print('sampto: ', segranges[i][1])
                print('channels: ', segsigs[i])



                record.segments[segnum] = rdsamp(os.path.join(dirname, record.segname[segnum]), 
                    sampfrom = segranges[i][0], sampto = segranges[i][1], 
                    channels = segsigs[i], physical = physical)

        # Arrange the fields of the overall object to reflect user input
        record.arrangefields(readsegs, segranges, channels)

        # Convert object into a single segment Record object 
        if stackmulti:
            record = record.multi_to_single()

    return record


# Read a WFDB header. Return a Record object or MultiRecord object
def rdheader(recordname):  

    # Read the header file. Separate comment and non-comment lines
    headerlines, commentlines = _headers.getheaderlines(recordname)

    # Get fields from record line
    d_rec = _headers.read_rec_line(headerlines[0])

    # Processing according to whether the header is single or multi segment

    # Single segment header - Process signal specification lines
    if d_rec['nseg'] is None:
        # Create a single-segment WFDB record object
        record = Record()
        # Read the fields from the signal lines
        d_sig = _headers.read_sig_lines(headerlines[1:])
        # Set the object's signal line fields
        for field in _headers.sigfieldspecs:
            setattr(record, field, d_sig[field])   
        # Set the object's record line fields
        for field in _headers.recfieldspecs:
            if field == 'nseg':
                continue
            setattr(record, field, d_rec[field])
    # Multi segment header - Process segment specification lines
    else:
        # Create a multi-segment WFDB record object
        record = MultiRecord()
        # Read the fields from the segment lines
        d_seg = _headers.read_seg_lines(headerlines[1:])    
        # Set the object's segment line fields
        for field in _headers.segfieldspecs:
            setattr(record, field, d_seg[field])  
        # Set the objects' record line fields
        for field in _headers.recfieldspecs:
            setattr(record, field, d_rec[field])
    # Set the comments field
    record.comments = []
    for line in commentlines:
        record.comments.append(line.strip('\s#'))

    return record


# Given some wanted signal names, and the signal names contained
# in a record, return the indices of the record channels that intersect. 
# Remember that the wanted signal names are already in order specified in user input channels. So it's good!
def wanted_siginds(wanted_signames, record_signames):
    contained_signals = [s for s in wanted_signames if s in record_signames]
    if contained_signals == []:
        return None
    else:
        return [record_signames.index(s) for s in contained_signals]


# A simple version of rdsamp for the average user
# Return the physical signals and a few essential fields
def srdsamp(recordname, sampfrom=0, sampto=None, channels = None):

    record = rdsamp(recordname, sampfrom, sampto, channels, True, True)

    signals = record.p_signals
    fields = {}
    for field in ['fs','units','signame']:
        fields[field] = getattr(record, field)

    return signals, fields

#------------------- /Reading Records -------------------#


# Simple function for single segment wrsamp for writing physical signals
def swrsamp(recordname, p_signals, fs, units, signames, fmt = None, comments=[]):
    
    # Create the Record object
    record = Record(recordname = recordname, p_signals = p_signals, fs = fs, fmt = fmt, units = units, signame = signames, comments = comments)
    # 2. Compute optimal fields to store the digital signal, carry out adc, and set the fields.
    record.set_d_features(do_adc = 1)
    # 3. Set default values of any missing field dependencies
    record.setdefaults()
    # 4. Write the record files - header and associated dat
    record.wrsamp()


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