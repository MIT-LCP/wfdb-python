# For wrheader(), all fields must be already filled in and cohesive with one another other. The signals field will not be used.
# For wrsamp(), the field to use will be d_signals (which is allowed to be empty for 0 channel records).
# set_p_features and set_d_features use characteristics of the p_signals or d_signals field to fill in other header fields.
# These are separate from another method 'setdefaults' which the user may call to set default header fields
# The checkfieldcohesion() function will be called in wrheader which checks all the header fields.
# The checksignalcohesion() function will be called in wrsamp in wrdat to check the d_signal against the header fields.

import numpy as np
import re
import os
import posixpath
from collections import OrderedDict
from calendar import monthrange
import requests
import multiprocessing
from . import _headers
from . import _signals
from . import downloads


# The base WFDB class to extend to create Record and MultiRecord. Contains shared helper functions and fields.
class BaseRecord(object):
    # Constructor
    def __init__(self, recordname=None, nsig=None,
                 fs=None, counterfreq=None, basecounter = None,
                 siglen = None, basetime = None, basedate = None,
                 comments = None, signame=None):
        self.recordname = recordname
        self.nsig = nsig
        self.fs = fs
        self.counterfreq = counterfreq
        self.basecounter = basecounter
        self.siglen = siglen
        self.basetime = basetime
        self.basedate = basedate
        self.comments = comments
        self.signame = signame

    # Check whether a single field is valid in its basic form. Does not check compatibility with other fields.
    # ch is only used for signal specification fields, specifying the channels to check. Other channels
    # can be None.
    # Be aware that this function is not just called from wrheader.
    def checkfield(self, field, channels=None):
        # Check that the field is present
        if getattr(self, field) is None:
            raise Exception("Missing field required: "+field)
           
        # Check the type of the field (and of its elements if it should be a list) 
        self.checkfieldtype(field, channels)

        # Expand to make sure all channels must have present field
        if channels == 'all':
            channels = [1]*len(getattr(self, field))

        # Individual specific field checks:
        if field == 'd_signals':
            # Check shape
            if self.d_signals.ndim != 2:
                raise TypeError("d_signals must be a 2d numpy array")
            # Check dtype
            if self.d_signals.dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
                raise TypeError('d_signals must be a 2d numpy array with dtype == int64, int32, int16, or int8.')
        elif field =='p_signals':
            # Check shape
            if self.p_signals.ndim != 2:
                raise TypeError("p_signals must be a 2d numpy array")

        elif field == 'e_d_signals':
            # Check shape
            for ch in range(len(channels)):
                if self.e_d_signals[ch].ndim != 1:
                    raise TypeError("e_d_signals must be a list of 1d numpy arrays")
                # Check dtype
                if self.e_d_signals[ch].dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
                    raise TypeError('e_d_d_signals must be a list of 1d numpy arrays with dtype == int64, int32, int16, or int8.')
        elif field =='e_p_signals':
            # Check shape
            for ch in range(0, len(channels)):
                if self.e_p_signals.ndim != 1:
                    raise TypeError("e_p_signals must be a list of 1d numpy arrays")

        #elif field == 'segments': # Nothing to check here. 
        # Record specification fields
        elif field == 'recordname':
            # Allow letters, digits, hyphens, and underscores.
            acceptedstring = re.match('[-\w]+', self.recordname)
            if not acceptedstring or acceptedstring.string != self.recordname:
                raise ValueError('recordname must only comprise of letters, digits, hyphens, and underscores.')
        elif field == 'nseg':
            if self.nseg <=0:
                raise ValueError('nseg must be a positive integer')
        elif field == 'nsig':
            if self.nsig <=0:
                raise ValueError('nsig must be a positive integer')
        elif field == 'fs':
            if self.fs<=0:
                raise ValueError('fs must be a positive number')
        elif field == 'counterfreq':
            if self.counterfreq <=0:
                raise ValueError('counterfreq must be a positive number')
        elif field == 'basecounter':
            if self.basecounter <=0:
                raise ValueError('basecounter must be a positive number') 
        elif field == 'siglen':
            if self.siglen <0:
                raise ValueError('siglen must be a non-negative integer')
        elif field == 'basetime':
            _ = parsetimestring(self.basetime)
        elif field == 'basedate':
            _ = parsedatestring(self.basedate)

        # Signal specification fields. Lists of elements to check.
        elif field in _headers.sigfieldspecs:

            for ch in range(0, len(channels)):
                f = getattr(self, field)[ch]

                # The channel element is allowed to be None
                if not channels[ch]:
                    if f is None:
                        continue

                if field == 'filename':
                    # Check for filename characters
                    acceptedstring = re.match('[\w]+\.?[\w]+',f)
                    if not acceptedstring or acceptedstring.string != f:
                        raise ValueError('File names should only contain alphanumerics and an extension. eg. record_100.dat')
                    # Check that dat files are grouped together 
                    if orderedsetlist(self.filename)[0] != orderednoconseclist(self.filename):
                        raise ValueError('filename error: all entries for signals that share a given file must be consecutive')
                elif field == 'fmt':
                    if f not in _signals.datformats:
                        raise ValueError('File formats must be valid WFDB dat formats: '+' , '.join(_signals.datformats))    
                elif field == 'sampsperframe':
                    if f < 1:
                        raise ValueError('sampsperframe values must be positive integers')
                elif field == 'skew':
                    if f < 0:
                        raise ValueError('skew values must be non-negative integers')
                elif field == 'byteoffset':
                    if f < 0:
                        raise ValueError('byteoffset values must be non-negative integers')
                elif field == 'adcgain':
                    if f <= 0:
                        raise ValueError('adcgain values must be positive numbers')
                elif field == 'baseline':
                    # Currently original WFDB library only has 4 bytes for baseline.
                    if f < -2147483648 or f> 2147483648:
                        raise ValueError('baseline values must be between -2147483648 (-2^31) and 2147483647 (2^31 -1)')
                elif field == 'units':
                    if re.search('\s', f):
                        raise ValueError('units strings may not contain whitespaces.')
                elif field == 'adcres':
                    if f < 0:
                        raise ValueError('adcres values must be non-negative integers')
                # elif field == 'adczero': nothing to check here
                # elif field == 'initvalue': nothing to check here
                # elif field == 'checksum': nothing to check here
                elif field == 'blocksize':
                    if f < 0:
                        raise ValueError('blocksize values must be non-negative integers')
                elif field == 'signame':
                    if re.search('\s', f):
                        raise ValueError('signame strings may not contain whitespaces.')
                    if len(set(self.signame)) != len(self.signame):
                        raise ValueError('signame strings must be unique.')

        # Segment specification fields
        elif field == 'segname':
            # Segment names must be alphanumerics or just a single '~'
            for f in self.segname:
                if f == '~':
                    continue
                acceptedstring = re.match('[-\w]+',f)
                if not acceptedstring or acceptedstring.string != f:
                    raise ValueError("Non-null segment names may only contain alphanumerics and dashes. Null segment names must be set to '~'")
        elif field == 'seglen':
            # For records with more than 1 segment, the first segment may be
            # the layout specification segment with a length of 0
            if len(self.seglen)>1:
                if self.seglen[0] < 0:
                    raise ValueError('seglen values must be positive integers. Only seglen[0] may be 0 to indicate a layout segment')
                sl = self.seglen[1:]
            else:
                sl = self.seglen
            for f in sl:
                if f < 1:
                    raise ValueError('seglen values must be positive integers. Only seglen[0] may be 0 to indicate a layout segment')
        # Comment field
        elif field == 'comments':
            for f in self.comments:
                if f=='': # Allow empty string comment lines
                    continue
                if f[0] == '#':
                    print("Note: comment strings do not need to begin with '#'. This library adds them automatically.")
                if re.search('[\t\n\r\f\v]', f):
                    raise ValueError('comments may not contain tabs or newlines (they may contain spaces and underscores).')
    # Check the data type of the specified field.
    # ch is used for signal spec fields
    # Some fields are lists. This must be checked, along with their elements.
    def checkfieldtype(self, field, ch=None):

        item = getattr(self, field)

        # Record specification field. Nonlist.
        if field in _headers.recfieldspecs:
            checkitemtype(item, field, _headers.recfieldspecs[field].allowedtypes)

        # Signal specification field. List.
        elif field in _headers.sigfieldspecs:
            checkitemtype(item, field, _headers.sigfieldspecs[field].allowedtypes, ch)

        # Segment specification field. List. All elements cannot be None
        elif field in _headers.segfieldspecs:
            checkitemtype(item, field, _headers.segfieldspecs[field].allowedtypes, 'all')

        # Comments field. List. Elements cannot be None
        elif field == 'comments':
            checkitemtype(item, field, [str], 'all')

        # Signals field.
        elif field in ['p_signals','d_signals']:
            checkitemtype(item, field, [np.ndarray])

        elif field in ['e_p_signals', 'e_d_signals']:
            checkitemtype(item, field, [np.ndarray], 'all')

        # Segments field. List. Elements may be None.
        elif field == 'segments':
            checkitemtype(item, field, [Record], 'none')

    # Ensure that input read parameters are valid for the record
    def checkreadinputs(self, sampfrom, sampto, channels, physical, m2s, smoothframes):
        # Data Type Check
        if not hasattr(sampfrom, '__index__'):
            raise TypeError('sampfrom must be an integer')
        if not hasattr(sampto, '__index__'):
            raise TypeError('sampto must be an integer')

        if type(channels) != list:
            raise TypeError('channels must be a list of integers')

        # Duration Ranges
        if sampfrom<0:
            raise ValueError('sampfrom must be a non-negative integer')
        if sampfrom>self.siglen:
            raise ValueError('sampfrom must be shorter than the signal length')
        if sampto<0:
            raise ValueError('sampto must be a non-negative integer')
        if sampto>self.siglen:
            raise ValueError('sampto must be shorter than the signal length')
        if sampto<=sampfrom:
            raise ValueError('sampto must be greater than sampfrom')

        # Channel Ranges
        for c in channels:
            if c<0:
                raise ValueError('Input channels must all be non-negative integers')
            if c>self.nsig-1:
                raise ValueError('Input channels must all be lower than the total number of channels')

        # Cannot expand multiple samples/frame for multi-segment records
        if type(self) == MultiRecord:

            # If m2s == True, Physical must be true. There is no
            # meaningful representation of digital signals transferred
            # from individual segments.
            if m2s is True and physical is not True:
                raise Exception('If m2s is True, physical must also be True.')

            if smoothframes is False:
                raise ValueError('This package version cannot expand all samples when reading multi-segment records. Must enable frame smoothing.')

# Check the item type. Vary the print message regarding whether the item can be None.
# Helper to checkfieldtype
# channels is a list of booleans indicating whether the field's channel must be present (1) or may be None (0)
# and is not just for signal specification fields
def checkitemtype(item, field, allowedtypes, channels=None):

    # Checking the list
    if channels is not None:

        # First make sure the item is a list
        if type(item) != list:
            raise TypeError("Field: '"+field+"' must be a list")

        # Expand to make sure all channels must have present field
        if channels == 'all':
            channels = [1]*len(item)

        # Expand to allow any channel to be None
        if channels == 'none':
            channels = [0]*len(item)

        for ch in range(0, len(channels)):

            mustexist=channels[ch]
            # The field must exist for the channel
            if mustexist:
                if type(item[ch]) not in allowedtypes:
                    raise TypeError("Channel "+str(ch)+" of field: '"+field+"' must be one of the following types:", allowedtypes)

            # The field may be None for the channel
            else:
                if type(item[ch]) not in allowedtypes and item[ch] is not None:
                    raise TypeError("Channel "+str(ch)+" of field: '"+field+"' must be a 'None', or one of the following types:", allowedtypes)

    # Single scalar to check
    else:
        if type(item) not in allowedtypes:
            raise TypeError("Field: '"+field+"' must be one of the following types:", allowedtypes)



class Record(BaseRecord, _headers.HeadersMixin, _signals.SignalsMixin):
    """
    The class representing WFDB headers, and single segment WFDB records.

    Record objects can be created using the constructor, by reading a WFDB header
    with 'rdheader', or a WFDB record (header and associated dat files) with rdsamp'
    or 'srdsamp'.

    The attributes of the Record object give information about the record as specified
    by https://www.physionet.org/physiotools/wag/header-5.htm

    In addition, the d_signals and p_signals attributes store the digital and physical
    signals of WFDB records with at least one channel.

    Contructor function:
    def __init__(self, p_signals=None, d_signals=None,
                 recordname=None, nsig=None,
                 fs=None, counterfreq=None, basecounter=None,
                 siglen=None, basetime=None, basedate=None,
                 filename=None, fmt=None, sampsperframe=None,
                 skew=None, byteoffset=None, adcgain=None,
                 baseline=None, units=None, adcres=None,
                 adczero=None, initvalue=None, checksum=None,
                 blocksize=None, signame=None, comments=None)

    Example Usage:
    import wfdb
    record1 = wfdb.Record(recordname='r1', fs=250, nsig=2, siglen=1000, filename=['r1.dat','r1.dat'])

    """
    # Constructor
    def __init__(self, p_signals=None, d_signals=None,
                 e_p_signals=None, e_d_signals=None,
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
                    basetime, basedate, comments, signame)

        self.p_signals = p_signals
        self.d_signals = d_signals
        self.e_p_signals = e_p_signals
        self.e_d_signals = e_d_signals       


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

    # Equal comparison operator for objects of this type
    def __eq__(self, other):
        att1 = self.__dict__
        att2 = other.__dict__

        if set(att1.keys()) != set(att2.keys()):
            return False

        for k in att1.keys():

            v1 = att1[k]
            v2 = att2[k]

            if type(v1) != type(v2):
                return False

            if type(v1) == np.ndarray:
                if not np.array_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False

        return True

    # Write a wfdb header file and associated dat files if any.
    # Uses d_signals (expanded=False) or e_d_signals to write the samples
    def wrsamp(self, expanded=False):

        # Perform field validity and cohesion checks, and write the header file.
        self.wrheader()
        if self.nsig>0:
            # Perform signal validity and cohesion checks, and write the associated dat files.
            self.wrdats(expanded)


    # Arrange/edit object fields to reflect user channel and/or signal range input
    # Account for case when signals are expanded
    def arrangefields(self, channels, expanded=False):

        # Rearrange signal specification fields
        for field in _headers.sigfieldspecs:
            item = getattr(self, field)
            setattr(self, field, [item[c] for c in channels])

        # Expanded signals - multiple samples per frame.
        if expanded:
            # Checksum and initvalue to be updated if present
            # unless the whole signal length was input
            if self.siglen != int(len(self.e_d_signals[0])/self.sampsperframe[0]):
                self.checksum = self.calc_checksum(expanded)
                self.initvalue = [s[0] for s in self.e_d_signals]

            self.nsig = len(channels)
            self.siglen = int(len(self.e_d_signals[0])/self.sampsperframe[0])

        # MxN numpy array d_signals
        else:
            # Checksum and initvalue to be updated if present
            # unless the whole signal length was input
            if self.siglen != self.d_signals.shape[0]:

                if self.checksum is not None:
                    self.checksum = self.calc_checksum()
                if self.initvalue is not None:
                    ival = list(self.d_signals[0, :])
                    self.initvalue = [int(i) for i in ival]

            # Update record specification parameters
            # Important that these get updated after^^
            self.nsig = len(channels)
            self.siglen = self.d_signals.shape[0]



# Class for multi segment WFDB records.
class MultiRecord(BaseRecord, _headers.MultiHeadersMixin):
    """
    The class representing multi-segment WFDB records.

    MultiRecord objects can be created using the constructor, or by reading a multi-segment
    WFDB record using 'rdsamp' with the 'm2s' (multi to single) input parameter set to False.

    The attributes of the MultiRecord object give information about the entire record as specified
    by https://www.physionet.org/physiotools/wag/header-5.htm

    In addition, the 'segments' parameter is a list of Record objects representing each
    individual segment, or 'None' representing empty segments, of the entire multi-segment record.

    Noteably, this class has no attribute representing the signals as a whole. The 'multi_to_single'
    instance method can be called on MultiRecord objects to return a single segment representation
    of the record as a Record object. The resulting Record object will have its 'p_signals' field set.

    Contructor function:
    def __init__(self, segments=None, layout=None,
                 recordname=None, nsig=None, fs=None,
                 counterfreq=None, basecounter=None,
                 siglen=None, basetime=None, basedate=None,
                 segname=None, seglen=None, comments=None,
                 signame=None, sigsegments=None)

    Example Usage:
    import wfdb
    recordM = wfdb.MultiRecord(recordname='rm', fs=50, nsig=8, siglen=9999,
                               segname=['rm_1', '~', rm_2'], seglen=[800, 200, 900])

    recordL = wfdb.rdsamp('s00001-2896-10-10-00-31', m2s = False)
    recordL = recordL.multi_to_single()
    """

    # Constructor
    def __init__(self, segments=None, layout=None,
                 recordname=None, nsig=None, fs=None,
                 counterfreq=None, basecounter=None,
                 siglen=None, basetime=None, basedate=None,
                 segname=None, seglen=None, comments=None,
                 signame=None, sigsegments=None):


        super(MultiRecord, self).__init__(recordname, nsig,
                    fs, counterfreq, basecounter, siglen,
                    basetime, basedate, comments, signame)

        self.layout = layout
        self.segments = segments
        self.segname = segname
        self.seglen = seglen
        self.sigsegments=sigsegments

    # Write a multi-segment header, along with headers and dat files for all segments
    def wrsamp(self):
        # Perform field validity and cohesion checks, and write the header file.
        self.wrheader()
        # Perform record validity and cohesion checks, and write the associated segments.
        for seg in self.segments:
            seg.wrsamp()


    # Check the cohesion of the segments field with other fields used to write the record
    def checksegmentcohesion(self):

        # Check that nseg is equal to the length of the segments field
        if self.nseg != len(self.segments):
            raise ValueError("Length of segments must match the 'nseg' field")

        for i in range(0, nseg):
            s = self.segments[i]

            # If segment 0 is a layout specification record, check that its file names are all == '~''
            if i==0 and self.seglen[0] == 0:
                for filename in s.filename:
                    if filename != '~':
                        raise ValueError("Layout specification records must have all filenames named '~'")

            # Check that sampling frequencies all match the one in the master header
            if s.fs != self.fs:
                raise ValueError("The 'fs' in each segment must match the overall record's 'fs'")

            # Check the signal length of the segment against the corresponding seglen field
            if s.siglen != self.seglen[i]:
                raise ValueError('The signal length of segment '+str(i)+' does not match the corresponding segment length')

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
            startseg = 1

        # Cumulative sum of segment lengths (ignoring layout segment)
        cumsumlengths = list(np.cumsum(self.seglen[startseg:]))
        # Get first segment
        readsegs = [[sampfrom < cs for cs in cumsumlengths].index(True)]
        # Get final segment
        if sampto == cumsumlengths[len(cumsumlengths) - 1]:
            readsegs.append(len(cumsumlengths) - 1)
        else:
            readsegs.append([sampto <= cs for cs in cumsumlengths].index(True))

        # Add 1 for variable layout records
        readsegs = list(np.add(readsegs,startseg))

        # Obtain the sampfrom and sampto to read for each segment
        if readsegs[1] == readsegs[0]:
            # Only one segment to read
            readsegs = [readsegs[0]]
            # The segment's first sample number relative to the entire record
            segstartsamp = sum(self.seglen[0:readsegs[0]])
            readsamps = [[sampfrom-segstartsamp, sampto-segstartsamp]]

        else:
            # More than one segment to read
            readsegs = list(range(readsegs[0], readsegs[1]+1))
            readsamps = [[0, self.seglen[s]] for s in readsegs]

            # Starting sample for first segment.
            readsamps[0][0] = sampfrom - ([0] + cumsumlengths)[readsegs[0]-startseg]

            # End sample for last segment
            readsamps[-1][1] = sampto - ([0] + cumsumlengths)[readsegs[-1]-startseg]

        return (readsegs, readsamps)

    # Get the channel numbers to be read from each segment
    def requiredsignals(self, readsegs, channels, dirname, pbdir):

        # Fixed layout. All channels are the same.
        if self.layout == 'Fixed':
            # Should we bother here with skipping empty segments?
            # They won't be read anyway.
            readsigs = [channels]*len(readsegs)
        # Variable layout: figure out channels by matching record names
        else:
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
                    s_signames = rdheader(os.path.join(dirname, self.segname[readsegs[i]]), pbdir = pbdir).signame
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
            # Keep the layout specifier segment
            self.segments = [self.segments[0]] + self.segments[readsegs[0]:readsegs[-1]+1]
            self.segname = [self.segname[0]] + self.segname[readsegs[0]:readsegs[-1]+1]
            self.seglen = [self.seglen[0]] + self.seglen[readsegs[0]:readsegs[-1]+1]

        # Update number of segments
        self.nseg = len(self.segments)

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

            for i in range(0, self.nseg):
                seg = self.segments[i]

                # Fixed layout signals cannot have empty segments
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
def rdsamp(recordname, sampfrom=0, sampto=None, channels = None, physical = True, pbdir = None,
           m2s = True, smoothframes = True, ignoreskew=False):
    """Read a WFDB record and return the signal and record descriptors as attributes in a
    Record or MultiRecord object.

    Usage:
    record = rdsamp(recordname, sampfrom=0, sampto=None, channels=None, physical=True, pbdir = None,
             m2s=True, smoothframes = True, ignoreskew=False)

    Input arguments:
    - recordname (required): The name of the WFDB record to be read (without any file extensions).
      If the argument contains any path delimiter characters, the argument will be interpreted as
      PATH/baserecord and the data files will be searched for in the local path.
    - sampfrom (default=0): The starting sample number to read for each channel.
    - sampto (default=None): The sample number at which to stop reading for each channel.
    - channels (default=all): Indices specifying the channel to be returned.
    - physical (default=True): Flag that specifies whether to return signals in physical units in
      the p_signals field (True), or digital units in the d_signals field (False).
    - pbdir (default=None): Option used to stream data from Physiobank. The Physiobank database
       directory from which to find the required record files.
      eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb', pbdir = 'mitdb'.
    - m2s (default=True): Flag used when reading multi-segment records. Specifies whether to
      directly return a wfdb MultiRecord object (False), or to convert it into and return a wfdb
      Record object (True).
    - smoothframes (default=True): Flag used when reading records with signals having multiple
      samples per frame. Specifies whether to smooth the samples in signals with more than
      one sample per frame and return an mxn uniform numpy array as the d_signals or p_signals
      field (True), or to return a list of 1d numpy arrays containing every expanded sample as
      the e_d_signals or e_p_signals field (False).
    - ignoreskew (default=False): Flag used when reading records with at least one skewed signal.
      Specifies whether to apply the skew to align the signals in the output variable (False), or
      to ignore the skew field and load in all values contained in the dat files unaligned (True).

    Output argument:
    - record: The wfdb Record or MultiRecord object representing the contents of the record read.

    Note: If a signal range or channel selection is specified when calling this function, the
          the resulting attributes of the returned object will be set to reflect the section
          of the record that is actually read, rather than necessarily what is in the header file.
          For example, if channels = [0, 1, 2] is specified when reading a 12 channel record, the
          'nsig' attribute will be 3, not 12.

    Note: The 'srdsamp' function exists as a simple alternative to 'rdsamp' for the most common
          purpose of extracting the physical signals and a few important descriptor fields.
          'srdsamp' returns two arguments: the physical signals array, and a dictionary of a
          few select fields, a subset of the original wfdb Record attributes.

    Example Usage:
    import wfdb
    ecgrecord = wfdb.rdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])
    """

    dirname, baserecordname = os.path.split(recordname)

    # Read the header fields into the appropriate record object
    record = rdheader(recordname, pbdir = pbdir, rdsegments = False)

    # Set defaults for sampto and channels input variables
    if sampto is None:
        sampto = record.siglen
    if channels is None:
        channels = list(range(record.nsig))

    # Ensure that input fields are valid for the record
    record.checkreadinputs(sampfrom, sampto, channels, physical, m2s, smoothframes)

    # A single segment record
    if type(record) == Record:

        # Only 1 sample/frame, or frames are smoothed. Return uniform numpy array
        if smoothframes or max([record.sampsperframe[c] for c in channels])==1:
            # Read signals from the associated dat files that contain wanted channels
            record.d_signals = _signals.rdsegment(record.filename, dirname, pbdir, record.nsig, record.fmt, record.siglen,
                                                  record.byteoffset, record.sampsperframe, record.skew,
                                                  sampfrom, sampto, channels, smoothframes, ignoreskew)
            
            # Arrange/edit the object fields to reflect user channel and/or signal range input
            record.arrangefields(channels, expanded=False)
            # Obtain physical values
            if physical == 1:
                # Perform dac to get physical signal
                record.p_signals = record.dac(expanded=False)
                # Clear memory
                record.d_signals = None
            # If the frames had to be smoothed, and d_signals is desired, 
            # the dtype must be cast back into int
            elif max(record.sampsperframe)>1:
                record.d_signals = record.d_signals.astype('int64')

        # Return each sample of the signals with multiple samples per frame
        else:
            record.e_d_signals = _signals.rdsegment(record.filename, dirname, pbdir, record.nsig, record.fmt, record.siglen,
                                                    record.byteoffset, record.sampsperframe, record.skew,
                                                    sampfrom, sampto, channels, smoothframes, ignoreskew)

            # Arrange/edit the object fields to reflect user channel and/or signal range input
            record.arrangefields(channels, expanded=True)
            # Obtain physical values
            if physical == 1:
                # Perform dac to get physical signal
                record.e_p_signals = record.dac(expanded=True)
                # Clear memory
                record.e_d_signals = None

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

        record.segments = [None]*record.nseg

        # Variable layout
        if record.seglen[0] == 0:
            record.layout = 'Variable'
            # Read the layout specification header
            record.segments[0] = rdheader(os.path.join(dirname, record.segname[0]), pbdir=pbdir)
        # Fixed layout
        else:
            record.layout = 'Fixed'

        # The segment numbers and samples within each segment to read.
        readsegs, segranges  = record.requiredsegments(sampfrom, sampto, channels)
        # The signals within each segment to read
        segsigs = record.requiredsignals(readsegs, channels, dirname, pbdir)

        # Read the desired samples in the relevant segments
        for i in range(0, len(readsegs)):
            segnum = readsegs[i]
            # Empty segment or segment with no relevant channels
            if record.segname[segnum] == '~' or segsigs[i] is None:
                record.segments[segnum] = None
            else:
                record.segments[segnum] = rdsamp(os.path.join(dirname, record.segname[segnum]),
                    sampfrom = segranges[i][0], sampto = segranges[i][1],
                    channels = segsigs[i], physical = physical, pbdir=pbdir)

        # Arrange the fields of the overall object to reflect user input
        record.arrangefields(readsegs, segranges, channels)

        # Convert object into a single segment Record object
        if m2s:
            record = record.multi_to_single()

    return record


# Read a WFDB header. Return a Record object or MultiRecord object
def rdheader(recordname, pbdir = None, rdsegments = False):
    """Read a WFDB header file and return the record descriptors as attributes in a Record object

    Usage:
    record = rdheader(recordname, pbdir = None, rdsegments = False)

    Input arguments:
    - recordname (required): The name of the WFDB record to be read (without any file extensions).
      If the argument contains any path delimiter characters, the argument will be interpreted as
      PATH/baserecord and the header file will be searched for in the local path.
    - pbdir (default=None): Option used to stream data from Physiobank. The Physiobank database
       directory from which to find the required record files.
      eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb', pbdir = 'mitdb'.
    - rdsegments (default=False): Boolean flag used when reading multi-segment headers. If True,
      segment headers will also be read (into the record object's 'segments' field).

    Output argument:
    - record: The wfdb Record or MultiRecord object representing the contents of the header read.

    Example Usage:
    import wfdb
    ecgrecord = wfdb.rdheader('sampledata/test01_00s', sampfrom=800, channels = [1,3])
    """

    # Read the header file. Separate comment and non-comment lines
    headerlines, commentlines = _headers.getheaderlines(recordname, pbdir)

    # Get fields from record line
    d_rec = _headers.read_rec_line(headerlines[0])

    # Processing according to whether the header is single or multi segment

    # Single segment header - Process signal specification lines
    if d_rec['nseg'] is None:
        # Create a single-segment WFDB record object
        record = Record()

        # There is at least one channel
        if len(headerlines)>1:
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
        # Determine whether the record is fixed or variable
        if record.seglen[0] == 0:
            record.layout = 'Variable'
        else:
            record.layout = 'Fixed'

        # If specified, read the segment headers
        if rdsegments:
            record.segments = []
            # Get the base record name (could be empty)
            dirname = os.path.split(recordname)[0]
            for s in record.segname:
                if s == '~':
                    record.segments.append(None)
                else:
                    record.segments.append(rdheader(os.path.join(dirname,s), pbdir))
            # Fill in the signame attribute
            record.signame = record.getsignames()
            # Fill in the sigsegments attribute
            record.sigsegments = record.getsigsegments()

    # Set the comments field
    record.comments = []
    for line in commentlines:
        record.comments.append(line.strip(' \t#'))

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


# A simple version of rdsamp for ease of use
# Return the physical signals and a few essential fields
def srdsamp(recordname, sampfrom=0, sampto=None, channels = None, pbdir = None):
    """Read a WFDB record and return the physical signal and a few important descriptor fields

    Usage:
    signals, fields = srdsamp(recordname, sampfrom=0, sampto=None, channels=None, pbdir=None)

    Input arguments:
    - recordname (required): The name of the WFDB record to be read (without any file extensions).
      If the argument contains any path delimiter characters, the argument will be interpreted as
      PATH/baserecord and the data files will be searched for in the local path.
    - sampfrom (default=0): The starting sample number to read for each channel.
    - sampto (default=None): The sample number at which to stop reading for each channel.
    - channels (default=all): Indices specifying the channel to be returned.

    Output arguments:
    - signals: A 2d numpy array storing the physical signals from the record.
    - fields: A dictionary specifying several key attributes of the read record:
        - fs: The sampling frequency of the record
        - units: The units for each channel
        - signame: The signal name for each channel
        - comments: Any comments written in the header

    Note: If a signal range or channel selection is specified when calling this function, the
          the resulting attributes of the returned object will be set to reflect the section
          of the record that is actually read, rather than necessarily what is in the header file.
          For example, if channels = [0, 1, 2] is specified when reading a 12 channel record, the
          'nsig' attribute will be 3, not 12.

    Note: The 'rdsamp' function is the base function upon which this one is built. It returns
          all attributes present, along with the signals, as attributes in a wfdb.Record object.
          The function, along with the returned data type, have more options than 'srdsamp' for
          users who wish to more directly manipulate WFDB files.

    Example Usage:
    import wfdb
    sig, fields = wfdb.srdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])
    """

    record = rdsamp(recordname, sampfrom, sampto, channels, True, pbdir, True)

    signals = record.p_signals
    fields = {}
    for field in ['fs','units','signame', 'comments']:
        fields[field] = getattr(record, field)

    return signals, fields

#------------------- /Reading Records -------------------#


# Function for writing single segment records
def wrsamp(recordname, fs, units, signames, p_signals = None, d_signals = None,
    fmt = None, gain = None, baseline = None, comments= None):
    """Write a single segment WFDB record, creating a WFDB header file and any associated dat files.

    Usage:
    wrsamp(recordname, fs, units, signames, p_signals = None, d_signals=None,
           fmt = None, gain = None, baseline = None, comments = None)

    Input arguments:
    - recordname (required): The string name of the WFDB record to be written (without any file extensions).
    - fs (required): The numerical sampling frequency of the record.
    - units (required): A list of strings giving the units of each signal channel.
    - signames (required): A list of strings giving the signal name of each signal channel.
    - p_signals (default=None): An MxN 2d numpy array, where M is the signal length. Gives the physical signal
      values intended to be written. Either p_signals or d_signals must be set, but not both. If p_signals
      is set, this method will use it to perform analogue-digital conversion, writing the resultant digital
      values to the dat file(s). If fmt is set, gain and baseline must be set or unset together. If fmt is
      unset, gain and baseline must both be unset.
    - d_signals (default=None): An MxN 2d numpy array, where M is the signal length. Gives the digital signal
      values intended to be directly written to the dat file(s). The dtype must be an integer type. Either
      p_signals or d_signals must be set, but not both. In addition, if d_signals is set, fmt, gain and baseline
      must also all be set.
    - fmt (default=None): A list of strings giving the WFDB format of each file used to store each channel.
      Accepted formats are: "80","212","16","24", and "32". There are other WFDB formats but this library
      will not write (though it will read) those file types.
    - gain (default=None): A list of integers specifying the DAC/ADC gain.
    - baseline (default=None): A list of integers specifying the digital baseline.
    - comments (default-None): A list of string comments to be written to the header file.

    Note: This gateway function was written to enable a simple way to write WFDB record files using
          the most frequently used parameters. Therefore not all WFDB fields can be set via this function.

          For more control over attributes, create a wfdb.Record object, manually set its attributes, and
          call its wrsamp() instance method. If you choose this more advanced method, see also the setdefaults,
          set_d_features, and set_p_features instance methods to help populate attributes.

    Example Usage (with the most common scenario of input parameters):
    import wfdb
    # Read part of a record from Physiobank
    sig, fields = wfdb.srdsamp('a103l', sampfrom = 50000, channels = [0,1], pbdir = 'challenge/2015/training')
    # Write a local WFDB record (manually inserting fields)
    wfdb.wrsamp('ecgrecord', fs = 250, units = ['mV', 'mV'], signames = ['I', 'II'], p_signals = sig, fmt = ['16', '16'])
    """

    # Check input field combinations
    if p_signals is not None and d_signals is not None:
        raise Exception('Must only give one of the inputs: p_signals or d_signals')
    if d_signals is not None:
        if fmt is None or gain is None or baseline is None:
            raise Exception("When using d_signals, must also specify 'fmt', 'gain', and 'baseline' fields.")
    # Depending on whether d_signals or p_signals was used, set other required features.
    if p_signals is not None:
        # Create the Record object
        record = Record(recordname = recordname, p_signals = p_signals, fs = fs, fmt = fmt, units = units,
                    signame = signames, adcgain = gain, baseline = baseline, comments = comments)
        # Compute optimal fields to store the digital signal, carry out adc, and set the fields.
        record.set_d_features(do_adc = 1)
    else:
        # Create the Record object
        record = Record(recordname = recordname, d_signals = d_signals, fs = fs, fmt = fmt, units = units,
                    signame = signames, adcgain = gain, baseline = baseline, comments = comments)
        # Use d_signals to set the fields directly
        record.set_d_features()

    # Set default values of any missing field dependencies
    record.setdefaults()
    # Write the record files - header and associated dat
    record.wrsamp()


# Time string parser for WFDB header - H(H):M(M):S(S(.sss)) format.
def parsetimestring(timestring):
    times = re.findall("(?P<hours>\d{1,2}):(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}[.\d+]*)", timestring)

    if not times:
        raise ValueError("Invalid time string: "+timestring+". Acceptable format is: 'Hours:Minutes:Seconds'")
    else:
        hours, minutes, seconds = times[0]

    if not hours or not minutes or not seconds:
        raise ValueError("Invalid time string: "+timestring+". Acceptable format is: 'Hours:Minutes:Seconds'")
        
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)

    if int(hours) >23:
        raise ValueError('hours must be < 24')
    elif hours<0:
        raise ValueError('hours must be positive')
    if minutes>59:
        raise ValueError('minutes must be < 60') 
    elif minutes<0:
        raise ValueError('minutes must be positive')  
    if seconds>59:
        raise ValueError('seconds must be < 60')
    elif seconds<0:
        raise ValueError('seconds must be positive')

    return (hours, minutes, seconds)

# Date string parser for WFDB header - DD/MM/YYYY
def parsedatestring(datestring):
    dates = re.findall(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})", datestring)

    if not dates:
        raise ValueError("Invalid date string. Acceptable format is: 'DD/MM/YYYY'")
    else:
        day, month, year = dates[0]

    day = int(day)
    month = int(month)
    year = int(year)

    if year<1:
        raise ValueError('year must be positive')
    if month<1 or month>12:
        raise ValueError('month must be between 1 and 12')
    if day not in range(1, monthrange(year, month)[1]+1):
        raise ValueError('day does not exist for specified year and month')
    
    return (day, month, year)

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




# *These downloading files gateway function rely on the Record/MultiRecord objects.
# They are placed here rather than in downloads.py in order to avoid circular imports


# Download WFDB files from a physiobank database
# This function only targets databases with WFDB records (EDF and MIT format).
# If the database doesn't have a 'RECORDS" file, it will fail.
def dldatabase(pbdb, dlbasedir, records = 'all', annotators = 'all' , keepsubdirs = True, overwrite = False):
    """Download WFDB record (and optionally annotation) files from a Physiobank database. The database
    must contain a 'RECORDS' file in its base directory which lists its WFDB records.

    Usage:
    dldatabase(pbdb, dlbasedir, records = 'all', annotators = 'all' , keepsubdirs = True, overwrite = False)

    Input arguments:
    - pbdb (required): The Physiobank database directory to download.
      eg. For database 'http://physionet.org/physiobank/database/mitdb', pbdb = 'mitdb'.
    - dlbasedir (required): The full local directory path in which to download the files.
    - records (default='all'): Specifier of the WFDB records to download. Is either a list of strings
      which each specify a record, or 'all' to download all records listed in the database's RECORDS file.
      eg. records = ['test01_00s', test02_45s] for database https://physionet.org/physiobank/database/macecgdb/
    - annotators (default='all'): Specifier of the WFDB annotation file types to download along with
      the record files. Is either None to skip downloading any annotations, 'all' to download all
      annotation types as specified by the ANNOTATORS file, or a list of strings which each specify an
      annotation extension.
      eg. annotators = ['anI'] for database https://physionet.org/physiobank/database/prcp/
    - keepsubdirs (default=True): Whether to keep the relative subdirectories of downloaded files
      as they are organized in Physiobank (True), or to download all files into the same base directory (False).
    - overwrite (default=False): If set to True, all files will be redownloaded regardless. If set to False,
      existing files with the same name and relative subdirectory will be checked. If the local file is
      the same size as the online file, the download is skipped. If the local file is larger, it will be deleted
      and the file will be redownloaded. If the local file is smaller, the file will be assumed to be
      partially downloaded and the remaining bytes will be downloaded and appended.

    Example Usage:
    import wfdb
    wfdb.dldatabase('ahadb', os.getcwd())
    """

    # Full url physiobank database
    dburl = posixpath.join(downloads.dbindexurl, pbdb)
    # Check if the database is valid
    r = requests.get(dburl)
    r.raise_for_status()


    # Get the list of records
    recordlist = downloads.getrecordlist(dburl, records)
    # Get the annotator extensions
    annotators = downloads.getannotators(dburl, annotators)

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
            record = rdheader(baserecname, pbdir = posixpath.join(pbdb, dirname))

            # Single segment record
            if type(record) == Record:
                # Add all dat files of the segment
                for file in record.filename:
                    allfiles.append(posixpath.join(dirname, file))

            # Multi segment record
            else:
                for seg in record.segname:
                    # Skip empty segments
                    if seg == '~':
                        continue
                    # Add the header
                    allfiles.append(posixpath.join(dirname, seg+'.hea'))
                    # Layout specifier has no dat files
                    if seg.endswith('_layout'):
                        continue
                    # Add all dat files of the segment
                    recseg = rdheader(seg, pbdir = posixpath.join(pbdb, dirname))
                    for file in recseg.filename:
                        allfiles.append(posixpath.join(dirname, file))
        # check whether the record has any requested annotation files
        if annotators is not None:
            for a in annotators:
                annfile = rec+'.'+a
                url = posixpath.join(downloads.dbindexurl, pbdb, annfile)
                rh = requests.head(url)

                if rh.status_code != 404:
                    allfiles.append(annfile)

    dlinputs = [(os.path.split(file)[1], os.path.split(file)[0], pbdb, dlbasedir, keepsubdirs, overwrite) for file in allfiles]

    # Make any required local directories
    downloads.makelocaldirs(dlbasedir, dlinputs, keepsubdirs)

    print('Downloading files...')
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.Pool(processes=2)
    pool.map(downloads.dlpbfile, dlinputs)
    print('Finished downloading files')

    return

# Download specific files from a physiobank database
def dldatabasefiles(pbdb, dlbasedir, files, keepsubdirs = True, overwrite = False):
    """Download specified files from a Physiobank database.

    Usage:
    dldatabasefiles(pbdb, dlbasedir, files, keepsubdirs = True, overwrite = False):

    Input arguments:
    - pbdb (required): The Physiobank database directory to download.
      eg. For database 'http://physionet.org/physiobank/database/mitdb', pbdb = 'mitdb'.
    - dlbasedir (required): The full local directory path in which to download the files.
    - files (required): A list of strings specifying the file names to download relative to the database
      base directory
    - keepsubdirs (default=True): Whether to keep the relative subdirectories of downloaded files
      as they are organized in Physiobank (True), or to download all files into the same base directory (False).
    - overwrite (default=False): If set to True, all files will be redownloaded regardless. If set to False,
      existing files with the same name and relative subdirectory will be checked. If the local file is
      the same size as the online file, the download is skipped. If the local file is larger, it will be deleted
      and the file will be redownloaded. If the local file is smaller, the file will be assumed to be
      partially downloaded and the remaining bytes will be downloaded and appended.

    Example Usage:
    import wfdb
    wfdb.dldatabasefiles('ahadb', os.getcwd(), ['STAFF-Studies-bibliography-2016.pdf', 'data/001a.hea', 'data/001a.dat'])
    """

    # Full url physiobank database
    dburl = posixpath.join(downloads.dbindexurl, pbdb)
    # Check if the database is valid
    r = requests.get(dburl)
    r.raise_for_status()

    # Construct the urls to download
    dlinputs = [(os.path.split(file)[1], os.path.split(file)[0], pbdb, dlbasedir, keepsubdirs, overwrite) for file in files]

    # Make any required local directories
    downloads.makelocaldirs(dlbasedir, dlinputs, keepsubdirs)

    print('Downloading files...')
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.Pool(processes=2)
    pool.map(downloads.dlpbfile, dlinputs)
    print('Finished downloading files')

    return
