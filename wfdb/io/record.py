# For wrheader(), all fields must be already filled in and cohesive with one another other. The signals field will not be used.
# For wrsamp(), the field to use will be d_signal (which is allowed to be empty for 0 channel records).
# set_p_features and set_d_features use characteristics of the p_signal or d_signal field to fill in other header fields.
# These are separate from another method 'set_defaults' which the user may call to set default header fields
# The check_field_cohesion() function will be called in wrheader which checks all the header fields.
# The check_sig_cohesion() function will be called in wrsamp in wrdat to check the d_signal against the header fields.

from calendar import monthrange
from collections import OrderedDict
import multiprocessing
import numpy as np
import os
import posixpath
import re
import requests

from . import _header
from . import _signal
from . import download


class BaseRecord(object):
    # The base WFDB class extended by the Record and MultiRecord classes.
    def __init__(self, record_name=None, n_sig=None,
                 fs=None, counter_freq=None, base_counter=None,
                 sig_len=None, base_time=None, base_date=None,
                 comments=None, sig_name=None):
        self.record_name = record_name
        self.n_sig = n_sig
        self.fs = fs
        self.counter_freq = counter_freq
        self.base_counter = base_counter
        self.sig_len = sig_len
        self.base_time = base_time
        self.base_date = base_date
        self.comments = comments
        self.sig_name = sig_name

    # Check whether a single field is valid in its basic form. Does not check compatibility with other fields.
    # ch is only used for signal specification fields, specifying the channels to check. Other channels
    # can be None.
    # Be aware that this function is not just called from wrheader.
    def check_field(self, field, channels=None):
        # Check that the field is present
        if getattr(self, field) is None:
            raise Exception("Missing field required: "+field)

        # Check the type of the field (and of its elements if it should be a list)
        self.check_field_type(field, channels)

        # Expand to make sure all channels must have present field
        if channels == 'all':
            channels = [1]*len(getattr(self, field))

        # Individual specific field checks:
        if field == 'd_signal':
            # Check shape
            if self.d_signal.ndim != 2:
                raise TypeError("d_signal must be a 2d numpy array")
            # Check dtype
            if self.d_signal.dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
                raise TypeError('d_signal must be a 2d numpy array with dtype == int64, int32, int16, or int8.')
        elif field =='p_signal':
            # Check shape
            if self.p_signal.ndim != 2:
                raise TypeError("p_signal must be a 2d numpy array")

        elif field == 'e_d_signal':
            # Check shape
            for ch in range(len(channels)):
                if self.e_d_signal[ch].ndim != 1:
                    raise TypeError("e_d_signal must be a list of 1d numpy arrays")
                # Check dtype
                if self.e_d_signal[ch].dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
                    raise TypeError('e_d_d_signal must be a list of 1d numpy arrays with dtype == int64, int32, int16, or int8.')
        elif field =='e_p_signal':
            # Check shape
            for ch in range(0, len(channels)):
                if self.e_p_signal.ndim != 1:
                    raise TypeError("e_p_signal must be a list of 1d numpy arrays")

        #elif field == 'segments': # Nothing to check here.
        # Record specification fields
        elif field == 'record_name':
            # Allow letters, digits, hyphens, and underscores.
            accepted_string = re.match('[-\w]+', self.record_name)
            if not accepted_string or accepted_string.string != self.record_name:
                raise ValueError('record_name must only comprise of letters, digits, hyphens, and underscores.')
        elif field == 'n_seg':
            if self.n_seg <=0:
                raise ValueError('n_seg must be a positive integer')
        elif field == 'n_sig':
            if self.n_sig <=0:
                raise ValueError('n_sig must be a positive integer')
        elif field == 'fs':
            if self.fs<=0:
                raise ValueError('fs must be a positive number')
        elif field == 'counter_freq':
            if self.counter_freq <=0:
                raise ValueError('counter_freq must be a positive number')
        elif field == 'base_counter':
            if self.base_counter <=0:
                raise ValueError('base_counter must be a positive number')
        elif field == 'sig_len':
            if self.sig_len <0:
                raise ValueError('sig_len must be a non-negative integer')
        elif field == 'base_time':
            _ = parse_timestring(self.base_time)
        elif field == 'base_date':
            _ = parse_datestring(self.base_date)

        # Signal specification fields. Lists of elements to check.
        elif field in _header.sig_field_specs:

            for ch in range(0, len(channels)):
                f = getattr(self, field)[ch]

                # The channel element is allowed to be None
                if not channels[ch]:
                    if f is None:
                        continue

                if field == 'file_name':
                    # Check for file_name characters
                    accepted_string = re.match('[-\w]+\.?[\w]+',f)
                    if not accepted_string or accepted_string.string != f:
                        raise ValueError('File names should only contain alphanumerics, hyphens, and an extension. eg. record_100.dat')
                    # Check that dat files are grouped together
                    if orderedsetlist(self.file_name)[0] != orderednoconseclist(self.file_name):
                        raise ValueError('file_name error: all entries for signals that share a given file must be consecutive')
                elif field == 'fmt':
                    if f not in _signal.dat_fmts:
                        raise ValueError('File formats must be valid WFDB dat formats: '+' , '.join(_signal.dat_fmts))
                elif field == 'samps_per_frame':
                    if f < 1:
                        raise ValueError('samps_per_frame values must be positive integers')
                elif field == 'skew':
                    if f < 0:
                        raise ValueError('skew values must be non-negative integers')
                elif field == 'byte_offset':
                    if f < 0:
                        raise ValueError('byte_offset values must be non-negative integers')
                elif field == 'adc_gain':
                    if f <= 0:
                        raise ValueError('adc_gain values must be positive numbers')
                elif field == 'baseline':
                    # Currently original WFDB library only has 4 bytes for baseline.
                    if f < -2147483648 or f> 2147483648:
                        raise ValueError('baseline values must be between -2147483648 (-2^31) and 2147483647 (2^31 -1)')
                elif field == 'units':
                    if re.search('\s', f):
                        raise ValueError('units strings may not contain whitespaces.')
                elif field == 'adc_res':
                    if f < 0:
                        raise ValueError('adc_res values must be non-negative integers')
                # elif field == 'adc_zero': nothing to check here
                # elif field == 'init_value': nothing to check here
                # elif field == 'checksum': nothing to check here
                elif field == 'block_size':
                    if f < 0:
                        raise ValueError('block_size values must be non-negative integers')
                elif field == 'sig_name':
                    if re.search('\s', f):
                        raise ValueError('sig_name strings may not contain whitespaces.')
                    if len(set(self.sig_name)) != len(self.sig_name):
                        raise ValueError('sig_name strings must be unique.')

        # Segment specification fields
        elif field == 'seg_name':
            # Segment names must be alphanumerics or just a single '~'
            for f in self.seg_name:
                if f == '~':
                    continue
                accepted_string = re.match('[-\w]+',f)
                if not accepted_string or accepted_string.string != f:
                    raise ValueError("Non-null segment names may only contain alphanumerics and dashes. Null segment names must be set to '~'")
        elif field == 'seg_len':
            # For records with more than 1 segment, the first segment may be
            # the layout specification segment with a length of 0
            if len(self.seg_len)>1:
                if self.seg_len[0] < 0:
                    raise ValueError('seg_len values must be positive integers. Only seg_len[0] may be 0 to indicate a layout segment')
                sl = self.seg_len[1:]
            else:
                sl = self.seg_len
            for f in sl:
                if f < 1:
                    raise ValueError('seg_len values must be positive integers. Only seg_len[0] may be 0 to indicate a layout segment')
        # Comment field
        elif field == 'comments':
            for f in self.comments:
                if f=='': # Allow empty string comment lines
                    continue
                if f[0] == '#':
                    print("Note: comment strings do not need to begin with '#'. This library adds them automatically.")
                if re.search('[\t\n\r\f\v]', f):
                    raise ValueError('comments may not contain tabs or newlines (they may contain spaces and underscores).')


    def check_field_type(self, field, ch=None):
        """
        Check the data type of the specified field.
        ch is used for signal specification fields
        Some fields are lists. This must be checked, along with their elements.
        """
        item = getattr(self, field)

        # Record specification field. Nonlist.
        if field in _header.rec_field_specs:
            check_item_type(item, field, _header.rec_field_specs[field].allowed_types)

        # Signal specification field. List.
        elif field in _header.sig_field_specs:
            check_item_type(item, field, _header.sig_field_specs[field].allowed_types, ch)

        # Segment specification field. List. All elements cannot be None
        elif field in _header.seg_field_specs:
            check_item_type(item, field, _header.seg_field_specs[field].allowed_types, 'all')

        # Comments field. List. Elements cannot be None
        elif field == 'comments':
            check_item_type(item, field, (str), 'all')

        # Signals field.
        elif field in ['p_signal','d_signal']:
            check_item_type(item, field, (np.ndarray))

        elif field in ['e_p_signal', 'e_d_signal']:
            check_item_type(item, field, (np.ndarray), 'all')

        # Segments field. List. Elements may be None.
        elif field == 'segments':
            check_item_type(item, field, (Record), 'none')

    # Ensure that input read parameters are valid for the record
    def check_read_inputs(self, sampfrom, sampto, channels, physical, m2s,
                          smooth_frames, return_res):
        # Data Type Check
        if not hasattr(sampfrom, '__index__'):
            raise TypeError('sampfrom must be an integer')
        if not hasattr(sampto, '__index__'):
            raise TypeError('sampto must be an integer')

        if not isinstance(channels, list):
            raise TypeError('channels must be a list of integers')

        # Duration Ranges
        if sampfrom<0:
            raise ValueError('sampfrom must be a non-negative integer')
        if sampfrom>self.sig_len:
            raise ValueError('sampfrom must be shorter than the signal length')
        if sampto<0:
            raise ValueError('sampto must be a non-negative integer')
        if sampto>self.sig_len:
            raise ValueError('sampto must be shorter than the signal length')
        if sampto<=sampfrom:
            raise ValueError('sampto must be greater than sampfrom')

        # Channel Ranges
        for c in channels:
            if c<0:
                raise ValueError('Input channels must all be non-negative integers')
            if c>self.n_sig-1:
                raise ValueError('Input channels must all be lower than the total number of channels')

        if return_res not in [64, 32, 16, 8]:
            raise ValueError("return_res must be one of the following: 64, 32, 16, 8")
        if physical is True and return_res == 8:
            raise ValueError("return_res must be one of the following when physical is True: 64, 32, 16")

        # Cannot expand multiple samples/frame for multi-segment records
        if isinstance(self, MultiRecord):

            # If m2s == True, Physical must be true. There is no
            # meaningful representation of digital signals transferred
            # from individual segments.
            if m2s is True and physical is not True:
                raise Exception('If m2s is True, physical must also be True.')

            if smooth_frames is False:
                raise ValueError('This package version cannot expand all samples when reading multi-segment records. Must enable frame smoothing.')

# Check the item type. Vary the print message regarding whether the item can be None.
# Helper to check_field_type
# channels is a list of booleans indicating whether the field's channel must be present (1) or may be None (0)
# and is not just for signal specification fields
def check_item_type(item, field, allowed_types, channels=None):

    # Checking the list
    if channels is not None:

        # First make sure the item is a list
        if not isinstance(item, list):
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
                if not isinstance(item[ch], allowed_types):
                    raise TypeError("Channel "+str(ch)+" of field: '"+field+"' must be one of the following types:", allowed_types)

            # The field may be None for the channel
            else:
                if not isinstance(item[ch], allowed_types) and item[ch] is not None:
                    raise TypeError("Channel "+str(ch)+" of field: '"+field+"' must be a 'None', or one of the following types:", allowed_types)

    # Single scalar to check
    else:
        if not isinstance(item, allowed_types):
            raise TypeError("Field: '"+field+"' must be one of the following types:", allowed_types)



class Record(BaseRecord, _header.HeaderMixin, _signal.SignalMixin):
    """
    The class representing WFDB headers, and single segment WFDB records.

    Record objects can be created using the initializer, by reading a WFDB
    header with `rdheader`, or a WFDB record (header and associated dat files)
    with `rdrecord`.

    The attributes of the Record object give information about the record as
    specified by: https://www.physionet.org/physiotools/wag/header-5.htm

    In addition, the d_signal and p_signal attributes store the digital and
    physical signals of WFDB records with at least one channel.

    Examples
    --------
    >>> record = wfdb.Record(record_name='r1', fs=250, n_sig=2, sig_len=1000,
                         file_name=['r1.dat','r1.dat'])

    """
    def __init__(self, p_signal=None, d_signal=None,
                 e_p_signal=None, e_d_signal=None,
                 record_name=None, n_sig=None,
                 fs=None, counter_freq=None, base_counter=None,
                 sig_len=None, base_time=None, base_date=None,
                 file_name=None, fmt=None, samps_per_frame=None,
                 skew=None, byte_offset=None, adc_gain=None,
                 baseline=None, units=None, adc_res=None,
                 adc_zero=None, init_value=None, checksum=None,
                 block_size=None, sig_name=None, comments=None):

        # Note the lack of the 'n_seg' field. Single segment records cannot
        # have this field. Even n_seg = 1 makes the header a multi-segment
        # header.

        super(Record, self).__init__(record_name, n_sig,
                    fs, counter_freq, base_counter, sig_len,
                    base_time, base_date, comments, sig_name)

        self.p_signal = p_signal
        self.d_signal = d_signal
        self.e_p_signal = e_p_signal
        self.e_d_signal = e_d_signal

        self.file_name = file_name
        self.fmt = fmt
        self.samps_per_frame = samps_per_frame
        self.skew = skew
        self.byte_offset = byte_offset
        self.adc_gain = adc_gain
        self.baseline = baseline
        self.units = units
        self.adc_res = adc_res
        self.adc_zero = adc_zero
        self.init_value = init_value
        self.checksum = checksum
        self.block_size = block_size

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


    def wrsamp(self, expanded=False, write_dir=''):
        """
        Write a wfdb header file and any associated dat files from this
        object.

        Parameters
        ----------
        expanded : bool, optional
            Whether to write the expanded signal (e_d_signal) instead
            of the uniform signal (d_signal).
        write_dir : str, optional
            The directory in which to write the files.

        """
        # Perform field validity and cohesion checks, and write the
        # header file.
        self.wrheader(write_dir=write_dir)
        if self.n_sig>0:
            # Perform signal validity and cohesion checks, and write the
            # associated dat files.
            self.wr_dats(expanded=expanded, write_dir=write_dir)


    def arrange_fields(self, channels, expanded=False):
        # Arrange/edit object fields to reflect user channel and/or signal range input
        # Account for case when signals are expanded

        # Rearrange signal specification fields
        for field in _header.sig_field_specs:
            item = getattr(self, field)
            setattr(self, field, [item[c] for c in channels])

        # Expanded signals - multiple samples per frame.
        if expanded:
            # Checksum and init_value to be updated if present
            # unless the whole signal length was input
            if self.sig_len != int(len(self.e_d_signal[0])/self.samps_per_frame[0]):
                self.checksum = self.calc_checksum(expanded)
                self.init_value = [s[0] for s in self.e_d_signal]

            self.n_sig = len(channels)
            self.sig_len = int(len(self.e_d_signal[0])/self.samps_per_frame[0])

        # MxN numpy array d_signal
        else:
            # Checksum and init_value to be updated if present
            # unless the whole signal length was input
            if self.sig_len != self.d_signal.shape[0]:

                if self.checksum is not None:
                    self.checksum = self.calc_checksum()
                if self.init_value is not None:
                    ival = list(self.d_signal[0, :])
                    self.init_value = [int(i) for i in ival]

            # Update record specification parameters
            # Important that these get updated after^^
            self.n_sig = len(channels)
            self.sig_len = self.d_signal.shape[0]


class MultiRecord(BaseRecord, _header.MultiHeaderMixin):
    """
    The class representing multi-segment WFDB records.

    MultiRecord objects can be created using the initializer, or by reading a
    multi-segment WFDB record using 'rdrecord' with the `m2s` (multi to single)
    input parameter set to False.

    The attributes of the MultiRecord object give information about the entire
    record as specified by: https://www.physionet.org/physiotools/wag/header-5.htm

    In addition, the `segments` parameter is a list of Record objects
    representing each individual segment, or None representing empty segments,
    of the entire multi-segment record.

    Notably, this class has no attribute representing the signals as a whole.
    The 'multi_to_single' instance method can be called on MultiRecord objects
    to return a single segment representation of the record as a Record object.
    The resulting Record object will have its 'p_signal' field set.

    Examples
    --------
    >>> record_m = wfdb.MultiRecord(record_name='rm', fs=50, n_sig=8,
                                    sig_len=9999, seg_name=['rm_1', '~', rm_2'],
                                    seg_len=[800, 200, 900])
    >>> # Get a MultiRecord object
    >>> record_s = wfdb.rdsamp('s00001-2896-10-10-00-31', m2s=False)
    >>> # Turn it into a
    >>> record_s = record_s.multi_to_single()

    record_s initially stores a `MultiRecord` object, and is then converted into
    a `Record` object.

    """
    def __init__(self, segments=None, layout=None,
                 record_name=None, n_sig=None, fs=None,
                 counter_freq=None, base_counter=None,
                 sig_len=None, base_time=None, base_date=None,
                 seg_name=None, seg_len=None, comments=None,
                 sig_name=None, sig_segments=None):


        super(MultiRecord, self).__init__(record_name, n_sig,
                    fs, counter_freq, base_counter, sig_len,
                    base_time, base_date, comments, sig_name)

        self.layout = layout
        self.segments = segments
        self.seg_name = seg_name
        self.seg_len = seg_len
        self.sig_segments = sig_segments


    def wrsamp(self, write_dir=''):
        """
        Write a multi-segment header, along with headers and dat files
        for all segments, from this object.
        """
        # Perform field validity and cohesion checks, and write the
        # header file.
        self.wrheader(write_dir=write_dir)
        # Perform record validity and cohesion checks, and write the
        # associated segments.
        for seg in self.segments:
            seg.wrsamp(write_dir=write_dir)


    # Check the cohesion of the segments field with other fields used to write the record
    def checksegmentcohesion(self):

        # Check that n_seg is equal to the length of the segments field
        if self.n_seg != len(self.segments):
            raise ValueError("Length of segments must match the 'n_seg' field")

        for i in range(0, n_seg):
            s = self.segments[i]

            # If segment 0 is a layout specification record, check that its file names are all == '~''
            if i==0 and self.seg_len[0] == 0:
                for file_name in s.file_name:
                    if file_name != '~':
                        raise ValueError("Layout specification records must have all file_names named '~'")

            # Check that sampling frequencies all match the one in the master header
            if s.fs != self.fs:
                raise ValueError("The 'fs' in each segment must match the overall record's 'fs'")

            # Check the signal length of the segment against the corresponding seg_len field
            if s.sig_len != self.seg_len[i]:
                raise ValueError('The signal length of segment '+str(i)+' does not match the corresponding segment length')

            totalsig_len = totalsig_len + getattr(s, 'sig_len')

        # No need to check the sum of sig_lens from each segment object against sig_len
        # Already effectively done it when checking sum(seg_len) against sig_len


    # Determine the segments and the samples
    # within each segment that have to be read in a
    # multi-segment record. Called during rdsamp.
    def required_segments(self, sampfrom, sampto, channels):

        # The starting segment with actual samples
        if self.layout == 'Fixed':
            startseg = 0
        else:
            startseg = 1

        # Cumulative sum of segment lengths (ignoring layout segment)
        cumsumlengths = list(np.cumsum(self.seg_len[startseg:]))
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
            segstartsamp = sum(self.seg_len[0:readsegs[0]])
            readsamps = [[sampfrom-segstartsamp, sampto-segstartsamp]]

        else:
            # More than one segment to read
            readsegs = list(range(readsegs[0], readsegs[1]+1))
            readsamps = [[0, self.seg_len[s]] for s in readsegs]

            # Starting sample for first segment.
            readsamps[0][0] = sampfrom - ([0] + cumsumlengths)[readsegs[0]-startseg]

            # End sample for last segment
            readsamps[-1][1] = sampto - ([0] + cumsumlengths)[readsegs[-1]-startseg]

        return (readsegs, readsamps)

    # Get the channel numbers to be read from each segment
    def required_signal(self, readsegs, channels, dirname, pb_dir):

        # Fixed layout. All channels are the same.
        if self.layout == 'Fixed':
            # Should we bother here with skipping empty segments?
            # They won't be read anyway.
            readsigs = [channels]*len(readsegs)
        # Variable layout: figure out channels by matching record names
        else:
            readsigs = []
            # The overall layout signal names
            l_sig_names = self.segments[0].sig_name
            # The wanted signals
            w_sig_names = [l_sig_names[c] for c in channels]

            # For each segment ...
            for i in range(0, len(readsegs)):
                # Skip empty segments
                if self.seg_name[readsegs[i]] == '~':
                    readsigs.append(None)
                else:
                    # Get the signal names of the current segment
                    s_sig_names = rdheader(os.path.join(dirname, self.seg_name[readsegs[i]]), pb_dir = pb_dir).sig_name
                    readsigs.append(wanted_siginds(w_sig_names, s_sig_names))

        return readsigs

    # Arrange/edit object fields to reflect user channel and/or signal range input
    def arrange_fields(self, readsegs, segranges, channels):

        # Update seg_len values for relevant segments
        for i in range(0, len(readsegs)):
            self.seg_len[readsegs[i]] = segranges[i][1] - segranges[i][0]

        # Update record specification parameters
        self.n_sig = len(channels)
        self.sig_len = sum([sr[1]-sr[0] for sr in segranges])

        # Get rid of the segments and segment line parameters
        # outside the desired segment range
        if self.layout == 'Fixed':
            self.segments = self.segments[readsegs[0]:readsegs[-1]+1]
            self.seg_name = self.seg_name[readsegs[0]:readsegs[-1]+1]
            self.seg_len = self.seg_len[readsegs[0]:readsegs[-1]+1]
        else:
            # Keep the layout specifier segment
            self.segments = [self.segments[0]] + self.segments[readsegs[0]:readsegs[-1]+1]
            self.seg_name = [self.seg_name[0]] + self.seg_name[readsegs[0]:readsegs[-1]+1]
            self.seg_len = [self.seg_len[0]] + self.seg_len[readsegs[0]:readsegs[-1]+1]

        # Update number of segments
        self.n_seg = len(self.segments)


    def multi_to_single(self, return_res=64):
        """
        Create a Record object from the MultiRecord object. All signal segments
        will be combined into the new object's `p_signal` field.

        Parameters
        ----------
        return_res : int
            The return resolution of the `p_signal` field. Options are 64, 32,
            and 16.

        """

        # The fields to transfer to the new object
        fields = self.__dict__.copy()

        # Remove multirecord fields
        del(fields['segments'])
        del(fields['seg_name'])
        del(fields['seg_len'])
        del(fields['n_seg'])

        # The output physical signals
        if return_res == 64:
            floatdtype = 'float64'
        elif return_res == 32:
            floatdtype = 'float32'
        else:
            floatdtype = 'float16'


        p_signal = np.zeros([self.sig_len, self.n_sig], dtype=floatdtype)

        # Get the physical samples from each segment

        # Start and end samples in the overall array
        # to place the segment samples into
        startsamps = [0] + list(np.cumsum(self.seg_len)[0:-1])
        endsamps = list(np.cumsum(self.seg_len))

        if self.layout == 'Fixed':
            # Get the signal names and units from the first segment
            fields['sig_name'] = self.segments[0].sig_name
            fields['units'] = self.segments[0].units

            for i in range(self.n_seg):
                p_signal[startsamps[i]:endsamps[i],:] = self.segments[i].p_signal
        # For variable layout, have to get channels by name
        else:
            # Get the signal names from the layout segment
            fields['sig_name'] = self.segments[0].sig_name
            fields['units'] = self.segments[0].units

            for i in range(1, self.n_seg):
                seg = self.segments[i]

                # Empty segment
                if seg is None:
                    p_signal[startsamps[i]:endsamps[i],:] = np.nan
                # Non-empty segment
                else:
                    # Figure out if there are any channels wanted and
                    # the output channels they are to be stored in
                    inchannels = []
                    outchannels = []
                    for s in fields['sig_name']:
                        if s in seg.sig_name:
                            inchannels.append(seg.sig_name.index(s))
                            outchannels.append(fields['sig_name'].index(s))

                    # Segment contains no wanted channels. Fill with nans.
                    if inchannels == []:
                        p_signal[startsamps[i]:endsamps[i],:] = np.nan
                    # Segment contains wanted channel(s). Transfer samples.
                    else:
                        # This statement is necessary in case this function is not called
                        # directly from rdsamp with m2s=True.
                        if not hasattr(seg, 'p_signal'):
                            seg.p_signal = seg.dac(return_res=return_res)
                        for ch in range(0, fields['n_sig']):
                            if ch not in outchannels:
                                p_signal[startsamps[i]:endsamps[i],ch] = np.nan
                            else:
                                p_signal[startsamps[i]:endsamps[i],ch] = seg.p_signal[:, inchannels[outchannels.index(ch)]]

        # Create the single segment Record object and set attributes
        record = Record()
        for field in fields:
            setattr(record, field, fields[field])
        record.p_signal = p_signal

        return record


#------------------- Reading Records -------------------#

def rdheader(record_name, pb_dir=None, rd_segments=False):
    """
    Read a WFDB header file and return the record descriptors as attributes
    in a Record object.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read (without any file extensions).
        If the argument contains any path delimiter characters, the argument
        will be interpreted as PATH/baserecord and the header file will be
        searched for in the local path.
    pb_dir : str, optional
        Option used to stream data from Physiobank. The Physiobank database
        directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb'
        pb_dir='mitdb'.

    rd_segments : bool, optional
        Used when reading multi-segment headers. If True, segment headers will
        also be read (into the record object's `segments` field).

    Returns
    -------
    record : Record or MultiRecord
        The wfdb Record or MultiRecord object representing the contents of the
        header read.

    Examples
    --------
    >>> ecg_record = wfdb.rdheader('sample-data/test01_00s', sampfrom=800,
                                   channels = [1,3])

    """

    # Read the header file. Separate comment and non-comment lines
    header_lines, comment_lines = _header.get_header_lines(record_name, pb_dir)

    # Get fields from record line
    d_rec = _header.read_rec_line(header_lines[0])

    # Processing according to whether the header is single or multi segment

    # Single segment header - Process signal specification lines
    if d_rec['n_seg'] is None:
        # Create a single-segment WFDB record object
        record = Record()

        # There is at least one channel
        if len(header_lines)>1:
            # Read the fields from the signal lines
            d_sig = _header.read_sig_lines(header_lines[1:])
            # Set the object's signal line fields
            for field in _header.sig_field_specs:
                setattr(record, field, d_sig[field])

        # Set the object's record line fields
        for field in _header.rec_field_specs:
            if field == 'n_seg':
                continue
            setattr(record, field, d_rec[field])
    # Multi segment header - Process segment specification lines
    else:
        # Create a multi-segment WFDB record object
        record = MultiRecord()
        # Read the fields from the segment lines
        d_seg = _header.read_seg_lines(header_lines[1:])
        # Set the object's segment line fields
        for field in _header.seg_field_specs:
            setattr(record, field, d_seg[field])
        # Set the objects' record line fields
        for field in _header.rec_field_specs:
            setattr(record, field, d_rec[field])
        # Determine whether the record is fixed or variable
        if record.seg_len[0] == 0:
            record.layout = 'Variable'
        else:
            record.layout = 'Fixed'

        # If specified, read the segment headers
        if rd_segments:
            record.segments = []
            # Get the base record name (could be empty)
            dirname = os.path.split(record_name)[0]
            for s in record.seg_name:
                if s == '~':
                    record.segments.append(None)
                else:
                    record.segments.append(rdheader(os.path.join(dirname,s), pb_dir))
            # Fill in the sig_name attribute
            record.sig_name = record.get_sig_name()
            # Fill in the sig_segments attribute
            record.sig_segments = record.get_sig_segments()

    # Set the comments field
    record.comments = []
    for line in comment_lines:
        record.comments.append(line.strip(' \t#'))

    return record


def rdrecord(record_name, sampfrom=0, sampto='end', channels='all',
             physical=True, pb_dir=None, m2s=True, smooth_frames=True,
             ignore_skew=False, return_res=64):
    """
    Read a WFDB record and return the signal and record descriptors as
    attributes in a Record or MultiRecord object.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read (without any file extensions).
        If the argument contains any path delimiter characters, the argument
        will be interpreted as PATH/baserecord and the data files will be
        searched for in the local path.
    sampfrom : int, optional
        The starting sample number to read for each channel.
    sampto : int, or 'end', optional
        The sample number at which to stop reading for each channel. Leave as
        'end' to read the entire duration.
    channels : list, or 'all', optional
        List of integer indices specifying the channels to be read. Leave as
        'all' to read all channels.
    physical : bool, optional
        Specifies whether to return signals in physical units in the p_signal
        field (True), or digital units in the d_signal field (False).
    pb_dir : str, optional
        Option used to stream data from Physiobank. The Physiobank database
        directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb'
        pb_dir='mitdb'.
    m2s : bool, optional
        Used when reading multi-segment records. Specifies whether to directly
        return a wfdb MultiRecord object (False), or to convert it into and
        return a wfdb Record object (True).
    smooth_frames : bool, optional
        Used when reading records with signals having multiple samples per
        frame. Specifies whether to smooth the samples in signals with more than
        one sample per frame and return an (MxN) uniform numpy array as the
        `d_signal` or `p_signal` field (True), or to return a list of 1d numpy
        arrays containing every expanded sample as the `e_d_signal` or
        `e_p_signal` field (False).
    ignore_skew : bool, optional
        Used when reading records with at least one skewed signal. Specifies
        whether to apply the skew to align the signals in the output variable
        (False), or to ignore the skew field and load in all values contained in
        the dat files unaligned (True).
    return_res : int, optional
        The numpy array dtype of the returned signals. Options are: 64, 32,
        16, and 8, where the value represents the numpy int or float dtype.
        Note that the value cannot be 8 when physical is True since there is no
        float8 format.

    Returns
    -------
    record : Record or MultiRecord
        The wfdb Record or MultiRecord object representing the contents of the
        record read.

    Notes
    -----
    If a signal range or channel selection is specified when calling this
    function, the resulting attributes of the returned object will be set to
    reflect the section of the record that is actually read, rather than
    necessarily the entire record. For example, if channels=[0, 1, 2] is
    specified when reading a 12 channel record, the 'n_sig' attribute will be 3,
    not 12.

    The `rdsamp` function exists as a simple alternative to `rdrecord` for
    the common purpose of extracting the physical signals and a few important
    descriptor fields. `rdsamp` returns two arguments:

    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/test01_00s', sampfrom=800,
                               channels = [1,3])

    """

    dirname, base_record_name = os.path.split(record_name)

    # Read the header fields
    record = rdheader(record_name, pb_dir=pb_dir, rd_segments=False)

    # Set defaults for sampto and channels input variables
    if sampto == 'end':
        sampto = record.sig_len
    if channels == 'all':
        channels = list(range(record.n_sig))

    # Ensure that input fields are valid for the record
    record.check_read_inputs(sampfrom, sampto, channels, physical, m2s, smooth_frames, return_res)

    # A single segment record
    if isinstance(record, Record):

        # Only 1 sample/frame, or frames are smoothed. Return uniform numpy array
        if smooth_frames or max([record.samps_per_frame[c] for c in channels])==1:
            # Read signals from the associated dat files that contain wanted channels
            record.d_signal = _signal.rd_segment(record.file_name, dirname, pb_dir, record.n_sig, record.fmt, record.sig_len,
                                                  record.byte_offset, record.samps_per_frame, record.skew,
                                                  sampfrom, sampto, channels, smooth_frames, ignore_skew)

            # Arrange/edit the object fields to reflect user channel and/or signal range input
            record.arrange_fields(channels, expanded=False)

            if physical:
                # Perform inplace dac to get physical signal
                record.dac(expanded=False, return_res=return_res, inplace=True)

        # Return each sample of the signals with multiple samples per frame
        else:
            record.e_d_signal = _signal.rd_segment(record.file_name, dirname, pb_dir, record.n_sig, record.fmt, record.sig_len,
                                                    record.byte_offset, record.samps_per_frame, record.skew,
                                                    sampfrom, sampto, channels, smooth_frames, ignore_skew)

            # Arrange/edit the object fields to reflect user channel and/or signal range input
            record.arrange_fields(channels, expanded=True)

            if physical is True:
                # Perform dac to get physical signal
                record.dac(expanded=True, return_res=return_res, inplace=True)

    # A multi segment record
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

        record.segments = [None]*record.n_seg

        # Variable layout
        if record.seg_len[0] == 0:
            record.layout = 'Variable'
            # Read the layout specification header
            record.segments[0] = rdheader(os.path.join(dirname, record.seg_name[0]), pb_dir=pb_dir)
        # Fixed layout
        else:
            record.layout = 'Fixed'

        # The segment numbers and samples within each segment to read.
        readsegs, segranges  = record.required_segments(sampfrom, sampto, channels)
        # The signals within each segment to read
        segsigs = record.required_signal(readsegs, channels, dirname, pb_dir)

        # Read the desired samples in the relevant segments
        for i in range(len(readsegs)):
            segnum = readsegs[i]
            # Empty segment or segment with no relevant channels
            if record.seg_name[segnum] == '~' or segsigs[i] is None:
                record.segments[segnum] = None
            else:
                record.segments[segnum] = rdrecord(
                    os.path.join(dirname, record.seg_name[segnum]),
                    sampfrom=segranges[i][0], sampto=segranges[i][1],
                    channels=segsigs[i], physical=True, pb_dir=pb_dir)

        # Arrange the fields of the overall object to reflect user input
        record.arrange_fields(readsegs, segranges, channels)

        # Convert object into a single segment Record object
        if m2s:
            record = record.multi_to_single(return_res=return_res)

    # Perform dtype conversion if necessary
    if isinstance(record, Record) and record.n_sig>0:
        record.convert_dtype(physical, return_res, smooth_frames)

    return record


def rdsamp(record_name, sampfrom=0, sampto='end', channels='all', pb_dir=None):
    """
    Read a WFDB record, and return the physical signals and a few important
    descriptor fields.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read (without any file extensions).
        If the argument contains any path delimiter characters, the argument
        will be interpreted as PATH/baserecord and the data files will be
        searched for in the local path.
    sampfrom : int, optional
        The starting sample number to read for each channel.
    sampto : int, or 'end', optional
        The sample number at which to stop reading for each channel. Leave as
        'end' to read the entire duration.
    channels : list, or 'all', optional
        List of integer indices specifying the channels to be read. Leave as
        'all' to read all channels.
    pb_dir : str, optional
        Option used to stream data from Physiobank. The Physiobank database
        directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb'
        pb_dir='mitdb'.

    Returns
    -------
    signals : numpy array
        A 2d numpy array storing the physical signals from the record.
    fields : dict
        A dictionary containing several key attributes of the read record:
          - fs: The sampling frequency of the record
          - units: The units for each channel
          - sig_name: The signal name for each channel
          - comments: Any comments written in the header

    Notes
    -----
    If a signal range or channel selection is specified when calling this
    function, the resulting attributes of the returned object will be set to
    reflect the section of the record that is actually read, rather than
    necessarily the entire record. For example, if channels=[0, 1, 2] is
    specified when reading a 12 channel record, the 'n_sig' attribute will be 3,
    not 12.

    The `rdrecord` function is the base function upon which this one is built.
    It returns all attributes present, along with the signals, as attributes in
    a `Record` object. The function, along with the returned data type, has more
    options than `rdsamp` for users who wish to more directly manipulate WFDB
    content.

    Examples
    --------
    >>> signals, fields = wfdb.rdsamp('sample-data/test01_00s', sampfrom=800,
                                      channel =[1,3])

    """

    record = rdrecord(record_name, sampfrom, sampto, channels, True, pb_dir, True)

    signals = record.p_signal
    fields = {}
    for field in ['fs','units','sig_name', 'comments']:
        fields[field] = getattr(record, field)

    return signals, fields


def wanted_siginds(wanted_sig_names, record_sig_names):
    """
    Given some wanted signal names, and the signal names contained
    in a record, return the indices of the record channels that intersect.
    Remember that the wanted signal names are already in order specified in user input channels.
    """
    contained_signal = [s for s in wanted_sig_names if s in record_sig_names]
    if contained_signal == []:
        return None
    else:
        return [record_sig_names.index(s) for s in contained_signal]




#------------------- /Reading Records -------------------#


def wrsamp(record_name, fs, units, sig_name, p_signal=None, d_signal=None,
    fmt=None, adc_gain=None, baseline=None, comments=None, base_time=None,
    base_date=None, write_dir=''):
    """
    Write a single segment WFDB record, creating a WFDB header file and any
    associated dat files.

    Parameters
    ----------
    record_name : str
        The string name of the WFDB record to be written (without any file
        extensions).
    fs : int, or float
        The sampling frequency of the record.
    units : list
        A list of strings giving the units of each signal channel.
    sig_name :
        A list of strings giving the signal name of each signal channel.
    p_signal : numpy array, optional
        An (MxN) 2d numpy array, where M is the signal length. Gives the
        physical signal values intended to be written. Either p_signal or
        d_signal must be set, but not both. If p_signal is set, this method will
        use it to perform analogue-digital conversion, writing the resultant
        digital values to the dat file(s). If fmt is set, gain and baseline must
        be set or unset together. If fmt is unset, gain and baseline must both
        be unset.
    d_signal : numpy array, optional
        An (MxN) 2d numpy array, where M is the signal length. Gives the
        digital signal values intended to be directly written to the dat
        file(s). The dtype must be an integer type. Either p_signal or d_signal
        must be set, but not both. In addition, if d_signal is set, fmt, gain
        and baseline must also all be set.
    fmt : list, optional
        A list of strings giving the WFDB format of each file used to store each
        channel. Accepted formats are: '80','212",'16','24', and '32'. There are
        other WFDB formats as specified by:
        https://www.physionet.org/physiotools/wag/signal-5.htm
        but this library will not write (though it will read) those file types.
    adc_gain : list, optional
        A list of numbers specifying the ADC gain.
    baseline : list, optional
        A list of integers specifying the digital baseline.
    comments : list, optional
        A list of string comments to be written to the header file.
    base_time : str, optional
        A string of the record's start time in 24h 'HH:MM:SS(.ms)' format.
    base_date : str, optional
        A string of the record's start date in 'DD/MM/YYYY' format.
    write_dir : str, optional
        The directory in which to write the files.

    Notes
    -----
    This is a gateway function, written as a simple method to write WFDB record
    files using the most common parameters. Therefore not all WFDB fields can be
    set via this function.

    For more control over attributes, create a `Record` object, manually set its
    attributes, and call its `wrsamp` instance method. If you choose this more
    advanced method, see also the `set_defaults`, `set_d_features`, and
    `set_p_features` instance methods to help populate attributes.

    Examples
    --------
    >>> # Read part of a record from Physiobank
    >>> signals, fields = wfdb.rdsamp('a103l', sampfrom=50000, channels=[0,1],
                                   pb_dir='challenge/2015/training')
    >>> # Write a local WFDB record (manually inserting fields)
    >>> wfdb.wrsamp('ecgrecord', fs = 250, units=['mV', 'mV'],
                    sig_name=['I', 'II'], p_signal=signals, fmt=['16', '16'])

    """

    # Check input field combinations
    if p_signal is not None and d_signal is not None:
        raise Exception('Must only give one of the inputs: p_signal or d_signal')
    if d_signal is not None:
        if fmt is None or adc_gain is None or baseline is None:
            raise Exception("When using d_signal, must also specify 'fmt', 'gain', and 'baseline' fields.")
    # Depending on whether d_signal or p_signal was used, set other required features.
    if p_signal is not None:
        # Create the Record object
        record = Record(record_name=record_name, p_signal=p_signal, fs=fs,
                        fmt=fmt, units=units, sig_name=sig_name,
                        adc_gain=adc_gain, baseline=baseline,
                        comments=comments, base_time=base_time,
                        base_date=base_date)
        # Compute optimal fields to store the digital signal, carry out adc,
        # and set the fields.
        record.set_d_features(do_adc=1)
    else:
        # Create the Record object
        record = Record(record_name=record_name, d_signal=d_signal, fs=fs,
                        fmt=fmt, units=units, sig_name=sig_name,
                        adc_gain=adc_gain, baseline=baseline,
                        comments=comments, base_time=base_time,
                        base_date=base_date)
        # Use d_signal to set the fields directly
        record.set_d_features()

    # Set default values of any missing field dependencies
    record.set_defaults()
    # Write the record files - header and associated dat
    record.wrsamp(write_dir=write_dir)


# Time string parser for WFDB header - H(H):M(M):S(S(.sss)) format.
def parse_timestring(timestring):
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
def parse_datestring(datestring):
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


def dl_database(db_dir, dl_dir, records='all', annotators='all', keep_subdirs=True,
                overwrite = False):
    """
    Download WFDB record (and optionally annotation) files from a
    Physiobank database. The database must contain a 'RECORDS' file in
    its base directory which lists its WFDB records.

    Parameters
    ----------
    db_dir : str
        The Physiobank database directory to download. eg. For database:
        'http://physionet.org/physiobank/database/mitdb', db_dir='mitdb'.
    dl_dir : str
        The full local directory path in which to download the files.
    records : list, or 'all', optional
        A list of strings specifying the WFDB records to download. Leave
        as 'all' to download all records listed in the database's
        RECORDS file.
        eg. records=['test01_00s', test02_45s] for database:
        https://physionet.org/physiobank/database/macecgdb/
    annotators : list, 'all', or None, optional
        A list of strings specifying the WFDB annotation file types to
        download along with the record files. Is either None to skip
        downloading any annotations, 'all' to download all annotation
        types as specified by the ANNOTATORS file, or a list of strings
        which each specify an annotation extension.
        eg. annotators = ['anI'] for database:
        https://physionet.org/physiobank/database/prcp/
    keep_subdirs : bool, optional
        Whether to keep the relative subdirectories of downloaded files
        as they are organized in Physiobank (True), or to download all
        files into the same base directory (False).
    overwrite : bool, optional
        If True, all files will be redownloaded regardless. If False,
        existing files with the same name and relative subdirectory will
        be checked. If the local file is the same size as the online
        file, the download is skipped. If the local file is larger, it
        will be deleted and the file will be redownloaded. If the local
        file is smaller, the file will be assumed to be partially
        downloaded and the remaining bytes will be downloaded and
        appended.

    Examples
    --------
    >>> wfdb.dl_database('ahadb', os.getcwd())

    """
    # Check if the database is valid
    r = requests.get(dburl)
    r.raise_for_status()

    # Get the list of records
    recordlist = download.get_record_list(db_dir, records)
    # Get the annotator extensions
    annotators = download.get_annotators(db_dir, annotators)

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
            record = rdheader(baserecname, pb_dir=posixpath.join(db_dir, dirname))

            # Single segment record
            if isinstance(record, Record):
                # Add all dat files of the segment
                for file in record.file_name:
                    allfiles.append(posixpath.join(dirname, file))

            # Multi segment record
            else:
                for seg in record.seg_name:
                    # Skip empty segments
                    if seg == '~':
                        continue
                    # Add the header
                    allfiles.append(posixpath.join(dirname, seg+'.hea'))
                    # Layout specifier has no dat files
                    if seg.endswith('_layout'):
                        continue
                    # Add all dat files of the segment
                    recseg = rdheader(seg, pb_dir=posixpath.join(db_dir, dirname))
                    for file in recseg.file_name:
                        allfiles.append(posixpath.join(dirname, file))
        # check whether the record has any requested annotation files
        if annotators is not None:
            for a in annotators:
                annfile = rec+'.'+a
                url = posixpath.join(download.db_index_url, db_dir, annfile)
                rh = requests.head(url)

                if rh.status_code != 404:
                    allfiles.append(annfile)

    dlinputs = [(os.path.split(file)[1], os.path.split(file)[0], db_dir, dl_dir, keep_subdirs, overwrite) for file in allfiles]

    # Make any required local directories
    download.make_local_dirs(dl_dir, dlinputs, keep_subdirs)

    print('Downloading files...')
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.Pool(processes=2)
    pool.map(download.dl_pb_file, dlinputs)
    print('Finished downloading files')

    return


# ---------- For storing WFDB Signal definitions ---------- #


# Unit scales used for default display scales.
unit_scale = {
    'Voltage': ['pV', 'nV', 'uV', 'mV', 'V', 'kV'],
    'Temperature': ['C'],
    'Pressure': ['mmHg'],
}



# Signal class with all its parameters
class SignalClass(object):
    def __init__(self, abbreviation, description, signalnames):
        self.abbreviation = abbreviation
        self.description = description
        # names that are assigned to this signal type
        self.signalnames = signalnames

    def __str__(self):
        return self.abbreviation

# All signal types. Make sure signal names are in lower case.
sig_classes = [
    SignalClass('BP', 'Blood Pressure', ['bp','abp','pap','cvp',]),
    SignalClass('CO2', 'Carbon Dioxide', ['co2']),
    SignalClass('CO', 'Carbon Monoxide', ['co']),
    SignalClass('ECG', 'Electrocardiogram', ['i','ii','iii','iv','v','avr']),
    SignalClass('EEG', 'Electroencephalogram',['eeg']),
    SignalClass('EMG', 'Electromyograph', ['emg']),
    SignalClass('EOG', 'Electrooculograph', ['eog']),
    SignalClass('HR', 'Heart Rate', ['hr']),
    SignalClass('MMG', 'Magnetomyograph', ['mmg']),
    SignalClass('O2', 'Oxygen', ['o2','sp02']),
    SignalClass('PLETH', 'Plethysmograph', ['pleth']),
    SignalClass('RESP', 'Respiration', ['resp']),
    SignalClass('SCG', 'Seismocardiogram', ['scg']),
    SignalClass('STAT', 'Status', ['stat','status']), # small integers indicating status
    SignalClass('ST', 'ECG ST Segment', ['st']),
    SignalClass('TEMP', 'Temperature', ['temp']),
    SignalClass('UNKNOWN', 'Unknown Class', []),
]
