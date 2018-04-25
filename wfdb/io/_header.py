import datetime
import os
import re

import numpy as np
import pandas as pd

from . import download
from . import _signal


int_types = (int, np.int64, np.int32, np.int16, np.int8)
float_types = int_types + (float, np.float64, np.float32)

"""
WFDB field specifications for each field.

Parameters
----------
allowed_types:
    Data types the field (or its elements) can be.
delimiter:
    The text delimiter that precedes the field in the header file.
write_required:
    Whether the field is required for writing a header (more stringent
    than origin WFDB library).
read_default:
    The default value for the field when read if any. Most fields do not
    have a default. The reason for the variation, is that we do not want
    to imply that some fields are present when they are not, unless the
    field is essential. See the notes.
write_default:
    The default value for the field to fill in before writing, if any.

Notes
-----
In the original WFDB package, certain fields have default values, but
not all of them. Some attributes need to be present for core
functionality, ie. baseline, whereas others are not essential, yet have
defaults, ie. base_time.

This inconsistency has likely resulted in the generation of incorrect
files, and general confusion. This library aims to make explicit,
whether certain fields are present in the file, by setting their values
to None if they are not written in, unless the fields are essential, in
which case an actual default value will be set.

The read vs write default values are different for 2 reasons:
1. We want to force the user to be explicit with certain important
   fields when writing WFDB records fields, without affecting
   existing WFDB headers when reading.
2. Certain unimportant fields may be dependencies of other
   important fields. When writing, we want to fill in defaults
   so that the user doesn't need to. But when reading, it should
   be clear that the fields are missing.

"""

_SPECIFICATION_COLUMNS = ['allowed_types', 'delimiter', 'dependency',
                         'write_required', 'read_default', 'write_default']

RECORD_SPECS = pd.DataFrame(
    index=['record_name', 'n_seg', 'n_sig', 'fs', 'counter_freq',
           'base_counter', 'sig_len', 'base_time', 'base_date'],
    columns=_SPECIFICATION_COLUMNS,
    data=[[(str), '', None, True, None, None], # record_name
          [int_types, '/', 'record_name', True, None, None], # n_seg
          [int_types, ' ', 'record_name', True, None, None], # n_sig
          [float_types, ' ', 'n_sig', True, 250, None], # fs
          [float_types, '/', 'fs', False, None, None], # counter_freq
          [float_types, '(', 'counter_freq', False, None, None], # base_counter
          [int_types, ' ', 'fs', True, None, None], # sig_len
          [(datetime.time), ' ', 'sig_len', False, None, '00:00:00'], # base_time
          [(datetime.date), ' ', 'base_time', False, None, None], # base_date
    ]
)

SIGNAL_SPECS = pd.DataFrame(
    index=['file_name', 'fmt', 'samps_per_frame', 'skew', 'byte_offset',
           'adc_gain', 'baseline', 'units', 'adc_res', 'adc_zero',
           'init_value', 'checksum', 'block_size', 'sig_name'],
    columns=_SPECIFICATION_COLUMNS,
    data=[[(str), '', None, True, None, None], # file_name
          [(str), ' ', 'file_name', True, None, None], # fmt
          [int_types, 'x', 'fmt', False, 1, None], # samps_per_frame
          [int_types, ':', 'fmt', False, None, None], # skew
          [int_types, '+', 'fmt', False, None, None], # byte_offset
          [float_types, ' ', 'fmt', True, 200., None], # adc_gain
          [int_types, '(', 'adc_gain', True, 0, None], # baseline
          [(str), '/', 'adc_gain', True, 'mV', None], # units
          [int_types, ' ', 'adc_gain', False, None, 0], # adc_res
          [int_types, ' ', 'adc_res', False, None, 0], # adc_zero
          [int_types, ' ', 'adc_zero', False, None, None], # init_value
          [int_types, ' ', 'init_value', False, None, None], # checksum
          [int_types, ' ', 'checksum', False, None, 0], # block_size
          [(str), ' ', 'block_size', False, None, None], # sig_name
    ]
)

SEGMENT_SPECS = pd.DataFrame(
    index=['seg_name', 'seg_len'],
    columns=_SPECIFICATION_COLUMNS,
    data=[[(str), '', None, True, None, None], # seg_name
          [int_types, ' ', 'seg_name', True, None, None], # seg_len
    ]
)

FIELD_SPECS = pd.concat((RECORD_SPECS, SIGNAL_SPECS, SEGMENT_SPECS))


# Regexp objects for reading headers

# Record Line Fields
_rx_record = re.compile(
    ''.join(
        [
            "(?P<record_name>[-\w]+)/?(?P<n_seg>\d*)[ \t]+",
            "(?P<n_sig>\d+)[ \t]*",
            "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<base_counter>\d*\.?\d*)\)?[ \t]*",
            "(?P<sig_len>\d*)[ \t]*",
            "(?P<base_time>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
            "(?P<base_date>\d{,2}/?\d{,2}/?\d{,4})"]))

# Signal Line Fields
_rx_signal = re.compile(
    ''.join(
        [
            "(?P<file_name>[-\w]+\.?[\w]*~?)[ \t]+(?P<fmt>\d+)x?"
            "(?P<samps_per_frame>\d*):?(?P<skew>\d*)\+?(?P<byte_offset>\d*)[ \t]*",
            "(?P<adc_gain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^\-\?%]*)[ \t]*",
            "(?P<adc_res>\d*)[ \t]*(?P<adc_zero>-?\d*)[ \t]*(?P<init_value>-?\d*)[ \t]*",
            "(?P<checksum>-?\d*)[ \t]*(?P<block_size>\d*)[ \t]*(?P<sig_name>[\S]?[^\t\n\r\f\v]*)"]))

# Segment Line Fields
_rx_segment = re.compile('(?P<seg_name>\w*~?)[ \t]+(?P<seg_len>\d+)')


class BaseHeaderMixin(object):
    """
    Mixin class with multi-segment header methods. Inherited by Record and
    MultiRecord classes
    """

    def get_write_subset(self, spec_type):
        """
        Get the fields used to write the header,


        Helper function for `get_write_fields`.

        Parameters
        ----------
        spec_type : str
            The set of specification fields desired. Either 'record' or
            'signal'.

        - For record fields,  returns a list of all fields needed.
        - For signal fields, it returns a dictionary of all fields needed,
        with keys = field and value = list of 1 or 0 indicating channel for the field

        """
        if spec_type == 'record':
            write_fields = []
            fieldspecs = OrderedDict(reversed(list(rec_field_specs.items())))
            # Remove this requirement for single segs
            if not hasattr(self, 'n_seg'):
                del(fieldspecs['n_seg'])

            for f in fieldspecs:
                if f in write_fields:
                    continue
                # If the field is required by default or has been defined by the user
                if fieldspecs[f].write_req or getattr(self, f) is not None:
                    rf=f
                    # Add the field and its recursive dependencies
                    while rf is not None:
                        write_fields.append(rf)
                        rf=fieldspecs[rf].dependency
            # Add comments if any
            if getattr(self, 'comments') is not None:
                write_fields.append('comments')

        # signal spec field. Need to return a potentially different list for each channel.
        elif spec_type == 'signal':
            # List of lists for each channel
            write_fields = []

            allwrite_fields = []
            fieldspecs = OrderedDict(reversed(list(SIGNAL_FIELDS.items())))

            for ch in range(self.n_sig):
                # The fields needed for this channel
                write_fieldsch = []
                for f in fieldspecs:
                    if f in write_fieldsch:
                        continue

                    fielditem = getattr(self, f)
                    # If the field is required by default or has been defined by the user
                    if fieldspecs[f].write_req or (fielditem is not None and fielditem[ch] is not None):
                        rf=f
                        # Add the field and its recursive dependencies
                        while rf is not None:
                            write_fieldsch.append(rf)
                            rf = fieldspecs[rf].dependency

                write_fields.append(write_fieldsch)

            # Convert the list of lists to a single dictionary.
            # keys = field and value = list of 1 or 0 indicating channel for the field
            dictwrite_fields = {}

            # For fields present in any channel:
            for f in set([i for wsub in write_fields for i in wsub]):
                dictwrite_fields[f] = [0]*self.n_sig

                for ch in range(self.n_sig):
                    if f in write_fields[ch]:
                        dictwrite_fields[f][ch] = 1

            write_fields = dictwrite_fields

        return write_fields


class HeaderMixin(BaseHeaderMixin):
    """
    Mixin class with single-segment header methods. Inherited by Record class.
    """

    def set_defaults(self):
        """
        Set defaults for fields needed to write the header if they have defaults.
        This is NOT called by rdheader. It is only automatically called by the gateway wrsamp for convenience.
        It is also not called by wrhea (this may be changed in the future) since
        it is supposed to be an explicit function.

        Not responsible for initializing the
        attributes. That is done by the constructor.
        """
        rfields, sfields = self.get_write_fields()
        for f in rfields:
            self.set_default(f)
        for f in sfields:
            self.set_default(f)

    # Write a wfdb header file. The signals or segments fields are not used.
    def wrheader(self, write_dir=''):

        # Get all the fields used to write the header
        rec_write_fields, sig_write_fields = self.get_write_fields()

        # Check the validity of individual fields used to write the header

        # Record specification fields (and comments)
        for field in rec_write_fields:
            self.check_field(field)

        # Signal specification fields.
        for field in sig_write_fields:
            self.check_field(field, channels=sig_write_fields[field])

        # Check the cohesion of fields used to write the header
        self.check_field_cohesion(rec_write_fields, list(sig_write_fields))

        # Write the header file using the specified fields
        self.wr_header_file(rec_write_fields, sig_write_fields, write_dir)


    def get_write_fields(self):
        """
        Get the list of fields used to write the header, separating
        record and signal specification fields.

        Does NOT include `d_signal` or `e_d_signal`.

        Returns the default required fields, the user defined fields, and their dependencies.
        rec_write_fields includes 'comment' if present.
        """

        # Record specification fields
        rec_write_fields = self.get_write_subset('record')

        # Add comments if any
        if self.comments != None:
            rec_write_fields.append('comments')

        # Determine whether there are signals. If so, get their required
        # fields.
        self.check_field('n_sig')
        if self.n_sig >  0:
            sig_write_fields = self.get_write_subset('signal')
        else:
            sig_write_fields = None

        return rec_write_fields, sig_write_fields

    # Set the object's attribute to its default value if it is missing
    # and there is a default. Not responsible for initializing the
    # attribute. That is done by the constructor.
    def set_default(self, field):

        # Record specification fields
        if field in rec_field_specs:
            # Return if no default to set, or if the field is already present.
            if rec_field_specs[field].write_def is None or getattr(self, field) is not None:
                return
            setattr(self, field, rec_field_specs[field].write_def)

        # Signal specification fields
        # Setting entire list default, not filling in blanks in lists.
        elif field in SIGNAL_FIELDS:

            # Specific dynamic case
            if field == 'file_name' and self.file_name is None:
                self.file_name = self.n_sig*[self.record_name+'.dat']
                return

            item = getattr(self, field)

            # Return if no default to set, or if the field is already present.
            if SIGNAL_FIELDS[field].write_def is None or item is not None:
                return

            # Set more specific defaults if possible
            if field == 'adc_res' and self.fmt is not None:
                self.adc_res=_signal.wfdbfmtres(self.fmt)
                return

            setattr(self, field, [SIGNAL_FIELDS[field].write_def]*self.n_sig)

    # Check the cohesion of fields used to write the header
    def check_field_cohesion(self, rec_write_fields, sig_write_fields):

        # If there are no signal specification fields, there is nothing to check.
        if self.n_sig>0:

            # The length of all signal specification fields must match n_sig
            # even if some of its elements are None.
            for f in sig_write_fields:
                if len(getattr(self, f)) != self.n_sig:
                    raise ValueError('The length of field: '+f+' must match field n_sig.')

            # Each file_name must correspond to only one fmt, (and only one byte offset if defined).
            datfmts = {}
            for ch in range(self.n_sig):
                if self.file_name[ch] not in datfmts:
                    datfmts[self.file_name[ch]] = self.fmt[ch]
                else:
                    if datfmts[self.file_name[ch]] != self.fmt[ch]:
                        raise ValueError('Each file_name (dat file) specified must have the same fmt')

            datoffsets = {}
            if self.byte_offset is not None:
                # At least one byte offset value exists
                for ch in range(self.n_sig):
                    if self.byte_offset[ch] is None:
                        continue
                    if self.file_name[ch] not in datoffsets:
                        datoffsets[self.file_name[ch]] = self.byte_offset[ch]
                    else:
                        if datoffsets[self.file_name[ch]] != self.byte_offset[ch]:
                            raise ValueError('Each file_name (dat file) specified must have the same byte offset')



    def wr_header_file(self, rec_write_fields, sig_write_fields, write_dir):
        # Write a header file using the specified fields
        header_lines=[]

        # Create record specification line
        recordline = ''
        # Traverse the ordered dictionary
        for field in rec_field_specs:
            # If the field is being used, add it with its delimiter
            if field in rec_write_fields:
                stringfield = str(getattr(self, field))
                # If fs is float, check whether it as an integer
                if field == 'fs' and isinstance(self.fs, float):
                    if round(self.fs, 8) == float(int(self.fs)):
                        stringfield = str(int(self.fs))
                recordline = recordline + rec_field_specs[field].delimiter + stringfield
        header_lines.append(recordline)

        # Create signal specification lines (if any) one channel at a time
        if self.n_sig>0:
            signallines = self.n_sig*['']
            for ch in range(self.n_sig):
                # Traverse the ordered dictionary
                for field in SIGNAL_FIELDS:
                    # If the field is being used, add each of its elements with the delimiter to the appropriate line
                    if field in sig_write_fields and sig_write_fields[field][ch]:
                        signallines[ch]=signallines[ch] + SIGNAL_FIELDS[field].delimiter + str(getattr(self, field)[ch])
                    # The 'baseline' field needs to be closed with ')'
                    if field== 'baseline':
                        signallines[ch]=signallines[ch] +')'

            header_lines = header_lines + signallines

        # Create comment lines (if any)
        if 'comments' in rec_write_fields:
            comment_lines = ['# '+comment for comment in self.comments]
            header_lines = header_lines + comment_lines

        lines_to_file(self.record_name+'.hea', write_dir, header_lines)


class MultiHeaderMixin(BaseHeaderMixin):
    """
    Mixin class with multi-segment header methods. Inherited by MultiRecord class.
    """

    # Set defaults for fields needed to write the header if they have defaults.
    # This is NOT called by rdheader. It is only called by the gateway wrsamp for convenience.
    # It is also not called by wrhea (this may be changed in the future) since
    # it is supposed to be an explicit function.

    # Not responsible for initializing the
    # attribute. That is done by the constructor.
    def set_defaults(self):
        for field in self.get_write_fields():
            self.set_default(field)

    # Write a wfdb header file. The signals or segments fields are not used.
    def wrheader(self, write_dir=''):

        # Get all the fields used to write the header
        write_fields = self.get_write_fields()

        # Check the validity of individual fields used to write the header
        for f in write_fields:
            self.check_field(f)

        # Check the cohesion of fields used to write the header
        self.check_field_cohesion()

        # Write the header file using the specified fields
        self.wr_header_file(write_fields, write_dir)


    # Get the list of fields used to write the multi-segment header.
    # Returns the default required fields, the user defined fields, and their dependencies.
    def get_write_fields(self):

        # Record specification fields
        write_fields=self.get_write_subset('record')

        # Segment specification fields are all mandatory
        write_fields = write_fields + ['seg_name', 'seg_len']

        # Comments
        if self.comments !=None:
            write_fields.append('comments')
        return write_fields

    # Set a field to its default value if there is a default.
    def set_default(self, field):

        # Record specification fields
        if field in rec_field_specs:
            # Return if no default to set, or if the field is already present.
            if rec_field_specs[field].write_def is None or getattr(self, field) is not None:
                return
            setattr(self, field, rec_field_specs[field].write_def)



    # Check the cohesion of fields used to write the header
    def check_field_cohesion(self):

        # The length of seg_name and seg_len must match n_seg
        for f in ['seg_name', 'seg_len']:
            if len(getattr(self, f)) != self.n_seg:
                raise ValueError('The length of field: '+f+' does not match field n_seg.')

        # Check the sum of the 'seg_len' fields against 'sig_len'
        if np.sum(self.seg_len) != self.sig_len:
            raise ValueError("The sum of the 'seg_len' fields do not match the 'sig_len' field")


    # Write a header file using the specified fields
    def wr_header_file(self, write_fields, write_dir):

        header_lines=[]

        # Create record specification line
        recordline = ''
        # Traverse the ordered dictionary
        for field in rec_field_specs:
            # If the field is being used, add it with its delimiter
            if field in write_fields:
                recordline = recordline + rec_field_specs[field].delimiter + str(getattr(self, field))
        header_lines.append(recordline)

        # Create segment specification lines
        segmentlines = self.n_seg*['']
        # For both fields, add each of its elements with the delimiter to the appropriate line
        for field in ['seg_name', 'seg_name']:
            for segnum in range(0, self.n_seg):
                segmentlines[segnum] = segmentlines[segnum] + seg_field_specs[field].delimiter + str(getattr(self, field)[segnum])

        header_lines = header_lines + segmentlines

        # Create comment lines (if any)
        if 'comments' in write_fields:
            comment_lines = ['# '+comment for comment in self.comments]
            header_lines = header_lines + comment_lines

        lines_to_file(self.record_name+'.hea', header_lines, write_dir)


    def get_sig_segments(self, sig_name=None):
        """
        Get a list of the segment numbers that contain a particular signal
        (or a dictionary of segment numbers for a list of signals)
        Only works if information about the segments has been read in
        """
        if self.segments is None:
            raise Exception("The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rsegment_fieldsments=True")

        # Default value = all signal names.
        if sig_name is None:
            sig_name = self.get_sig_name()

        if isinstance(sig_name, list):
            sigdict = {}
            for sig in sig_name:
                sigdict[sig] = self.get_sig_segments(sig)
            return sigdict
        elif isinstance(sig_name, str):
            sigsegs = []
            for i in range(self.n_seg):
                if self.seg_name[i] != '~' and sig_name in self.segments[i].sig_name:
                    sigsegs.append(i)
            return sigsegs
        else:
            raise TypeError('sig_name must be a string or a list of strings')

    # Get the signal names for the entire record
    def get_sig_name(self):
        if self.segments is None:
            raise Exception("The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rsegment_fieldsments=True")

        if self.layout == 'fixed':
            for i in range(self.n_seg):
                if self.seg_name[i] != '~':
                    sig_name = self.segments[i].sig_name
                    break
        else:
            sig_name = self.segments[0].sig_name

        return sig_name


# Read header file to get comment and non-comment lines
def get_header_lines(record_name, pb_dir):
    # Read local file
    if pb_dir is None:
        with open(record_name + ".hea", 'r') as fp:
            # Record line followed by signal/segment lines if any
            header_lines = []
            # Comment lines
            comment_lines = []
            for line in fp:
                line = line.strip()
                # Comment line
                if line.startswith('#'):
                    comment_lines.append(line)
                # Non-empty non-comment line = header line.
                elif line:
                    # Look for a comment in the line
                    ci = line.find('#')
                    if ci > 0:
                        header_lines.append(line[:ci])
                        # comment on same line as header line
                        comment_lines.append(line[ci:])
                    else:
                        header_lines.append(line)
    # Read online header file
    else:
        header_lines, comment_lines = download.stream_header(record_name, pb_dir)

    return header_lines, comment_lines


def _read_record_line(record_line):
    """
    Extract fields from a record line string into a dictionary

    """
    # Dictionary for record fields
    record_fields = {}

    # Read string fields from record line
    (record_fields['record_name'], record_fields['n_seg'],
     record_fields['n_sig'], record_fields['fs'],
     record_fields['counter_freq'], record_fields['base_counter'],
     record_fields['sig_len'], record_fields['base_time'],
     record_fields['base_date']) = re.findall(_rx_record, record_line)[0]

    for field in rec_field_specs:
        # Replace empty strings with their read defaults (which are
        # mostly None)
        if record_fields[field] == '':
            record_fields[field] = rec_field_specs[field].read_default
        # Typecast non-empty strings for numerical and date/time fields
        else:
            if rec_field_specs[field].allowed_types is int_types:
                record_fields[field] = int(record_fields[field])
            # fs may be read as float or int
            elif field == 'fs':
                fs = float(record_fields['fs'])
                if round(fs, 8) == float(int(fs)):
                    fs = int(fs)
                record_fields['fs'] = fs

    return record_fields


# Extract fields from signal line strings into a dictionary
def _read_signal_lines(signal_lines):
    # Dictionary for signal fields
    signal_fields = {}

    # Each dictionary field is a list
    for field in SIGNAL_FIELDS:
        signal_fields[field] = [None]*len(signal_lines)

    # Read string fields from signal line
    for i in range(len(signal_lines)):
        (signal_fields['file_name'][i], signal_fields['fmt'][i],
         signal_fields['samps_per_frame'][i], signal_fields['skew'][i],
         signal_fields['byte_offset'][i], signal_fields['adc_gain'][i],
         signal_fields['baseline'][i], signal_fields['units'][i],
         signal_fields['adc_res'][i], signal_fields['adc_zero'][i],
         signal_fields['init_value'][i], signal_fields['checksum'][i],
         signal_fields['block_size'][i],
         signal_fields['sig_name'][i]) = _rx_signal.findall(signal_lines[i])[0]

        for field in SIGNAL_FIELDS:
            # Replace empty strings with their read defaults (which are mostly None)
            # Note: Never set a field to None. [None]* n_sig is accurate, indicating
            # that different channels can be present or missing.
            if signal_fields[field][i] == '':
                signal_fields[field][i] = SIGNAL_FIELDS[field].read_default

                # Special case: missing baseline defaults to ADCzero if present
                if field == 'baseline' and signal_fields['adc_zero'][i] != '':
                    signal_fields['baseline'][i] = int(signal_fields['adc_zero'][i])
            # Typecast non-empty strings for numerical fields
            else:
                if SIGNAL_FIELDS[field].allowed_types is int_types:
                    signal_fields[field][i] = int(signal_fields[field][i])
                elif SIGNAL_FIELDS[field].allowed_types is float_types:
                    signal_fields[field][i] = float(signal_fields[field][i])
                    # Special case: gain of 0 means 200
                    if field == 'adc_gain' and signal_fields['adc_gain'][i] == 0:
                        signal_fields['adc_gain'][i] = 200.

    return signal_fields


def _read_segment_lines(segment_lines):
    """
    Extract fields from segment line strings into a dictionary

    """
    # Dictionary for segment fields
    segment_fields = {}

    # Each dictionary field is a list
    for field in seg_field_specs:
        segment_fields[field] = [None]*len(segment_lines)

    # Read string fields from signal line
    for i in range(0, len(segment_lines)):
        (segment_fields['seg_name'][i], segment_fields['seg_len'][i]) = _rx_segment.findall(segment_lines[i])[0]

        for field in seg_field_specs:
            # Replace empty strings with their read defaults (which are mostly None)
            if segment_fields[field][i] == '':
                segment_fields[field][i] = seg_field_specs[field].read_default
            # Typecast non-empty strings for numerical field
            else:
                if field == 'seg_len':
                    segment_fields[field][i] = int(segment_fields[field][i])

    return segment_fields


def lines_to_file(file_name, write_dir, lines):
    # Write each line in a list of strings to a text file
    f = open(os.path.join(write_dir, file_name), 'w')
    for l in lines:
        f.write("%s\n" % l)
    f.close()
