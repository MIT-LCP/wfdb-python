from calendar import monthrange
from collections import OrderedDict
import numpy as np
import os
import re

from . import download
from . import _signal


class FieldSpecification(object):
    """
    Class for storing specifications for wfdb record fields
    """
    def __init__(self, allowed_types, delimiter, dependency, write_req,
                 read_def, write_def):
        # Data types the field (or its elements) can be
        self.allowed_types = allowed_types
        # The text delimiter that preceeds the field if it is a field that gets written to header files.
        self.delimiter = delimiter
        # The required/dependent field which must also be present
        self.dependency = dependency
        # Whether the field is always required for writing a header (more stringent than origin WFDB library)
        self.write_req = write_req
        # The default value for the field when read if any
        self.read_def = read_def
        # The default value for the field to fill in before writing if any
        self.write_def = write_def

        # The read vs write default values are different for 2 reasons:
        # 1. We want to force the user to be explicit with certain important
        #    fields when writing WFDB records fields, without affecting
        #    existing WFDB headers when reading.
        # 2. Certain unimportant fields may be dependencies of other
        #    important fields. When writing, we want to fill in defaults
        #    so that the user doesn't need to. But when reading, it should
        #    be clear that the fields are missing.

int_types = (int, np.int64, np.int32, np.int16, np.int8)
float_types = int_types + (float, np.float64, np.float32)
int_dtypes = ('int64', 'uint64', 'int32', 'uint32','int16','uint16')

# Record specification fields
rec_field_specs = OrderedDict([('record_name', FieldSpecification((str), '', None, True, None, None)),
                         ('n_seg', FieldSpecification(int_types, '/', 'record_name', True, None, None)),
                         ('n_sig', FieldSpecification(int_types, ' ', 'record_name', True, None, None)),
                         ('fs', FieldSpecification(float_types, ' ', 'n_sig', True, 250, None)),
                         ('counter_freq', FieldSpecification(float_types, '/', 'fs', False, None, None)),
                         ('base_counter', FieldSpecification(float_types, '(', 'counter_freq', False, None, None)),
                         ('sig_len', FieldSpecification(int_types, ' ', 'fs', True, None, None)),
                         ('base_time', FieldSpecification((str), ' ', 'sig_len', False, None, '00:00:00')),
                         ('base_date', FieldSpecification((str), ' ', 'base_time', False, None, None))])

# Signal specification fields.
sig_field_specs = OrderedDict([('file_name', FieldSpecification((str), '', None, True, None, None)),
                         ('fmt', FieldSpecification((str), ' ', 'file_name', True, None, None)),
                         ('samps_per_frame', FieldSpecification(int_types, 'x', 'fmt', False, 1, None)),
                         ('skew', FieldSpecification(int_types, ':', 'fmt', False, None, None)),
                         ('byte_offset', FieldSpecification(int_types, '+', 'fmt', False, None, None)),
                         ('adc_gain', FieldSpecification(float_types, ' ', 'fmt', True, 200., None)),
                         ('baseline', FieldSpecification(int_types, '(', 'adc_gain', True, 0, None)),
                         ('units', FieldSpecification((str), '/', 'adc_gain', True, 'mV', None)),
                         ('adc_res', FieldSpecification(int_types, ' ', 'adc_gain', False, None, 0)),
                         ('adc_zero', FieldSpecification(int_types, ' ', 'adc_res', False, None, 0)),
                         ('init_value', FieldSpecification(int_types, ' ', 'adc_zero', False, None, None)),
                         ('checksum', FieldSpecification(int_types, ' ', 'init_value', False, None, None)),
                         ('block_size', FieldSpecification(int_types, ' ', 'checksum', False, None, 0)),
                         ('sig_name', FieldSpecification((str), ' ', 'block_size', False, None, None))])

# Segment specification fields.
seg_field_specs = OrderedDict([('seg_name', FieldSpecification((str), '', None, True, None, None)),
                         ('seg_len', FieldSpecification(int_types, ' ', 'seg_name', True, None, None))])


# Regexp objects for reading headers

# Record Line Fields
rx_record = re.compile(
    ''.join(
        [
            "(?P<record_name>[-\w]+)/?(?P<n_seg>\d*)[ \t]+",
            "(?P<n_sig>\d+)[ \t]*",
            "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<base_counter>\d*\.?\d*)\)?[ \t]*",
            "(?P<sig_len>\d*)[ \t]*",
            "(?P<base_time>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
            "(?P<base_date>\d{,2}/?\d{,2}/?\d{,4})"]))

# Signal Line Fields
rx_signal = re.compile(
    ''.join(
        [
            "(?P<file_name>[-\w]+\.?[\w]*~?)[ \t]+(?P<fmt>\d+)x?"
            "(?P<samps_per_frame>\d*):?(?P<skew>\d*)\+?(?P<byte_offset>\d*)[ \t]*",
            "(?P<adc_gain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^\-\?%]*)[ \t]*",
            "(?P<adc_res>\d*)[ \t]*(?P<adc_zero>-?\d*)[ \t]*(?P<init_value>-?\d*)[ \t]*",
            "(?P<checksum>-?\d*)[ \t]*(?P<block_size>\d*)[ \t]*(?P<sig_name>[\S]?[^\t\n\r\f\v]*)"]))

# Segment Line Fields
rx_segment = re.compile('(?P<seg_name>\w*~?)[ \t]+(?P<seg_len>\d+)')


class BaseHeaderMixin(object):
    """
    Mixin class with multi-segment header methods. Inherited by Record and
    MultiRecord classes
    """

    def get_write_subset(self, spec_fields):
        """
        Helper function for get_write_fields.

        - spec_fields is the set of specification fields
        For record specs, it returns a list of all fields needed.
        For signal specs, it returns a dictionary of all fields needed,
        with keys = field and value = list of 1 or 0 indicating channel for the field
        """

        # record specification fields
        if spec_fields == 'record':
            write_fields=[]
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
        elif spec_fields == 'signal':
            # List of lists for each channel
            write_fields=[]

            allwrite_fields=[]
            fieldspecs = OrderedDict(reversed(list(sig_field_specs.items())))

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
                            rf=fieldspecs[rf].dependency

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
        recwrite_fields, sigwrite_fields = self.get_write_fields()

        # Check the validity of individual fields used to write the header

        # Record specification fields (and comments)
        for f in recwrite_fields:
            self.check_field(f)

        # Signal specification fields.
        for f in sigwrite_fields:
            self.check_field(f, sigwrite_fields[f])

        # Check the cohesion of fields used to write the header
        self.check_field_cohesion(recwrite_fields, list(sigwrite_fields))

        # Write the header file using the specified fields
        self.wr_header_file(recwrite_fields, sigwrite_fields, write_dir)


    # Get the list of fields used to write the header. (Does NOT include d_signal or e_d_signal.)
    # Separate items by record and signal specification field.
    # Returns the default required fields, the user defined fields, and their dependencies.
    # recwrite_fields includes 'comment' if present.
    def get_write_fields(self):

        # Record specification fields
        recwrite_fields=self.get_write_subset('record')

        # Add comments if any
        if self.comments != None:
            recwrite_fields.append('comments')

        # Determine whether there are signals. If so, get their required fields.
        self.check_field('n_sig')
        if self.n_sig>0:
            sigwrite_fields=self.get_write_subset('signal')
        else:
            sigwrite_fields = None

        return recwrite_fields, sigwrite_fields

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
        elif field in sig_field_specs:

            # Specific dynamic case
            if field == 'file_name' and self.file_name is None:
                self.file_name = self.n_sig*[self.record_name+'.dat']
                return

            item = getattr(self, field)

            # Return if no default to set, or if the field is already present.
            if sig_field_specs[field].write_def is None or item is not None:
                return

            # Set more specific defaults if possible
            if field == 'adc_res' and self.fmt is not None:
                self.adc_res=_signal.wfdbfmtres(self.fmt)
                return

            setattr(self, field, [sig_field_specs[field].write_def]*self.n_sig)

    # Check the cohesion of fields used to write the header
    def check_field_cohesion(self, recwrite_fields, sigwrite_fields):

        # If there are no signal specification fields, there is nothing to check.
        if self.n_sig>0:

            # The length of all signal specification fields must match n_sig
            # even if some of its elements are None.
            for f in sigwrite_fields:
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



    def wr_header_file(self, recwrite_fields, sigwrite_fields, write_dir):
        # Write a header file using the specified fields
        header_lines=[]

        # Create record specification line
        recordline = ''
        # Traverse the ordered dictionary
        for field in rec_field_specs:
            # If the field is being used, add it with its delimiter
            if field in recwrite_fields:
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
                for field in sig_field_specs:
                    # If the field is being used, add each of its elements with the delimiter to the appropriate line
                    if field in sigwrite_fields and sigwrite_fields[field][ch]:
                        signallines[ch]=signallines[ch] + sig_field_specs[field].delimiter + str(getattr(self, field)[ch])
                    # The 'baseline' field needs to be closed with ')'
                    if field== 'baseline':
                        signallines[ch]=signallines[ch] +')'

            header_lines = header_lines + signallines

        # Create comment lines (if any)
        if 'comments' in recwrite_fields:
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
            raise Exception("The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rd_segments=True")

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
            raise Exception("The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rd_segments=True")

        if self.layout == 'Fixed':
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


# Extract fields from a record line string into a dictionary
def read_rec_line(rec_line):

    # Dictionary for record fields
    d_rec = {}

    # Read string fields from record line
    (d_rec['record_name'], d_rec['n_seg'], d_rec['n_sig'], d_rec['fs'],
    d_rec['counter_freq'], d_rec['base_counter'], d_rec['sig_len'],
    d_rec['base_time'], d_rec['base_date']) = re.findall(rx_record, rec_line)[0]

    for field in rec_field_specs:
        # Replace empty strings with their read defaults (which are mostly None)
        if d_rec[field] == '':
            d_rec[field] = rec_field_specs[field].read_def
        # Typecast non-empty strings for numerical fields
        else:
            if rec_field_specs[field].allowed_types is int_types:
                d_rec[field] = int(d_rec[field])
            # fs may be read as float or int
            elif field == 'fs':
                fs = float(d_rec['fs'])
                if round(fs, 8) == float(int(fs)):
                    fs = int(fs)
                d_rec['fs'] = fs

    return d_rec

# Extract fields from signal line strings into a dictionary
def read_sig_lines(sig_lines):
    # Dictionary for signal fields
    d_sig = {}

    # Each dictionary field is a list
    for field in sig_field_specs:
        d_sig[field] = [None]*len(sig_lines)

    # Read string fields from signal line
    for i in range(0, len(sig_lines)):
        (d_sig['file_name'][i], d_sig['fmt'][i],
            d_sig['samps_per_frame'][i],
            d_sig['skew'][i],
            d_sig['byte_offset'][i],
            d_sig['adc_gain'][i],
            d_sig['baseline'][i],
            d_sig['units'][i],
            d_sig['adc_res'][i],
            d_sig['adc_zero'][i],
            d_sig['init_value'][i],
            d_sig['checksum'][i],
            d_sig['block_size'][i],
            d_sig['sig_name'][i]) = rx_signal.findall(sig_lines[i])[0]

        for field in sig_field_specs:
            # Replace empty strings with their read defaults (which are mostly None)
            # Note: Never set a field to None. [None]* n_sig is accurate, indicating
            # that different channels can be present or missing.
            if d_sig[field][i] == '':
                d_sig[field][i] = sig_field_specs[field].read_def

                # Special case: missing baseline defaults to ADCzero if present
                if field == 'baseline' and d_sig['adc_zero'][i] != '':
                    d_sig['baseline'][i] = int(d_sig['adc_zero'][i])
            # Typecast non-empty strings for numerical fields
            else:
                if sig_field_specs[field].allowed_types is int_types:
                    d_sig[field][i] = int(d_sig[field][i])
                elif sig_field_specs[field].allowed_types is float_types:
                    d_sig[field][i] = float(d_sig[field][i])
                    # Special case: gain of 0 means 200
                    if field == 'adc_gain' and d_sig['adc_gain'][i] == 0:
                        d_sig['adc_gain'][i] = 200.

    return d_sig


# Extract fields from segment line strings into a dictionary
def read_seg_lines(seg_lines):

    # Dictionary for signal fields
    d_seg = {}

    # Each dictionary field is a list
    for field in seg_field_specs:
        d_seg[field] = [None]*len(seg_lines)

    # Read string fields from signal line
    for i in range(0, len(seg_lines)):
        (d_seg['seg_name'][i], d_seg['seg_len'][i]) = rx_segment.findall(seg_lines[i])[0]

        for field in seg_field_specs:
            # Replace empty strings with their read defaults (which are mostly None)
            if d_seg[field][i] == '':
                d_seg[field][i] = seg_field_specs[field].read_def
            # Typecast non-empty strings for numerical field
            else:
                if field == 'seg_len':
                    d_seg[field][i] = int(d_seg[field][i])

    return d_seg


def lines_to_file(file_name, write_dir, lines):
    # Write each line in a list of strings to a text file
    f = open(os.path.join(write_dir, file_name), 'w')
    for l in lines:
        f.write("%s\n" % l)
    f.close()
