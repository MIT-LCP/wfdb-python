import numpy as np
import re
import os
from collections import OrderedDict
from calendar import monthrange
from . import _signals
from . import downloads

# Class of common methods for single and multi-segment headers
class BaseHeadersMixin(object):


    # Helper function for getwritefields
    # specfields is the set of specification fields
    # For record specs, it returns a list of all fields needed.
    # For signal specs, it returns a dictionary of all fields needed,
    # with keys = field and value = list of 1 or 0 indicating channel for the field
    def getwritesubset(self, specfields):
        
        # record specification fields
        if specfields == 'record':
            writefields=[]
            fieldspecs = OrderedDict(reversed(list(recfieldspecs.items())))
            # Remove this requirement for single segs
            if not hasattr(self, 'nseg'): 
                del(fieldspecs['nseg'])

            for f in fieldspecs:
                if f in writefields:
                    continue
                # If the field is required by default or has been defined by the user
                if fieldspecs[f].write_req or getattr(self, f) is not None:
                    rf=f
                    # Add the field and its recursive dependencies
                    while rf is not None:
                        writefields.append(rf)
                        rf=fieldspecs[rf].dependency
            # Add comments if any
            if getattr(self, 'comments') is not None:
                writefields.append('comments')
            
        # signal spec field. Need to return a potentially different list for each channel. 
        elif specfields == 'signal':
            # List of lists for each channel
            writefields=[]
            
            allwritefields=[]
            fieldspecs = OrderedDict(reversed(list(sigfieldspecs.items())))

            for ch in range(self.nsig):
                # The fields needed for this channel
                writefieldsch = []
                for f in fieldspecs:
                    if f in writefieldsch:
                        continue

                    fielditem = getattr(self, f)

                    # If the field is required by default or has been defined by the user
                    if fieldspecs[f].write_req or (fielditem is not None and fielditem[ch] is not None):
                        rf=f
                        # Add the field and its recursive dependencies
                        while rf is not None:
                            writefieldsch.append(rf)
                            rf=fieldspecs[rf].dependency

                writefields.append(writefieldsch)

            # Convert the list of lists to a single dictionary.
            # keys = field and value = list of 1 or 0 indicating channel for the field
            dictwritefields = {}

            # For fields present in any channel:     
            for f in set([i for wsub in writefields for i in wsub]):
                dictwritefields[f] = [0]*self.nsig

                for ch in range(self.nsig):
                    if f in writefields[ch]:
                        dictwritefields[f][ch] = 1

            writefields = dictwritefields

        
        return writefields
        
        
# Class with single-segment header methods
# To be inherited by WFDBrecord from records.py.
class HeadersMixin(BaseHeadersMixin):
    
    # Set defaults for fields needed to write the header if they have defaults.
    # This is NOT called by rdheader. It is only called by the gateway wrsamp for convenience.
    # It is also not called by wrhea (this may be changed in the future) since 
    # it is supposed to be an explicit function. 

    # Not responsible for initializing the 
    # attribute. That is done by the constructor. 
    def setdefaults(self):

        rfields, sfields = self.getwritefields()
        for f in rfields:
            self.setdefault(f)
        for f in sfields:
            self.setdefault(f)

    # Write a wfdb header file. The signals or segments fields are not used. 
    def wrheader(self):

        # Get all the fields used to write the header
        recwritefields, sigwritefields = self.getwritefields()

        # Check the validity of individual fields used to write the header 

        # Record specification fields (and comments)
        for f in recwritefields:
            self.checkfield(f)

        # Signal specification fields.
        for f in sigwritefields:
            self.checkfield(f, sigwritefields[f])

        # Check the cohesion of fields used to write the header
        self.checkfieldcohesion(recwritefields, list(sigwritefields))
        
        # Write the header file using the specified fields
        self.wrheaderfile(recwritefields, sigwritefields)
    

    # Get the list of fields used to write the header. (Does NOT include d_signals.)
    # Separate items by record and signal specification field.
    # Returns the default required fields, the user defined fields, and their dependencies.
    # recwritefields includes 'comment' if present.
    def getwritefields(self):

        # Record specification fields
        recwritefields=self.getwritesubset('record')

        # Add comments if any
        if self.comments != None:
            recwritefields.append('comments')

        # Determine whether there are signals. If so, get their required fields.
        self.checkfield('nsig')
        if self.nsig>0:
            sigwritefields=self.getwritesubset('signal')
        else:
            sigwritefields = None
        
        return recwritefields, sigwritefields

    # Set the object's attribute to its default value if it is missing 
    # and there is a default. Not responsible for initializing the 
    # attribute. That is done by the constructor. 
    def setdefault(self, field):
        
        # Record specification fields
        if field in recfieldspecs:
            # Return if no default to set, or if the field is already present.
            if recfieldspecs[field].write_def is None or getattr(self, field) is not None:
                return
            setattr(self, field, recfieldspecs[field].write_def)
        
        # Signal specification fields
        # Setting entire list default, not filling in blanks in lists.
        elif field in sigfieldspecs:
            
            # Specific dynamic case
            if field == 'filename' and self.filename is None:
                self.filename = self.nsig*[self.recordname+'.dat']
                return
            
            item = getattr(self, field)

            # Return if no default to set, or if the field is already present.
            if sigfieldspecs[field].write_def is None or item is not None:
                return

            # Set more specific defaults if possible
            if field == 'adcres' and self.fmt is not None:
                self.adcres=_signals.wfdbfmtres(self.fmt)
                return
                
            setattr(self, field, [sigfieldspecs[field].write_def]*self.nsig)

    # Check the cohesion of fields used to write the header
    def checkfieldcohesion(self, recwritefields, sigwritefields):

        # If there are no signal specification fields, there is nothing to check. 
        if self.nsig>0:

            # The length of all signal specification fields must match nsig
            # even if some of its elements are None. 
            for f in sigwritefields:
                if len(getattr(self, f)) != self.nsig:
                    raise ValueError('The length of field: '+f+' must match field nsig.')

            # Each filename must correspond to only one fmt, (and only one byte offset if defined). 
            datfmts = {}
            for ch in range(self.nsig):
                if self.filename[ch] not in datfmts:
                    datfmts[self.filename[ch]] = self.fmt[ch]
                else:
                    if datfmts[self.filename[ch]] != self.fmt[ch]:
                        raise ValueError('Each filename (dat file) specified must have the same fmt')
            
            datoffsets = {}
            if self.byteoffset is not None:
                # At least one byte offset value exists
                for ch in range(self.nsig):
                    if self.byteoffset[ch] is None:
                        continue
                    if self.filename[ch] not in datoffsets:
                        datoffsets[self.filename[ch]] = self.byteoffset[ch]
                    else:
                        if datoffsets[self.filename[ch]] != self.byteoffset[ch]:
                            raise ValueError('Each filename (dat file) specified must have the same byte offset')


    # Write a header file using the specified fields
    def wrheaderfile(self, recwritefields, sigwritefields):

        headerlines=[]

        # Create record specification line
        recordline = ''
        # Traverse the ordered dictionary
        for field in recfieldspecs:
            # If the field is being used, add it with its delimiter
            if field in recwritefields:
                recordline = recordline + recfieldspecs[field].delimiter + str(getattr(self, field))
        headerlines.append(recordline)

        # Create signal specification lines (if any) one channel at a time
        if self.nsig>0:
            signallines = self.nsig*['']
            for ch in range(self.nsig):
                # Traverse the ordered dictionary
                for field in sigfieldspecs:
                    # If the field is being used, add each of its elements with the delimiter to the appropriate line 
                    if field in sigwritefields and sigwritefields[field][ch]:
                        signallines[ch]=signallines[ch] + sigfieldspecs[field].delimiter + str(getattr(self, field)[ch])
                    # The 'baseline' field needs to be closed with ')'
                    if field== 'baseline':
                        signallines[ch]=signallines[ch] +')'

            headerlines = headerlines + signallines

        # Create comment lines (if any)
        if 'comments' in recwritefields:
            commentlines = ['# '+comment for comment in self.comments]
            headerlines = headerlines + commentlines

        linestofile(self.recordname+'.hea', headerlines)



# Class with multi-segment header methods
# To be inherited by WFDBmultirecord from records.py.
class MultiHeadersMixin(BaseHeadersMixin):
    
    # Set defaults for fields needed to write the header if they have defaults.
    # This is NOT called by rdheader. It is only called by the gateway wrsamp for convenience.
    # It is also not called by wrhea (this may be changed in the future) since 
    # it is supposed to be an explicit function. 

    # Not responsible for initializing the 
    # attribute. That is done by the constructor. 
    def setdefaults(self):
        for field in self.getwritefields():
            self.setdefault(field)

    # Write a wfdb header file. The signals or segments fields are not used. 
    def wrheader(self):

        # Get all the fields used to write the header
        writefields = self.getwritefields()

        # Check the validity of individual fields used to write the header 
        for f in writefields:
            self.checkfield(f)
        
        # Check the cohesion of fields used to write the header
        self.checkfieldcohesion()
        
        # Write the header file using the specified fields
        self.wrheaderfile(writefields)


    # Get the list of fields used to write the multi-segment header. 
    # Returns the default required fields, the user defined fields, and their dependencies.
    def getwritefields(self):

        # Record specification fields
        writefields=self.getwritesubset('record')

        # Segment specification fields are all mandatory
        writefields = writefields + ['segname', 'seglen']

        # Comments
        if self.comments !=None:
            writefields.append('comments')
        return writefields

    # Set a field to its default value if there is a default.
    def setdefault(self, field):
        
        # Record specification fields
        if field in recfieldspecs:
            # Return if no default to set, or if the field is already present.
            if recfieldspecs[field].write_def is None or getattr(self, field) is not None:
                return
            setattr(self, field, recfieldspecs[field].write_def)

            

    # Check the cohesion of fields used to write the header
    def checkfieldcohesion(self):

        # The length of segname and seglen must match nseg
        for f in ['segname', 'seglen']:
            if len(getattr(self, f)) != self.nseg:
                raise ValueError('The length of field: '+f+' does not match field nseg.')

        # Check the sum of the 'seglen' fields against 'siglen'
        if np.sum(self.seglen) != self.siglen:
            raise ValueError("The sum of the 'seglen' fields do not match the 'siglen' field")


    # Write a header file using the specified fields
    def wrheaderfile(self, writefields):

        headerlines=[]

        # Create record specification line
        recordline = ''
        # Traverse the ordered dictionary
        for field in recfieldspecs:
            # If the field is being used, add it with its delimiter
            if field in writefields:
                recordline = recordline + recfieldspecs[field].delimiter + str(getattr(self, field))
        headerlines.append(recordline)

        # Create segment specification lines 
        segmentlines = self.nseg*['']
        # For both fields, add each of its elements with the delimiter to the appropriate line 
        for field in ['segname', 'segname']:
            for segnum in range(0, self.nseg):
                segmentlines[segnum] = segmentlines[segnum] + segfieldspecs[field].delimiter + str(getattr(self, field)[segnum])

        headerlines = headerlines + segmentlines

        # Create comment lines (if any)
        if 'comments' in writefields:
            commentlines = ['# '+comment for comment in self.comments]
            headerlines = headerlines + commentlines

        linestofile(self.recordname+'.hea', headerlines)


# Regexp objects for reading headers

# Record Line Fields
rxRECORD = re.compile(
    ''.join(
        [
            "(?P<recordname>[-\w]+)/?(?P<nseg>\d*)[ \t]+",
            "(?P<nsig>\d+)[ \t]*",
            "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<basecounter>\d*\.?\d*)\)?[ \t]*",
            "(?P<siglen>\d*)[ \t]*",
            "(?P<basetime>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
            "(?P<basedate>\d{,2}/?\d{,2}/?\d{,4})"]))

# Signal Line Fields
rxSIGNAL = re.compile(
    ''.join(
        [
            "(?P<filename>[\w]*\.?[\w]*~?)[ \t]+(?P<fmt>\d+)x?"
            "(?P<sampsperframe>\d*):?(?P<skew>\d*)\+?(?P<byteoffset>\d*)[ \t]*",
            "(?P<adcgain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^/-]*)[ \t]*",
            "(?P<adcres>\d*)[ \t]*(?P<adczero>-?\d*)[ \t]*(?P<initvalue>-?\d*)[ \t]*",
            "(?P<checksum>-?\d*)[ \t]*(?P<blocksize>\d*)[ \t]*(?P<signame>[\S]*)"]))

# Segment Line Fields
rxSEGMENT = re.compile('(?P<segname>\w*~?)[ \t]+(?P<seglen>\d+)')


# Read header file to get comment and non-comment lines
def getheaderlines(recordname, pbdir):
    # Read local file
    if pbdir is None:
        with open(recordname + ".hea", 'r') as fp:
            # Record line followed by signal/segment lines if any
            headerlines = []  
            # Comment lines
            commentlines = []  
            for line in fp:
                line = line.strip()
                # Comment line
                if line.startswith('#'):  
                    commentlines.append(line)
                # Non-empty non-comment line = header line.
                elif line:  
                    # Look for a comment in the line
                    ci = line.find('#')
                    if ci > 0:
                        headerlines.append(line[:ci])
                        # comment on same line as header line
                        commentlines.append(line[ci:])
                    else:
                        headerlines.append(line)
    # Read online header file
    else:
        headerlines, commentlines = downloads.streamheader(recordname, pbdir)

    return headerlines, commentlines


# Extract fields from a record line string into a dictionary
def read_rec_line(recline):

    # Dictionary for record fields
    d_rec = {}

    # Read string fields from record line
    (d_rec['recordname'], d_rec['nseg'], d_rec['nsig'], d_rec['fs'], 
    d_rec['counterfreq'], d_rec['basecounter'], d_rec['siglen'],
    d_rec['basetime'], d_rec['basedate']) = re.findall(rxRECORD, recline)[0]

    for field in recfieldspecs:
        # Replace empty strings with their read defaults (which are mostly None)
        if d_rec[field] == '':
            d_rec[field] = recfieldspecs[field].read_def
        # Typecast non-empty strings for numerical fields
        else:
            if recfieldspecs[field].allowedtypes is inttypes:
                d_rec[field] = int(d_rec[field])
            elif recfieldspecs[field].allowedtypes is floattypes:
                d_rec[field] = float(d_rec[field])

    return d_rec

# Extract fields from signal line strings into a dictionary
def read_sig_lines(siglines):
    # Dictionary for signal fields
    d_sig = {}

    # Each dictionary field is a list
    for field in sigfieldspecs:
        d_sig[field] = [None]*len(siglines)

    # Read string fields from signal line
    for i in range(0, len(siglines)):
        (d_sig['filename'][i], d_sig['fmt'][i],
            d_sig['sampsperframe'][i],
            d_sig['skew'][i],
            d_sig['byteoffset'][i],
            d_sig['adcgain'][i],
            d_sig['baseline'][i],
            d_sig['units'][i],
            d_sig['adcres'][i],
            d_sig['adczero'][i],
            d_sig['initvalue'][i],
            d_sig['checksum'][i],
            d_sig['blocksize'][i],
            d_sig['signame'][i]) = rxSIGNAL.findall(siglines[i])[0]

        for field in sigfieldspecs:
            # Replace empty strings with their read defaults (which are mostly None)
            # Note: Never set a field to None. [None]* nsig is accurate, indicating 
            # that different channels can be present or missing. 
            if d_sig[field][i] == '':
                d_sig[field][i] = sigfieldspecs[field].read_def

                # Special case: missing baseline defaults to ADCzero if present
                if field == 'baseline' and d_sig['adczero'][i] != '':
                    d_sig['baseline'][i] = int(d_sig['adczero'][i])
            # Typecast non-empty strings for numerical fields
            else:
                if sigfieldspecs[field].allowedtypes is inttypes:
                    d_sig[field][i] = int(d_sig[field][i])
                elif sigfieldspecs[field].allowedtypes is floattypes:
                    d_sig[field][i] = float(d_sig[field][i])
                    # Special case: gain of 0 means 200
                    if field == 'adcgain' and d_sig['adcgain'][i] == 0:
                        d_sig['adcgain'][i] = 200.

    return d_sig


# Extract fields from segment line strings into a dictionary
def read_seg_lines(seglines):

    # Dictionary for signal fields
    d_seg = {}

    # Each dictionary field is a list
    for field in segfieldspecs:
        d_seg[field] = [None]*len(seglines)

    # Read string fields from signal line
    for i in range(0, len(seglines)):
        (d_seg['segname'][i], d_seg['seglen'][i]) = rxSEGMENT.findall(seglines[i])[0]

        for field in segfieldspecs:
            # Replace empty strings with their read defaults (which are mostly None)
            if d_seg[field][i] == '':
                d_seg[field][i] = segfieldspecs[field].read_def
            # Typecast non-empty strings for numerical field
            else:
                if field == 'seglen':
                    d_seg[field][i] = int(d_seg[field][i])
                                 
    return d_seg

# Write each line in a list of strings to a text file
def linestofile(filename, lines):
    f = open(filename,'w')
    for l in lines:
        f.write("%s\n" % l)
    f.close()              


# Specifications of WFDB header fields.
class WFDBheaderspecs():
    
    def __init__(self, allowedtypes, delimiter, dependency, write_req, read_def, write_def):
        # Data types the field (or its elements) can be
        self.allowedtypes = allowedtypes
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

inttypes = [int, np.int64, np.int32]
floattypes = inttypes + [float, np.float64, np.float32]

# Record specification fields            
recfieldspecs = OrderedDict([('recordname', WFDBheaderspecs([str], '', None, True, None, None)),
                         ('nseg', WFDBheaderspecs(inttypes, '/', 'recordname', True, None, None)), 
                         ('nsig', WFDBheaderspecs(inttypes, ' ', 'recordname', True, None, None)),
                         ('fs', WFDBheaderspecs(floattypes, ' ', 'nsig', True, 250, None)),
                         ('counterfreq', WFDBheaderspecs(floattypes, '/', 'fs', False, None, None)),
                         ('basecounter', WFDBheaderspecs(floattypes, '(', 'counterfreq', False, None, None)),
                         ('siglen', WFDBheaderspecs(inttypes, ' ', 'fs', True, None, None)),
                         ('basetime', WFDBheaderspecs([str], ' ', 'siglen', False, None, '00:00:00')),
                         ('basedate', WFDBheaderspecs([str], ' ', 'basetime', False, None, None))])
 
# Signal specification fields.
sigfieldspecs = OrderedDict([('filename', WFDBheaderspecs([str], '', None, True, None, None)),
                         ('fmt', WFDBheaderspecs([str], ' ', 'filename', True, None, None)),
                         ('sampsperframe', WFDBheaderspecs(inttypes, 'x', 'fmt', False, None, None)),
                         ('skew', WFDBheaderspecs(inttypes, ':', 'fmt', False, None, None)),
                         ('byteoffset', WFDBheaderspecs(inttypes, '+', 'fmt', False, None, None)),
                         ('adcgain', WFDBheaderspecs(floattypes, ' ', 'fmt', True, 200., None)),
                         ('baseline', WFDBheaderspecs(inttypes, '(', 'adcgain', True, 0, None)),
                         ('units', WFDBheaderspecs([str], '/', 'adcgain', True, 'mV', None)),
                         ('adcres', WFDBheaderspecs(inttypes, ' ', 'adcgain', False, None, 0)),
                         ('adczero', WFDBheaderspecs(inttypes, ' ', 'adcres', False, None, 0)),
                         ('initvalue', WFDBheaderspecs(inttypes, ' ', 'adczero', False, None, None)),
                         ('checksum', WFDBheaderspecs(inttypes, ' ', 'initvalue', False, None, None)),
                         ('blocksize', WFDBheaderspecs(inttypes, ' ', 'checksum', False, None, 0)),
                         ('signame', WFDBheaderspecs([str], ' ', 'blocksize', False, None, None))])
    
# Segment specification fields.
segfieldspecs = OrderedDict([('segname', WFDBheaderspecs([str], '', None, True, None, None)),
                         ('seglen', WFDBheaderspecs(inttypes, ' ', 'segname', True, None, None))])



# For storing WFDB Signal definitions.

# SignalType class with all its parameters
class SignalType():
    def __init__(self, description, measurement=None, default_display=None, signalnames=None):
        self.description = description
        self.unitscale = unitscale
        # Tuple pair (a, b). The plot displays range a, of unit b.
        self.default_display = default_display
        self.signalnames = signalnames

unitscale = {
    'Voltage': ['pV', 'nV', 'uV', 'mV', 'V', 'kV'],
    'Temperature': ['C'],
    'Pressure': ['mmHg'],
}

# All signal types
signaltypes = {
    'BP': SignalType('Blood Pressure', 'Pressure'),
    'CO2': SignalType('Carbon Dioxide'),
    'CO': SignalType('Carbon Monoxide'),
    'ECG': SignalType('Electrocardiogram'),
    'EEG': SignalType('Electroencephalogram'),
    'EMG': SignalType('Electromyograph'),
    'EOG': SignalType('Electrooculograph'),
    'HR': SignalType('Heart Rate'),
    'MMG': SignalType('Magnetomyograph'),
    'O2': SignalType('Oxygen'),
    'PLETH': SignalType('Plethysmograph'),
    'RESP': SignalType('Respiration'),
    'SCG': SignalType('Seismocardiogram'),
    'STAT': SignalType('Status'), # small integers indicating status
    'ST': SignalType('ECG ST Segment'),
    'TEMP': SignalType('Temperature'),
}


