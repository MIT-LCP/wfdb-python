import numpy as np
import re
import os
import sys
from collections import OrderedDict
from calendar import monthrange
from . import _signals

# Class of common methods for single and multi-segment headers
class BaseHeadersMixin():

    # Write a wfdb header file. The signals fields are not used. 
    def wrheader(self):

        # Get all the fields used to write the header
        writefields = self.getwritefields()

        # Check the validity of individual fields used to write the header 
        for f in writefields:
            self.checkfield(f) 
        
        # Check the cohesion of fields used to write the header
        self.checkfieldcohesion(writefields)
        
        # Write the header file using the specified fields
        self.wrheaderfile(writefields)

    # Set defaults for fields needed to write the header if they have defaults.
    # Returns a list of fields set by this function.
    # Set overwrite == 1 to enable overwriting of populated fields.
    def setdefaults(self, overwrite = 0):
        setfields = [] # The fields set
        for field in self.getwritefields():
            if self.setdefault(field) == 1:
                setfields.append(field)
        return setfields

    # Helper function for getwritefields
    def getwritesubset(self, fieldspecs):
        writefields=[]
        for f in fieldspecs:
            if f in writefields:
                continue
            # If the field is required by default or has been defined by the user
            if fieldspecs[f].write_req or getattr(self, f)!=None:
                rf=f
                # Add the field and its recursive dependencies
                while rf!=None:
                    writefields.append(rf)
                    rf=fieldspecs[rf].dependency

        return writefields


# Class with single-segment header methods
# To be inherited by WFDBrecord from records.py.
class HeadersMixin(HeadersBaseMixin):

    # Set a field to its default value if there is a default. Returns 1 if the field is set. 
    # Set overwrite == 1 to enable overwriting of populated fields.
    # In the future, this function and getdefault (to be written) may share a new dictionary/object field.  
    def setdefault(self, field, overwrite = 0):
        
        # Check whether the field is empty before trying to set.
        if overwrite == 0:
            if getattr(self, field) != None:
                return 0
                
        # Going to set the field. 
        # Record specification fields
        if field == 'basetime':
            self.basetime = '00:00:00'
        # Signal specification fields    
        elif field == 'filename':
            self.filename = self.nsig*[self.recordname+'.dat']
        elif field == 'adcres':
            if self.fmt is None:
                self.adcres = self.nsig*[0]
            else:
                self.adcres=_signals.wfdbfmtres(self.fmt)
        elif field == 'adczero':
            self.adczero = self.nsig*[0]
        elif field == 'blocksize':
            self.blocksize = self.nsig*[0]
        # Most fields have no default. 
        else:
            return 0
        return 1

    # Get the list of fields used to write the header. (Does NOT include d_signals.)
    # Returns the default required fields, the user defined fields, and their dependencies.
    def getwritefields(self):

        # Record specification fields
        writefields=self.getwritesubset(OrderedDict(reversed(list(recfieldspecs.items()))))
        writefields.remove('nseg')

        # Determine whether there are signals. If so, get their required fields.
        self.checkfield('nsig')
        if self.nsig>0:
            writefields=writefields+self.getwritesubset(OrderedDict(reversed(list(sigfieldspecs.items()))))

        # Comments
        if self.comments != None:
            writefields.append('comments')
        return writefields

    

    # Check the cohesion of fields used to write the header
    def checkfieldcohesion(self, writefields):

        # If there are no signal specification fields, there is nothing to check. 
        if self.nsig>0:
            # The length of all signal specification fields must match nsig
            for f in writefields:
                if f in sigfieldspecs:
                    if len(getattr(self, f)) != self.nsig:
                        sys.exit('The length of field: '+f+' does not match field nsig.')

            # Each filename must correspond to only one fmt, and only one byte offset (if defined). 
            datfmts = {}
            datoffsets = {}
            for ch in range(0, self.nsig):
                if self.filename[ch] not in datfmts:
                    datfmts[self.filename[ch]] = self.fmt[ch]
                else:
                    if datfmts[self.filename[ch]] != self.fmt[ch]:
                        sys.exit('Each filename (dat file) specified must have the same fmt')
                
                if self.byteoffset != None:
                    if self.byteoffset[ch] not in datoffsets:
                        print('self.byteoffset[ch]:', self.byteoffset[ch])
                        print('datoffsets: ', datoffsets)
                        datoffsets[self.byteoffset[ch]] = self.byteoffset[ch]
                    else:
                        if datoffsets[self.filename[ch]] != self.byteoffset[ch]:
                            sys.exit('Each filename (dat file) specified must have the same byte offset')


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

        # Create signal specification lines (if any)
        if self.nsig>0:
            signallines = self.nsig*['']
            # Traverse the ordered dictionary
            for field in sigfieldspecs:
                # If the field is being used, add each of its elements with the delimiter to the appropriate line 
                if field in writefields:
                    for ch in range(0, self.nsig):
                        signallines[ch]=signallines[ch] + sigfieldspecs[field].delimiter + str(getattr(self, field)[ch])
                # The 'baseline' field needs to be closed with ')'
                if field== 'baseline':
                    for ch in range(0, self.nsig):
                        signallines[ch]=signallines[ch] +')'

            headerlines = headerlines + signallines

        # Create comment lines (if any)
        if 'comments' in writefields:
            commentlines = ['# '+comment for comment in self.comments]
            headerlines = headerlines + commentlines

        linestofile(self.recordname+'.hea', headerlines)



# Class with multi-segment header methods
# To be inherited by WFDBmultirecord from records.py.
class MultiHeadersMixin(BaseHeadersMixin):

    
    # Get the list of fields used to write the multi-segment header. 
    # Returns the default required fields, the user defined fields, and their dependencies.
    def getwritefields(self):

        # Record specification fields
        writefields=self.getwritesubset(OrderedDict(reversed(list(recfieldspecs.items()))))

        # Segment specification fields are all mandatory
        writefields = writefields + ['segname', 'seglen']

        # Comments
        if self.comments !=None:
            writefields.append('comments')
        return writefields


    # Set a field to its default value if there is a default. Returns 1 if the field is set. 
    # Set overwrite == 1 to enable overwriting of populated fields.
    # In the future, this function and getdefault (to be written) may share a new dictionary/object field.  
    def setdefault(self, field, overwrite = 0):
        
        # Check whether the field is empty before trying to set.
        if overwrite == 0:
            if getattr(self, field) != None:
                return 0
                
        # Going to set the field. 
        # Record specification fields
        if field == 'basetime':
            self.basetime = '00:00:00'
            return 1
        else:
            return 0


    # Check the cohesion of fields used to write the header
    def checkfieldcohesion(self, writefields):

        # The length of segname and seglen must match nseg
        for f in ['segname', 'seglen']:
            if len(getattr(self, f)) != self.nseg:
                sys.exit('The length of field: '+f+' does not match field nseg.')

        # Check the sum of the 'seglen' fields against 'siglen'
        if np.sum(self.seglen) != self.siglen:
            sys.exit("The sum of the 'seglen' fields do not match the 'siglen' field")


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



# Specifications of WFDB header fields.
class WFDBheaderspecs():
    
    def __init__(self, speclist):
        # Data types the field (or its elements) can be
        self.allowedtypes = speclist[0]
        # The text delimiter that preceeds the field if it is a field that gets written to header files.
        self.delimiter = speclist[1]
        # The required/dependent field which must also be present
        self.dependency = speclist[2]
        # Whether the field is always required for writing a header (WFDB requirements + extra rules enforced by this library).
        self.write_req = speclist[3]


# Record specification fields  
# Note: nseg is essential for multi but not a field in single. 
# getwritefields defined in _headers.Headers_Mixin will remove it.           
recfieldspecs = OrderedDict([('recordname', WFDBheaderspecs([[str], '', None, True])),
                         ('nseg', WFDBheaderspecs([[int, np.int64, np.int32], '/', 'recordname', True])), 
                         ('nsig', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'recordname', True])),
                         ('fs', WFDBheaderspecs([[int, float, np.float64, np.float32], ' ', 'nsig', True])),
                         ('counterfreq', WFDBheaderspecs([[int, np.int64, np.int32, float, np.float64, np.float32], '/', 'fs', False])),
                         ('basecounter', WFDBheaderspecs([[int, np.int64, np.int32, float, np.float64, np.float32], '(', 'counterfreq', False])),
                         ('siglen', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'fs', True])),
                         ('basetime', WFDBheaderspecs([[str], ' ', 'siglen', False])),
                         ('basedate', WFDBheaderspecs([[str], ' ', 'basetime', False]))])
 
# Signal specification fields. Type will be list.
sigfieldspecs = OrderedDict([('filename', WFDBheaderspecs([[str], '', None, True])),
                         ('fmt', WFDBheaderspecs([[str], ' ', 'filename', True])),
                         ('sampsperframe', WFDBheaderspecs([[int, np.int64, np.int32], 'x', 'fmt', False])),
                         ('skew', WFDBheaderspecs([[int, np.int64, np.int32], ':', 'fmt', False])),
                         ('byteoffset', WFDBheaderspecs([[int, np.int64, np.int32], '+', 'fmt', False])),
                         ('adcgain', WFDBheaderspecs([[int, np.int64, np.int32, float, np.float64, np.float32], ' ', 'fmt', True])),
                         ('baseline', WFDBheaderspecs([[int, np.int64, np.int32], '(', 'adcgain', True])),
                         ('units', WFDBheaderspecs([[str], '/', 'adcgain', True])),
                         ('adcres', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'adcgain', False])),
                         ('adczero', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'adcres', False])),
                         ('initvalue', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'adczero', False])),
                         ('checksum', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'initvalue', False])),
                         ('blocksize', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'checksum', False])),
                         ('signame', WFDBheaderspecs([[str], ' ', 'blocksize', False]))])
    
# Segment specification fields. Type will be list. 
segfieldspecs = OrderedDict([('segname', WFDBheaderspecs([[str], '', None, True])),
                         ('seglen', WFDBheaderspecs([[int, np.int64, np.int32], ' ', 'segname', True]))])





# Write each line in a list of strings to a text file
def linestofile(filename, lines):
    f = open(filename,'w')
    for l in lines:
        f.write("%s\n" % l)
    f.close()              
