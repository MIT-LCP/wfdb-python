import numpy as np
import re
import os
import sys
from collections import OrderedDict
from calendar import monthrange


# Class with header methods
# To be inherited by WFDBrecord from records.py. (May split this into multi and single header)
class HeadersMixin():

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


    # Get the list of fields used to write the header. (Does NOT include d_signals.)
    # Returns the default required fields, the user defined fields, and their dependencies.
    def getwritefields(self):

        # Record specification fields
        writefields=self.getreqsubset(sreversed(list(recfieldspecs.items())))
        writefields.remove('nseg')

        # Determine whether there are signals. If so, get their required fields.
        self.checkfield('nsig')
        if self.nsig>0:
            writefields=writefields+self.getwritesubset(reversed(list(fieldspeclist[2].items())))

        # Comments
        if self.comments !=None:
            writefields.append('comments')
        return writefields

    # Helper function for getwritefields
    def getwritesubset(self, fieldspecs):
        writefields=[]
        for f in fieldspecs:
            if f in writefields:
                continue
            # If the field is required by default or has been defined by the user
            if f[1].write_req or self.f!=None:
                rf=f
                # Add the field and its recursive dependencies
                while rf!=None:
                    writefields.append(rf)
                    rf=fieldspecs[rf].dependency

        return writefields


    # Check the cohesion of fields used to write the header
    def checkfieldcohesion(self, writefields):

        # If there are no signal specification fields, there is nothing to check. 
        if nsig>0:
            # The length of all signal specification fields must match nsig
            for f in writefields:
                if f in sigfieldspecs:
                    if len(getattr(self, f)) != self.nsig:
                        sys.exit('The length of field: '+f+' does not match field nsig.')

            # Each filename must correspond to only one fmt, and only one blocksize if defined. 
            datfmts = {}
            datoffsets = {}
            for ch in range(0, self.nsig):
                if self.filename[ch] not in datfmts:
                    datfmts[self.filename[ch]] = self.fmt[ch]
                else:
                    if datfmts[self.filename[ch]] != self.fmt[ch]:
                        sys.exit('Each filename (dat file) specified must have the same fmt')
                    
                if self.filename[ch] not in datoffsets:
                    datoffsets[self.filename[ch]] = self.byteoffset[ch]
                else:
                    if datoffsets[self.filename[ch]] != self.byteoffset[ch]:
                        sys.exit('Each filename (dat file) specified must have the same byte offset')

    


    # Set defaults for fields needed to write the header if they have defaults.
    # Returns a list of fields set by this function.
    # Set overwrite == 1 to enable overwriting of populated fields.
    def setdefaults(self, overwrite = 0):
        setfields = [] # The fields set
        for field in self.getwritefields():
            if setfields.append(self.setdefault(field)):
                setfields.append(field)
        return setfields

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
            self.adcres=wfdbfmtres(self.fmt)
        elif field == 'adczero':
            self.adczero = self.nsig*[0]
        elif field == 'blocksize':
            self.blocksize = self.nsig*[0]
        # Most fields have no default. 
        else:
            return 0
        return 1

    # Write a header file using the specified fields
    def writeheaderfile(self, writefields):

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
                    for ch in range(0, nsig):
                        signallines[ch]=signallines[ch] + recfieldspecs[field].delimiter + str(getattr(self, field)[ch])
            headerlines = headerlines + signallines

        # Create comment lines (if any)
        if 'comments' in writefields
            commentlines = ['# '+comment for comment in self.comments]
            headerlines = headerlines + commentlines

        linestofile(self.recordname+'.hea', headerlines)
