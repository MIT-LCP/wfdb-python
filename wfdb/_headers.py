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
                        signallines[ch]=signallines[ch] + recfieldspecs[field].delimiter + str(getattr(self, field)[ch])
            headerlines = headerlines + signallines

        # Create comment lines (if any)
        if 'comments' in writefields:
            commentlines = ['# '+comment for comment in self.comments]
            headerlines = headerlines + commentlines

        linestofile(self.recordname+'.hea', headerlines)
