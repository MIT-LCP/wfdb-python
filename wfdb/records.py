# Actions:
# For wrheader(), all fields must be already filled in and consistent with each other. The signals field will not be used.
# For wrsamp(), the field to use will be d_signals (which may be empty). 
# There will be a function to call on the object, with usesignals = digital or physical, which fills in the respective fields. setsignalfeatures( usesignals = 'digital' or 'physical'). 
# This is separate from another method 'setdefaults' which they will have to call too. 
# The checkfieldcohesion() function will be called in wrheader and check all the header fields, excluding the signals. 
# The checkfieldcohesion() function will be called in wrsamp after wrhea and before wrdat to check the signals against each field.

import numpy as np
import re
import os
import sys
from collections import OrderedDict
from calendar import monthrange


# The base WFDB class to extend to create WFDBrecord and WFDBmultirecord. Contains shared helper functions and fields.             
class WFDBbaserecord():
    # Constructor
    
    def __init__(self, recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter = None, 
                 siglen = None, basetime = None,basedate = None, 
                 comments = None)


# Class for single segment WFDB records.
class WFDBrecord(WFDBbaserecord, signals.Signals):
    
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
        
        super(self, recordname=recordname, nsig=nsig, 
              fs=fs, counterfreq=counterfreq, basecounter =basecounter,
              siglen = siglen, basetime = basetime, basedate = basedate, 
              comments = comments)
        
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
    def wrsamp(self, recordname):

        # Perform field validity and cohesion checks, and write the header file.
        self.wrheader(recordname, usesignals)

        # Perform signal validity and cohesion checks, and write the associated dat files.
        self.wrdats()

        # Perform field validity and cohesion checks, and write the header file
        self.wrheader(recordname)

        # Perform signal cohesiveness checks, and write all the dat files associated with the record
        self.wrdats()
            
    # Write a wfdb header file. The signals fields are not used. 
    def wrheader(self, recordname, usesignals=0):

         # Get all the fields used to write the header
        writefields = self.getwritefields(usesignals)

        # Check the validity of individual fields used to write the header 
        for f in writefields:
            self.checkfield(f) 
        
        # Check the compatibility of fields used to write the header
        self.checkfieldcohesion(usesignals)
        
        # Write the header file
        self.wrheaderfile(recordname, writefields)

    

        

        
            
            
            
            
            
            
            
            
            