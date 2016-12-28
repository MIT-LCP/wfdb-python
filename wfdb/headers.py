import numpy as np
import re
import os
import sys
import requests
from collections import OrderedDict

        
# The old confused fields ....####################
WFDBfields = {
        'recordname': [],
        'nseg': [],
        'nsig': [],
        'fs': [],
        'siglen': [],
        'basetime': [],
        'basedate': [],
    
        'filename': [],
        'fmt': [],
        'sampsperframe': [],
        'skew': [],
        'byteoffset': [],
        'gain': [],
        'units': [],
        'baseline': [],
        'initvalue': [],
        'signame': [],
        'nsampseg': [],
        'comments': []}
###############################################


# The fields permitted in the WFDBrecord class. Need to check this before writing records in case users add foreign fields
wfdbsinglefields = ['signals', 'recordname', 'nsig', 'fs', 'counterfreq', 'basecounter', 'siglen', 'basetime', 'basedate', 'filename', 'fmt', 'sampsperframe', 'skew', 'byteoffset', 'adcgain', 'baseline', 'units', 'adcres', 'adczero', 'initvalue', 'checksum', 'blocksize', 'signame', 'comments']

# A class for storing a single segment wfdb record.
class WFDBrecord():
    
    # Constructor
    def __init__(self, signals=None, recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter=None, 
                 siglen=None, basetime=None, basedate=None, 
                 filename=None, fmt=None, sampsperframe=None, 
                 skew=None, byteoffset=None, adcgain=None, 
                 baseline=None, units=None, adcres=None, 
                 adczero=None, initvalue=None, checksum=None, 
                 blocksize=None, signame=None, comments=None):
        
        # Note the lack of 'nseg' field. Single segment records cannot have this field. Even nseg = 1 makes 
        # the header a multi-segment header. 
        
        self.signals = signals
        
        self.recordname=recordname
        self.nsig=nsig
        self.fs=fs
        self.counterfreq=counterfreq
        self.basecounter=basecounter
        self.siglen=siglen
        self.basetime=basetime
        self.basedate=basedate
        
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
        
        self.comments=comments
        
    # Set default fields or infer them based on the 'signals' field. 
    def setfields(self):
        
        for field in self.fields:
            if WFDB[5] == 1 and signals!=None:
                setfield(field)
            elif WFDB[5] == 2
                setfield()
    
    # Check the validity of the fields
    def checkfields(self):
        
        # Check for foreign fields added by users.
        _checkforeignfields(self)
        
        # Get the list of required fields to check: default required fields, set fields, and their dependencies. Watch out for nseg. 
        fieldstocheck = getreqfields(self)
        
        # Check whether the fields' values are permitted, and if signals is included, valid. 
        _checkfieldvalues(self, fields)
        
        
        
        # Check whether essential fields have been set (including dependencies)
        
        # Check whether dependent fields
        
       
    
    
    
    
    # option 1: check set variables, get list of all required, check all required variables. Redundant!
    # Option 2: Get list of all required, check all required variables.  USE THIS!!! 
    
    
    # Check for and possibly remove foreign fields
    def _checkforeignfields(self):
        for f in vars(self)
            if f not in wfdbsinglefields:
                choice = []
                while choice not in ['y','n', 'a']:
                    choice = input("Foreign attribute '" + f + "' in WFDBrecord object not allowed.\n"
                                   "Remove attribute? [y = yes, n = no, a = remove all foreign fields]: ")
                if choice=='y':
                    delattr(self, f)
                elif choice=='n':
                    sys.exit('Exiting. WFDBrecord objects may only contain the following attributes:\n'+wfdbsinglefields)
                else:
                    _clearforeignfields(self)
    
    # Remove all the foreign user added fields in the object
    def _clearforeignfields(self):
        for f in vars(self):
            if f not in wfdbsinglefields:
                delattr(self, f)
    
    # Get the list of required fields needed to write the wfdb record
    def getreqfields(self):
        reqfields = []
        
        # The default required fields
        for f in wfdbsinglefields:
            if singlefieldspecs[f].write_req:
                reqfields.append(f)
        
        # The set fields and their dependencies 
        for f in vars(self):
            if f !=None:
                reqfields.append(f)
                if singlefieldspecs[f].dependency!=None:
                    reqfields.append()
        
        return reqfields
        
        
    # Check whether the the field values are valid
    def _checkfieldvalues(self, fields):
        # check the data types
        _checkfieldtypes(self, fields)
        
        
        
    
    
    # Check the data types of the specified fields
    def _checkfieldtypes(self, fields):
        for f in vars(self):
            if type(getattr(self, f)) not in allfieldspecs[f]:
                sys.exit('WFDBrecord field '+f+'must be one of types: '+singlefieldspecs[f])
                
    # Return a list of all the fields this class is allowed to have    
    def getallowedfields():
        return wfdbsinglefields
    
    
    
    
wfdbmultifields = ['segments', 'recordname', 'nseg', 'nsig', 'fs', 'counterfreq', 'basecounter', 'siglen', 'basetime', 'basedate', 'seglen', 'segname', 'comments']

# Multi segment WFDB record. Stores a list of individual WFDBrecord objects in the 'segment' field.
class WFDBmultirecord():
     def __init__(self, segments=None, recordname=None, nseg=None, nsig=None, 
                  fs=None, counterfreq=None, basecounter=None,
                  siglen=None, basetime=None, basedate=None,
                  seglen=None, segname=None, segment=None, 
                  comments=None):
        
        # Perhaps this should go in the checking functions below...
        if nseg==None or nseg<1:
            sys.exit("The WFDBmultirecord class is for multi-segment "
                     "records. The 'nseg' field must be specified."
                     "\nUse the 'WFDBrecord' class for creating "
                     "single segment records.")
    
        self.segments=sements  
    
        self.recordname=recordname
        self.nseg=nseg
        self.fs=fs
        self.counterfreq=counterfreq
        self.basecounter=basecounter
        self.siglen=siglen
        self.basetime=basetime
        self.basedate=basedate
        
        self.seglen=seglen
        self.segname=segname
        
        self.comments=comments
    
    # Return a list of all the fields this class is allowed to have    
    def getallowedfields():
        return wfdbmultifields
        


# The specifications of a WFDB field
# 0. type
# 1. delimiter
# 2. dependency
# 3. required to write? (nseg starts as F but will be set by wrmultiheader and wrmultiseg to True.)
# 4. default fill function available? 0-2. 
class WFDBfieldspecs():
    
    def __init__(self, speclist):
    
        # Data types the field can take
        self.allowedtypes = speclist[0]
        
        # The text delimiter that preceeds the field
        self.delimiter = speclist[1]
        
        # The required/dependent field which must also be present
        self.dependency = speclist[2]
        
        # Whether the field is mandatory for writing a header (WFDB requirements + extra rules enforced by this library).
        # Being required for writing is not the same as the user having to specify via wrsamp/wrhea. These functions can set defaults.
        self.write_req = speclist[3]
        
        # Whether there is a default value for this field that can be inferred or calculated, and also whether ther setfield function will actually set the field. 
        # 0 = no, 1 = yes but only if signals is present, 2 = yes without needing signals (although it may still use it).  
        # Watch out for: adcgain, baseline. Set them to 1 but have a check statement in the function. Should be 0 if isdigital.
        # sampsperframe, skew, and byteoffset do have defaults but the setfields will not return anything because we do not want to write anything. So value = 0. Conversely when reading, these fields will be left as None if not present. 
        # This field is used to see if we can call the 'setfield' function on that field.
        self.has_write_default = speclist[4]
         

# The signal field
signalspecs = OrderedDict([('signals', WFDBfield([[np.ndarray], '', None, False, 2]))])

# The segment field
segmentspecs = OrderedDict([('signals', WFDBfield([[np.ndarray], '', None, False, 2]))])

# Record specification fields            
recfieldspecs = OrderedDict([('recordname', WFDBfield([[str], '', None, True, 0])),
                         ('nseg', WFDBfield([[int], '/', 'recordname', False, 0])), 
                         ('nsig', WFDBfield([[int], ' ', 'recordname', True, 1])),
                         ('fs', WFDBfield([[int, float], ' ', 'nsig', True, 0])),
                         ('counterfreq', WFDBfield([[int, float], '/', 'fs', False, 0])),
                         ('basecounter', WFDBfield([[int, float], '(', 'counterfreq', False, 0])),
                         ('siglen', WFDBfield([[int], ' ', 'fs', True, 1])),
                         ('basetime', WFDBfield([[str], ' ', 'siglen', False, 2])),
                         ('basedate', WFDBfield([[str], ' ', 'basetime', False, 0]))])
# Signal specification fields 
sigfieldspecs = OrderedDict([('filename', WFDBfield([[str], '', None, True, 2])),
                         ('fmt', WFDBfield([[int, str], ' ', 'filename', True, 2])),
                         ('sampsperframe', WFDBfield([[int], 'x', 'fmt', False, 0])),
                         ('skew', WFDBfield([[int], ':', 'fmt', False, 0])),
                         ('byteoffset', WFDBfield([[int], '+', 'fmt', False, 0])),
                         ('adcgain', WFDBfield([[int], ' ', 'fmt', True, 1])),
                         ('baseline', WFDBfield([[int], '(', 'adcgain', True, 1])),
                         ('units', WFDBfield([[str], '/', 'adcgain', True, 2])),
                         ('adcres', WFDBfield([[int], ' ', 'adcgain', False, 2])),
                         ('adczero', WFDBfield([[int], ' ', 'adcres', False, 2])),
                         ('initvalue', WFDBfield([[int], ' ', 'adczero', False, 1])),
                         ('checksum', WFDBfield([[int], ' ', 'initvalue', False, 1])),
                         ('blocksize', WFDBfield([[int], ' ', 'checksum', False, 2])),
                         ('signame', WFDBfield([[str], ' ', 'blocksize', False, 1]))])
    
# Segment specification fields
segfieldspecs = OrderedDict([('segname', WFDBfield([[str], '', None, True, 0])),
                         ('seglen', WFDBfield([[int], ' ', 'segname', True, 0]))])
# Comment field
comfieldspecs = OrderedDict([('comments', WFDBfield([[int], '', None, False, False]))])


singlefieldspecs = mergeODlist([signalspecs.copy(), recfieldspecs.copy(), sigfieldspecs.copy(), comfieldspecs.copy()])
multifieldspecs = mergeODlist([segmentspecs.copy(), recfieldspecs.copy(), segfieldspecs.copy(), comfieldspecs.copy()])
multifieldspecs['nseg'].write_req = True



# The useful summary information contained in a wfdb record.
# Note: NOT a direct subset of WFDBrecord's fields. 
infofields = [['recordname',
               'nseg',
               'nsig',
               'fs',
               'siglen',
               'basetime',
               'basedate'],
              
              ['filename',
               'maxresolution',
               'sampsperframe',
               'units',
               'signame'],
              
              ['segname',
               'seglen'],
              
              'comments']

# Write a single segment wfdb header file.
# When setsigreqs = 1, missing signal specification dependencies will be set. When it is 0, they must be explicitly set.
# setinfo == 1 when wrheader is called from wrsamp. setinfo parameter should be 0 when wrheader is directly called. Should not be changeable. sig is set as the input signal when wrheader is called from wrsamp. 
# recinfo is a WFDBrecord object.
def wrheader(recinfo, targetdir=os.cwd(), sig=None):
    
    # Make sure user is not trying to write multi-segment headers. The nseg field is not allowed in regular headers. 
    if getattr(recinfo, 'nseg')!=None:
        sys.error("The 'nseg' field is not allowed in regular WFDB headers."
                  "\nFor writing multi-segment headers, use the 'wrmultiheader' function.")
        
    WFDBfieldlist = [recfields.copy(), sigfields.copy(), comfields.copy()]
    
    
    # Check the record info
    checkhrecinfo(recinfo, WFDBfieldlist, setinfo, ismultiseg)  
    
    
    # Check the header keys
    #keycheckedinfo = checkheaderkeys(recinfo, WFDBfieldlist, setinfo, 0)
    
    # Check the header values
    #valuecheckedinfo = checkheadervalues(keycheckedinfo, WFDBfields, setinfo, 0)

    
    
    
    
    
    # Write the output header file
    _writeheaderfile(finalfields)
    
                   
        
        
        
# Write a multi-segment wfdb header file. 
def wrmultiheader(recinfo, targetdir=os.cwd(), setinfo=0):
    
    # Make sure user is not trying to write single segment headers. 
    if getattr(recinfo, 'nseg')!=None:
        if type(getattr(recinfo, 'nseg'))!= int:
            
            sys.exit("The 'nseg' field must be an integer.")
            
        if getattr(recinfo, 'nseg')==0:
            
            sys.exit("The 'nseg' field is 0. You cannot write a multi-segment header with zero segments.")
            
        elif getattr(recinfo, 'nseg')==1:
            
            print("Warning: The 'nseg' field is 1. You are attempting to write a multi-segment header which encompasses only one segment.\nContinuing ...")
            
    else:
        sys.exit("Missing input field 'nseg' for writing multi-segment header.\nFor writing regular WFDB headers, use the 'wrheader' function.")
                  
    
    WFDBfieldlist = [recfields.copy(), segfields.copy(), comfields.copy()]
                  
    
    keycheckedfields = _checkheaderkeys(inputfields, WFDBfieldlist, setsigreqs, 1)
    
    # Check the header values
    valuecheckedfields = checkheadervalues(keycheckedfields, WFDBfields, setsigreqs, 0)
    
    # check that each signal component has the same fs and the correct number of signals. 
    
    # Write the header file
    

    
# Merge the ordered dictionaries in a list into one ordered dictionary. 
def _mergeODlist(ODlist):
    mergedOD=ODlist[0].copy()
    for od in ODlist[1:]:
        mergedOD.update(od)
    return mergedOD
                  

              
                  
                  
# Check the values of the record information according to WFDB standards 
# Will perform differently depending on whether it's called from wrheader or wrsamp. 
def checkrecinfo(recinfo, WFDBfieldlist, setinfos, ismultiseg):
    WFDBfields=_mergeODlist(WFDBfieldlist)
    
    # Make sure the input record info is a WFDBrecord obejct.
    if type(recinfo).__name__!='WFDBrecord':
        sys.exit("'recinfo' must be a WFDBrecord object")
    
    
    
    # First, get a list of all the fields that are required to write this header, which are then to be checked. This list is comprised of the always essential fields, the fields not equal to None set by the user, and their recurrent dependent fields. 
    required_info = []
    # Check the fields in reverse to capture all dependencies. 
    rev_rfields = OrderedDict(reversed(list(WFDBfieldlist[0].items())))
    rev_sfields = OrderedDict(reversed(list(WFDBfieldlist[1].items())))
    
    # 1. record fields
    for f in rev_rfields:
        # The user has set this field. Add it and its recurrent dependencies. 
        if getattr(recinfo, f)!=None:
            
            if f in required_info:
                continue
            
            required_info.append(f)
            
            depfield = rev_rfields[f].dependency
            
            #while rev_rfields[depfield].dependency!=None:
            while depfield!=None:
                required_info.append(depfield)
                depfield=rev_rfields[depfield].dependency
        
        # Default mandatory write fields
        if rev_rfields[f].write_req:
            if f in required_info:
                continue
            required_info.append(f)
            # No need to recursively add dependencies because dependencies of required fields are all required. 
        
        
    # Check the info fields. If specified, place the defaults in them before checking. 
    
    for i in required_info:
        if setinfo and rev_rfields[i].has_write_default:
            recinfo=add_default(recinfo, )
        
        checkinfovalue()
        
        
        
        
        
        
    # 2. signal/segment fields
    
    if ismultiseg:
        
    else:
        
          
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
                
                  
# Check that the dictionary of fields contains valid keys according to WFDB standards.
def checkheaderkeys(inputfields, WFDBfieldlist, setinfo, ismultiseg):
    
    WFDBfields=_mergeODlist(WFDBfieldlist): 
    
    # Make sure the input field is a dictionary.
    if type(inputfields)!=dict:
        sys.exit("'fields' must be a dictionary")

    # Check for invalid keys.
    for infield in inputfields:
        if infield not in WFDBfields:
            sys.exit('Invalid input key: '+infield)
        
    # Check for required keys and dependencies of entered keys    
    
    # Record specification fields
    for f in WFDBfieldlist[0]:
        if (WFDBfieldlist[0][f].write_req) and (inputfields.f ):
            sys.exit('Missing required input key: '+f)
            
    # Check value of nsig. 
    _checkval('nsig', inputfields['nsig'], WFDBfields)    
        
    # Check segment or signal specification keys
    if multiseg or inputfields['nsig']:
        for f in WFDBfieldlist[1]:
            if (WFDBfieldlist[1][f].is_req) and (f not in inputfields):
                sys.exit('Missing required input key: '+f)
    
    # Check that fields have their dependent field keys present
    keycheckedfields = _checkkeydeps(inputfields, setsigreqs, multiseg)
    
    return keycheckedfields



# Check that each header field key's dependent key is also present.
# Helper function to _checkheaderkeys
def _checkkeydeps(fields, setsigreqs, multiseg):
                  
   
    # The required preceding field for each header field. Must check dependencies in reverse order.
    # record line fields
    rec_dependencies = OrderedDict(reversed(list(WFDBfields[0].items())))
    # signal or segment line fields
    other_dependencies = OrderedDict(reversed(list(WFDBfields[1].items())))
    
    # Add dependent keys if option is specified.
    if setsigreqs:
        # Add empty keys for record line
        for d in rec_dependencies:
            if (d in fields) and (rec_dependencies[d].dependency not in fields):
                fields[rec_dependencies[d].dependency]=[]
        # Add empty keys for signal line. All segment line fields are mandatory.         
        if not multiseg:
            for d in other_dependencies:
                if (d in fields) and (other_dependencies[d].dependency not in fields):
                    fields[other_dependencies[d].dependency]=[]
    

    # Check for missing record line dependent fields.         
    for d in rec_dependencies:
        if (d in fields) and (rec_dependencies[d].write_req not in fields):
            sys.exit('Missing field dependency: '+rec_dependencies[d].write_req+' for field: '+d)
    # Check for missing signal line dependent fields. All segment line fields are mandatory.         
    if not multiseg:
        for d in other_dependencies:
            if (d in fields) and (sig_dependencies[d].write_req not in fields):
                sys.exit('Missing field dependency: '+other_dependencies[d].write_req+' for field: '+d)
            
    return fields
    
    
    
    
    
# Check the validity of each header value. Includes setting defaults. Remember that gain crap...     
def _checkheadervalues(keycheckedinfo, WFDBfieldlist, setsiginfo, multiseg):
    
    # At this point, all the required keys should be present. Now need to check their values. 
    
    WFDBfields=_mergeODlist(WFDBfieldlist):
    
    # Check the mandatory explicit fields
    for f in keycheckedfields:
        if WFDBfields[f].write_req:
            _checkval(f, keycheckedfields[f], WFDBfields)
                    
                
                
                
    # Fill missing values to defaults if option is specified.
    # Only fill fields where write_req==0. Users must explicitly set fields where write_req==1. No need to worry about those because
    # they would've thrown an error already. 
    if setsigreqs:
        
        rec_ = 
        sig
        
        
        for i in WFDBfields[0]:
            

            
            
            
            
            
            
            
        
    # Check the content of the fields
    nsig = fields['nsig'] 
    
    
    # MAKE SURE FILES names are consecutive and not interleaved. 
    
    
    # Check the size of signal specification fields.
    if field in WFDBfields[1]:
        if np.size(value)==1 
            # Expand 
        elif np.size(value)!=nsig:
            sys.exit("The ")
    
    
    return valuecheckedfields
    
    
    
    
# Check an individual header value. Helper subfunction to checkheadervalues. Does not set defaults. Other functions should have set defaults by this point if intended. Does not allow empty values.
def _checkval(key, value, WFDBfields):
        
    # Check the data type
    if type(value) not in WFDBfields[key].allowedtypes:
        sys.exit("fields["+key+"] must be one of the following types: "+WFDBfields[key].allowedtypes)
        
    # If numeric, check the allowed range
    
    if type(value) in [int, float]:
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
def _writeheaderfile(fields):
    
    f=open(fields['recordname']+'.hea','w')
    
    
    
    f.close()
    
    
    
    
    

    


    
    
# For reading WFDB header files
def rdheader(recordname): 

    # To do: Allow exponential input format for some fields

    # Output dictionary
    fields=WFDBfields
    
    # filename stores file names for both multi and single segment headers.
    # nsampseg is only for multi-segment


    # RECORD LINE fields (o means optional, delimiter is space or tab unless specified):
    # record name, nsegments (o, delim=/), nsignals, fs (o), counter freq (o, delim=/, needs fs),
    # base counter (o, delim=(), needs counter freq), siglen (o, needs fs), base time (o),
    # base date (o needs base time).

    # Regexp object for record line
    rxRECORD = re.compile(
        ''.join(
            [
                "(?P<name>[\w]+)/?(?P<nseg>\d*)[ \t]+",
                "(?P<nsig>\d+)[ \t]*",
                "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<basecounter>\d*\.?\d*)\)?[ \t]*",
                "(?P<siglen>\d*)[ \t]*",
                "(?P<basetime>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
                "(?P<basedate>\d{,2}/?\d{,2}/?\d{,4})"]))
    # Watch out for potential floats: fs (and also exponent notation...),
    # counterfs, basecounter

    # SIGNAL LINE fields (o means optional, delimiter is space or tab unless specified):
    # file name, format, samplesperframe(o, delim=x), skew(o, delim=:), byteoffset(o,delim=+),
    # ADCgain(o), baseline(o, delim=(), requires ADCgain), units(o, delim=/, requires baseline),
    # ADCres(o, requires ADCgain), ADCzero(o, requires ADCres), initialvalue(o, requires ADCzero),
    # checksum(o, requires initialvalue), blocksize(o, requires checksum),
    # signame(o, requires block)

    # Regexp object for signal lines. Consider flexible filenames, and also ~
    rxSIGNAL = re.compile(
        ''.join(
            [
                "(?P<filename>[\w]*\.?[\w]*~?)[ \t]+(?P<format>\d+)x?"
                "(?P<sampsperframe>\d*):?(?P<skew>\d*)\+?(?P<byteoffset>\d*)[ \t]*",
                "(?P<ADCgain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^/-]*)[ \t]*",
                "(?P<ADCres>\d*)[ \t]*(?P<ADCzero>-?\d*)[ \t]*(?P<initialvalue>-?\d*)[ \t]*",
                "(?P<checksum>-?\d*)[ \t]*(?P<blocksize>\d*)[ \t]*(?P<signame>[\S]*)"]))

    # Units characters: letters, numbers, /, ^, -,
    # Watch out for potentially negative fields: baseline, ADCzero, initialvalue, checksum,
    # Watch out for potential float: ADCgain.

    # Read the header file and get the comment and non-comment lines
    headerlines, commentlines = _getheaderlines(recordname)

    # Get record line parameters
    (_, nseg, nsig, fs, counterfs, basecounter, siglen,
    basetime, basedate) = rxRECORD.findall(headerlines[0])[0]

    # These fields are either mandatory or set to defaults.
    if not nseg:
        nseg = '1'
    if not fs:
        fs = '250'

    fields['nseg'] = int(nseg)
    fields['fs'] = float(fs)
    fields['nsig'] = int(nsig)
    fields['recordname'] = name

    # These fields might by empty
    if siglen:
        fields['siglen'] = int(siglen)
    fields['basetime'] = basetime
    fields['basedate'] = basedate


    # Signal or Segment line paramters
    # Multi segment header - Process segment spec lines in current master
    # header.
    if int(nseg) > 1:
        for i in range(0, int(nseg)):
            (filename, nsampseg) = re.findall(
                '(?P<filename>\w*~?)[ \t]+(?P<nsampseg>\d+)', headerlines[i + 1])[0]
            fields["filename"].append(filename)
            fields["nsampseg"].append(int(nsampseg))
    # Single segment header - Process signal spec lines in regular header.
    else:
        for i in range(0, int(nsig)):  # will not run if nsignals=0
            # get signal line parameters
            (filename,
            fmt,
            sampsperframe,
            skew,
            byteoffset,
            adcgain,
            baseline,
            units,
            adcres,
            adczero,
            initvalue,
            checksum,
            blocksize,
            signame) = rxSIGNAL.findall(headerlines[i + 1])[0]
            
            # Setting defaults
            if not sampsperframe:
                # Setting strings here so we can always convert strings case
                # below.
                sampsperframe = '1'
            if not skew:
                skew = '0'
            if not byteoffset:
                byteoffset = '0'
            if not adcgain:
                adcgain = '200'
            if not baseline:
                if not adczero:
                    baseline = '0'
                else:
                    baseline = adczero  # missing baseline actually takes adczero value if present
            if not units:
                units = 'mV'
            if not initvalue:
                initvalue = '0'
            if not signame:
                signame = "ch" + str(i + 1)
            if not initvalue:
                initvalue = '0'

            fields["filename"].append(filename)
            fields["fmt"].append(fmt)
            fields["sampsperframe"].append(int(sampsperframe))
            fields["skew"].append(int(skew))
            fields['byteoffset'].append(int(byteoffset))
            fields["gain"].append(float(adcgain))
            fields["baseline"].append(int(baseline))
            fields["units"].append(units)
            fields["initvalue"].append(int(initvalue))
            fields["signame"].append(signame)

    for comment in commentlines:
        fields["comments"].append(comment.strip('\s#'))

    return fields


# Read header file to get comment and non-comment lines
def _getheaderlines(recordname):
    with open(recordname + ".hea", 'r') as fp:
        headerlines = []  # Store record line followed by the signal lines if any
        commentlines = []  # Comments
        for line in fp:
            line = line.strip()
            if line.startswith('#'):  # comment line
                commentlines.append(line)
            elif line:  # Non-empty non-comment line = header line.
                ci = line.find('#')
                if ci > 0:
                    headerlines.append(line[:ci])  # header line
                    # comment on same line as header line
                    commentlines.append(line[ci:])
                else:
                    headerlines.append(line)
    return headerlines, commentlines


# Create a multi-segment header file
def wrmultisegheader():
    