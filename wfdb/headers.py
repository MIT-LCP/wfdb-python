import numpy as np
import re
import os
import sys
import requests
from collections import OrderedDict

        
# The information contained in a WFDB header file
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

# The frisking will not be done here, but rather in wrsamp and wrheader. 
# Single segment wfdb record.
class WFDBrecord():
    
    def __init__(self, recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter=None, 
                 siglen=None, basetime=None, basedate=None, 
                 filename=None, fmt=None, sampsperframe=None, 
                 skew=None, byteoffset=None, adcgain=None, 
                 baseline=None, units=None, adcres=None, 
                 adczero=None, initvalue=None, checksum=None, 
                 blocksize=None, signame=None, comments=None):
        
        if nseg>1:
            sys.exit("The WFDBrecord class is for single segment "
                     "records. Use the 'WFDBmultirecord' class for "
                     "multi-segment records.")
        
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
        
# Multi segment WFDB record. Stores a list of individual WFDBrecord objects in the 'segment' field.
class WFDBmultirecord():
     def __init__(self, recordname=None, nseg=None, nsig=None, 
                  fs=None, counterfreq=None, basecounter=None,
                  siglen=None, basetime=None, basedate=None,
                  seglen=None, segname=None, segment=None, 
                  comments=None):
        
        if nseg<2:
            sys.exit("The WFDBmultirecord class is for multi-segment "
                     "records. Use the 'WFDBrecord' class for "
                     "single segment records.")
    
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
        self.segment=sement
        
        self.comments=comments
        
        
# item:
# 0. type
# 1. delimiter
# 2. dependency
# 3. required to write? (nseg starts as F but will be set by wrmultiheader and wrmultiseg to True.)
# 4. default fill function
# A field of the WFDB header file
class WFDBfield():
    
    def __init__(self, speclist):
    
        # Data types the field can take
        self.allowedtypes = speclist[0]
        
        # The text delimiter that preceeds the field
        self.delimiter = speclist[1]
        
        # The required/dependent field which must also be present
        self.dependency = speclist[2]
        
        # Whether the field is mandatory for writing (WFDB requirements + extra rules enforced by this library).
        # Being required for writing is not the same as the user having to specify via wrsamp. wrsamp sets defaults (wrheader does not).  
        self.write_req = speclist[3]
        
        # The default function to call to obtain the default. Only called via wrsamp, not wrhea.
        # If 'None', it means there is nothing to call, not that 'None' is is the default value of the field. 
        
        # Or perhaps there should only be one function to call with if statements. Kind of inappropriate to 
        # define a bunch of lambda functions with various input requirements. Just redefine the values once for 
        # setdefault. In this case, this field will just be true or false whether there is a potential default 
        # from wrsamp (there is none for wrhea)
        # Watch out for: adcgain, baseline, units. Set them to true but have a check statement in the function. 
        self.has_write_default = speclist[4]
        
# write_req = True if called from multi. It is also NOT ALLOWED in regular wfdb headers. It is to be omitted.         
recfields = OrderedDict([('recordname', WFDBfield([[str], '', None, True, False])),
                         ('nseg', WFDBfield([[int], '/', 'recordname', False, False])), 
                         ('nsig', WFDBfield([[int], ' ', 'recordname', True, True])),
                         ('fs', WFDBfield([[int, float], ' ', 'nsig', True, False])),
                         ('counterfreq', WFDBfield([[int, float], '/', 'fs', False, False])),
                         ('basecounter', WFDBfield([[int, float], '(', 'counterfreq', False, False])),
                         ('siglen', WFDBfield([[int], ' ', 'fs', True, True])),
                         ('basetime', WFDBfield([[str], ' ', 'siglen', False, True])),
                         ('basedate', WFDBfield([[str], ' ', 'basetime', False, False]))])

sigfields = OrderedDict([('filename', WFDBfield([[str], '', None, True, True])),
                         ('fmt', WFDBfield([[int, str], ' ', 'filename', True, True])),
                         ('sampsperframe', WFDBfield([[int], 'x', 'fmt', False, False])),
                         ('skew', WFDBfield([[int], ':', 'fmt', False, False])),
                         ('byteoffset', WFDBfield([[int], '+', 'fmt', False, False])),
                         ('adcgain', WFDBfield([[int], ' ', 'fmt', True, True])),
                         ('baseline', WFDBfield([[int], '(', 'adcgain', True, True])),
                         ('units', WFDBfield([[str], '/', 'adcgain', True, True])),
                         ('adcres', WFDBfield([[int], ' ', 'adcgain', False, True])),
                         ('adczero', WFDBfield([[int], ' ', 'adcres', False, True])),
                         ('initvalue', WFDBfield([[int], ' ', 'adczero', False, True])),
                         ('checksum', WFDBfield([[int], ' ', 'initvalue', False, True])),
                         ('blocksize', WFDBfield([[int], ' ', 'checksum', False, True])),
                         ('signame', WFDBfield([[str], ' ', 'blocksize', False, True]))])
    

segfields = OrderedDict([('segname', WFDBfield([[str], '', None, True, False])),
                         ('seglen', WFDBfield([[int], ' ', 'segname', True, False]))])

comfields = OrderedDict([('comments', WFDBfield([[int], '', None, False, False]))])

# The useful summary information contained in a wfdb record.
# Note: Not a direct subset of WFDBfields. 
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
    