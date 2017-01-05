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
#########################################            
    
    
# The base WFDB class to extend. Containings common helper functions and fields.             
class WFDBbaserecord():
    # Constructor
    
    def __init__(self, recordname=None, nsig=None, 
                 fs=None, counterfreq=None, basecounter = None, 
                 siglen = None, basetime = None,basedate = None, 
                 comments = None)

    # Get the list of required fields needed to write the wfdb record. Method for both single and multi-segment records. 
    # Returns the default required fields, the user defined fields, and their dependencies.
    def getreqfields(self, fieldspeclist):

        # Record specification fields
        reqfields=getreqsubset(self, reversed(list(fieldspeclist[1].items())))

        # Single-segment record
        if fieldspeclist[0] == signalspecs:

            checkfield(self, 'nsig')

            if self.nsig>0:
                # signals is not always required because this method is called via wrheader. 
                # Enforce the presence of signals at the start of rdsamp to ensure it gets added here. 
                if self.signals!=None:
                    reqfields.append('signals')

                reqfields=reqfields+getreqsubset(self, reversed(list(fieldspeclist[2].items())))

        # Multi-segment record
        else:
            # Segment specification fields and segments
            reqfields=reqfields+['segments', 'seglen', 'segname']

        # Comments
        if self.comments !=None:
            reqfields.append('comments')
        return reqfields


    # Helper function to getreqfields
    def getreqsubset(self, fieldspecs):
        reqfields=[]
        for f in fieldspecs:
            if f in reqfields:
                continue
            # If the field is default required or has been defined by the user...
            if f[1].write_req or self.f!=None:
                rf=f
                # Add the field and its recurrent dependencies
                while rf!=None:
                    reqfields.append(rf)
                    rf=fieldspeclist[1][rf].dependency

        return reqfields
            
    # Set the single field specified if possible. Most fields do not have an automatically set value. 
    # This function should only be called on fields == none. It should NOT be used to overwrite set fields. 
    def setfield(self, field):

        # if the field cannot be set, return without doing anything. 

        if field == 'nseg':
            checkfield(self, 'segments')
            self.nseg = len(self.nsegments)

        elif field == 'nsig':
            if self.signals == None:
                self.nsig=0
            else:
                checkfield(self, 'signals')
                self.nsig = np.shape(self.signals)[1]

        elif field == 'siglen':    


        elif field == 'basetime':  
            self.basetime = '00:00:00'

        elif field == 'filename':  
            checkfield(self, 'recordname')
            checkfield(nsig)
            self.filename = [self.recordname+'.dat']*nsig

        elif field == 'fmt':      

        elif field == 'adcgain':    
        elif field == 'baseline':  
        elif field == 'units':
            checkfield(nsig)
            print('Note: Setting units to nU or no units. If there are units please set them.')
            self.units = ['nU']*nsig

        elif field == 'adcres':  
        elif field == 'adczero':      
            self.adczero = [0]*nsig
        elif field == 'initvalue':
            checkfield(signals)

            if signals:
                self.initvalue = list(signals[0,:])

        elif field == 'checksum':  
            checkfield(signals)

        elif field == 'blocksize':
            self.blocksize = [0]*nsig

        # Not filling in channel names.  

        elif field == 'seglen':  
            self.seglen=[]
            for s in self.segments:
                self.seglen.append(np.shape(s)[])
      
    # Check the single field specified. 
    def checkfield(self, field):

        if field == 'signals':

        elif field == 'segments':


        elif field == 'recordname':
        elif field == 'nseg':
        elif field == 'nsig':
        elif field == 'fs':
        elif field == 'counterfreq':
        elif field == 'basecounter':
        elif field == 'siglen':    
        elif field == 'basetime':  
        elif field == 'basedate': 

        elif field == 'filename':  
        elif field == 'fmt':      
        elif field == 'sampsperframe':  
        elif field == 'skew':  
        elif field == 'byteoffset':  
        elif field == 'adcgain':    
        elif field == 'baseline':  
        elif field == 'units':  
        elif field == 'adcres':  
        elif field == 'adczero':      
        elif field == 'initvalue':  
        elif field == 'checksum':  
        elif field == 'blocksize':
        elif field == 'signame': 


        elif field == 'segname':      
        elif field == 'seglen':  

        elif field == 'comments':
            
            
    # Check for and possibly remove foreign fields
    def _checkforeignfields(self, allowedfields, objecttype):
        for f in vars(self)
            if f not in allowedfields:
                choice = []
                while choice not in ['y','n', 'a']:
                    choice = input("Foreign attribute '"+f+"' in "+objecttype+" object not allowed.\n"
                                   "Remove attribute? [y = yes, n = no, a = remove all foreign fields]: ")
                if choice=='y':
                    delattr(self, f)
                elif choice=='n':
                    sys.exit('Exiting. '+objecttype+' objects may only contain the following attributes:\n'+allowedfields)
                else:
                    _clearforeignfields(self, allowedfields)
    
    # Remove all the foreign user added fields in the object
    def _clearforeignfields(self, allowedfields):
        for f in vars(self):
            if f not in allowedfields:
                delattr(self, f)         
            
            
            
# Class representing a single segment WFDB record.
class WFDBrecord(WFDBbaserecord):
    
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
        
        super(self, recordname=recordname, nsig=nsig, 
              fs=fs, counterfreq=counterfreq, basecounter =basecounter,
              siglen = siglen, basetime = basetime, basedate = basedate, 
              comments = comments)
        
        self.signals = signals
        
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
                                 
        
    # Set missing fields that a default, or can be inferred.
    # This method does NOT overwrite values. 
    # If a field requires another field which is invalid or not set, just return. 
    def setfields(self, fieldstoset = getreqfields(self, singlefieldspeclist.copy())):
                                 
        # Set the missing required fields if possible. 
        for f in fieldstoset:
            # Do not overwrite fields set by the user
            if f == None:
                setfield(self, f)
    
    
    # Check the specified fields of the WFDBrecord object for validity. 
    def checkfields(self ,fieldstocheck = getreqfields(self), objecttype='WFDBrecord'):
        
        fielderrors=[]
        
        # Check for foreign fields
        _checkforeignfields(self, allowedfields = self.getallowedfields())
        
        # Check whether the fields' values are valid. 
        _checkfieldvalues(self, fieldstocheck)
        
        
        
        
        
        
        
        
        
        
                
    # Return a list of all the fields this class is allowed to have    
    def getallowedfields():
        return list(mergeODlist(singlefieldspeclist).keys())
    
    


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
    
        super(self, recordname=recordname, nsig=nsig, 
              fs=fs, counterfreq=counterfreq, basecounter =basecounter,
              siglen = siglen, basetime = basetime, basedate = basedate, 
              comments = comments)
    
        self.segments=sements  
        self.nseg=nseg
        
        self.seglen=seglen
        self.segname=segname
    
    # Return a list of all the fields this class is allowed to have    
    def getallowedfields():
        return list(mergeODlist(multifieldspeclist).keys())
        


# The specifications of a WFDB field
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
signalspecs = OrderedDict([('signal', WFDBfield([[np.ndarray], '', None, False, 2]))])

# The segment field. A list of WFDBrecord objects
segmentspecs = OrderedDict([('segment', WFDBfield([[list], '', None, True, 2]))])

# Record specification fields            
recfieldspecs = OrderedDict([('recordname', WFDBfield([[str], '', None, True, 0])),
                         ('nseg', WFDBfield([[int], '/', 'recordname', True, 0])), # Essential for multi but not present in single.
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


singlefieldspeclist = [signalspecs.copy(), recfieldspecs.copy(), sigfieldspecs.copy(), comfieldspecs.copy()]
del(singlefieldspeclist[1]['nseg']
multifieldspeclist = [segmentspecs.copy(), recfieldspecs.copy(), segfieldspecs.copy(), comfieldspecs.copy()]




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
              
              ['comments']]

# Write a single segment wfdb header file.
# record is a WFDBrecord object
def wrheader(record, targetdir=os.cwd()):
    
    # The fields required to write this header
    requiredfields = record.getreqfields()
    
    # Fill in any missing info possible 
    # for the set of required fields
    record.setfields(requiredfields)
    
    # Check every field to be used
    record.checkfields(requiredfields)  
    
    # Write the output header file
    writeheaderfile(record)
    
    # The reason why this is more complicated than the ML toolbox's rdsamp:
    # That one only accepts a few fields, and sets defaults for the rest. 
    # This one accepts any combination of fields and tries to set what it can. Also does checking. 
        
        
        
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
    