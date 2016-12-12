import numpy as np
import re
import os
import sys
import requests

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


# The information that can be contained in a WFDB header file. Separated into record line and signal lines. 
WFDBfields = [['recordname',
               'nseg',
               'nsig',
               'fs',
               'siglen',
               'basetime',
               'basedate'], 
              
              ['filename',
               'fmt',
               'sampsperframe',
               'skew',
               'byteoffset',
               'gain',
               'units',
               'baseline',
               'initvalue',
               'signame',
               'nsampseg',
               'comments']]

# The required explicit fields contained in all WFDB headers. 
req_read_fields = [['recordname', 'nsig'], ['filename', 'fmt']]

# The required input fields used to write WFDB headers. This python library requires mandatory fs and siglen. 
req_write_fields = [['recordname', 'nsig', 'fs', 'siglen'], ['filename', 'fmt']]


# Write a wfdb header file
def wrheader(inputfields, targetdir=os.cwd()):
       
    # Check that the input fields are valid
    checkheaderfields(inputfields)
    
    # Expand the signal specification fields of length 1
    finalfields = expand_signal_fields(inputfields)

    
    ### Wait, note that the writing can be separated by dat files....
    
    
    # Write the output file
    writeheaderfile(finalfields)
            
            
# Check that the dictionary of fields contains valid keys/values according to WFDB standards.
def checkheaderfields(inputfields):
    errors = []
    
    # Make sure the input field is a dictionary.
    if type(inputfields)!=dict:
        sys.exit("'fields' must be a dictionary")

    # Check for invalid dictionary keys. Cannot proceed with invalid keys. 
    for inputfield in inputfields:
        if inputfield not in WFDBfields[0]+WFDBfields[1]:
            sys.exit('Invalid input key: '+inputfield)

    # Check for mandatory input fields
    for req_field in req_write_fields[0]:
        if req_field not in list(inputfields):
            sys.exit('Missing required input key: '+req_field)
    if nsig>0:
        for req_field in req_write_fields[1]:
            if req_field not in list(inputfields):
                sys.exit('Missing required input key: '+req_field)
    
    # Check that signal specification fields have their dependent fields present
    checkdependentfields(inputfields)
    
    # Check for invalid dictionary values
    for field in inputfields:
        valueerrors=checkheadervalue(field, inputfields[field], nsig)
    if valueerrors:
        for ve in valueerrors:
            errors.append(ve)

    return errors
    
    
    
    
# Check the validity of each header value    
def checkheadervalue(field, value, nsig)
    
    errors=[]
    
    # These fields must have string values
    stringfields = []
    
    # Check the size of signal specification fields
    if field in WFDBfields[1]:
        if np.size(value)!=1 & np.size(value)!=nsig:
            error.append()
    
    return errors
    

# Expand length 1 signal specification fields into the length of the number of signals.   
def expandsignalfields(fields):
    
    nsig=fields['nsig']
    if nsig==1 or nsig==0:
        return fields
    
    for fd in fields:
        if fd in WFDBfields[1] & np.size(fields[fd])==1:
            if type(inputfields[infd])==list:   
                inputfields[infd] = nsig*inputfields[infd]
            else:
                inputfields[infd] = nsig*[inputfields[infd]]
   
    return fields
    
    
def writeheaderfile(fields):
    
    f=open(fields['recordname']+'.hea','w')
    
    
    f.close()
    
    
    
    

def rdheader(recordname):  # For reading signal headers

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
    