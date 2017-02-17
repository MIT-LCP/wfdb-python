import numpy as np
import pandas as pd
from . import headers

# Class for WFDB annotations
class Annotation():

    def __init__(annsamp, anntype, num = None, subtype = None, chan = None, aux = None, fs = None):
        self.annsamp = annsamp
        self.anntype = anntype
        self.num = num
        self.subtype = subtype
        self.chan = chan
        self.aux = aux
        self.fs = fs

    # Write an annotation file
    def wrann(self):
        # Check the validity of individual fields used to write the annotation file
        self.checkfields() 

        # Check the cohesion of fields used to write the annotation
        self.checkfieldcohesion(writefields)
        
        # Write the header file using the specified fields
        self.wrannfile()

    # Check the mandatory and set fields of the annotation object
    # Return indices of anntype field which are not encoded, and thus need
    # to be moved to the aux field.
    def checkfields(self):
        # Enforce mandatory write fields
        for field in ['annsamp', 'anntype']:
            if not getattr(self, field):
                print('The ', field, ' field is mandatory for writing annotation files')
                sys.exit()

        # Check all set fields
        for field in annfields:
            if field is not None:
                # Check the type of the field's elements
                self.checkfield(field)

    # Check a particular annotation field
    def checkfield(self, field):

        if field == 'fs':
            # Check the field type
            if type(self.fs) not in annfieldtypes['fs']:
                print('The fs field must be one of the following types: ', annfieldtypes['fs'])
                sys.exit()
            # Field specific check
            if self.fs <=0:
                sys.exit('The fs field must be a non-negative number')

        else:
            fielditem = getattr(self, field)

            # Ensure the field item is a list or 1d numpy array
            if type(fielditem) not in [list, np.ndarray]:
                print('The ', field, ' field must be a list or a 1d numpy array')
                sys.exit()          
            if type(fielditem) == np.ndarray and fielditem.ndim != 1:
                print('The ', field, ' field must be a list or a 1d numpy array')
                sys.exit()

            # Check the data types of the elements
            for item in fielditem:
                if type(item) not in annfieldtypes[field]+[None]:
                    print('All elements of the ' field, 'field must be None or one of the following types: ', annfieldtypes[field])
                    sys.exit()

            # Field specific checks
            if field == 'annsamp':
                sampdiffs = np.diff(self.annsamp)
                if min(self.annsamp) < 0 :
                    sys.exit('The annsamp field must only contain non-negative integers')
                if min(sampdiffs) < 0 :
                    sys.exit('The annsamp field must contain monotonically increasing sample numbers')
            elif field == 'anntype':
                # Ensure all fields lie in standard WFDB annotation codes
                if set(self.anntype) - set(annsyms.values()) != set():
                    print('The anntype field contains items not encoded in the WFDB annotation library.')
                    print('To see the valid annotation codes, call: showanncodes()')
                    print('To transfer non-encoded anntype items into the aux field, call: self.type2aux')
                    sys.exit()
            elif field == 'num':
                if min(self.num) < 0 :
                    sys.exit('The num field must only contain non-negative integers')
            elif field == 'subtype':
                if min(self.subtype) < 0 :
                    sys.exit('The subtype field must only contain non-negative integers')
            elif field == 'chan':
                if min(self.chan) < 0 :
                    sys.exit('The chan field must only contain non-negative integers')
            elif field == 'aux':
                if min(self.aux) < 0 :
                    sys.exit('The aux field must only contain non-negative integers')


    # Ensure all set annotation fields have the same length
    def checkfieldcohesion(self):
        # Number of annotation samples
        nannots = len(getattr(self, annsamp))

        for field in annfields[1:-1]:
            if getarr(self, field) is not None:
                if len(getattr(self, field)) != nannots:
                    sys.exit('All set annotation fields (aside from fs) must have the same length')


    def wrannfile(self):
        print('on it')

    # Move non-encoded anntype elements into the aux field 
    def type2aux(self):

        external_anntypes = set(self.anntype) - set(annsyms.values())

        # Nothing to do
        if external_anntypes == set():
            return

        # There is at least one external value.

        # Initialize aux field if necessary 
        if self.aux == None:
            self.aux = [None]*len(self.annsamp)

        # Move the anntype fields
        for ext in external_anntypes:
            for i in np.where(self.anntype == ext):
                if self.aux[i] == None:
                    self.aux[i] = self.anntype[i]
                    self.anntype[i] = ''
                else:
                    self.aux[i] = self.anntype[i]+' '+self.aux[i]
                    self.anntype[i] = ''


def showanncodes():
    symcodes


## ------------- Reading Annotations ------------- ##

def rdann(recordname, annot, sampfrom=0, sampto=None, anndisp=1):
    """ Read a WFDB annotation file recordname.annot and return the fields as lists or arrays

    Usage: annotation = rdann(recordname, annot, sampfrom=0, sampto=[], anndisp=1)

    Input arguments:
    - recordname (required): The record name of the WFDB annotation file. ie. for 
      file '100.atr', recordname='100'
    - annot (required): The annotator extension of the annotation file. ie. for 
      file '100.atr', annot='atr'
    - sampfrom (default=0): The minimum sample number for annotations to be returned.
    - sampto (default=None): The maximum sample number for 
      annotations to be returned.
    - anndisp (default = 1): The annotation display flag that controls the data type 
      of the 'anntype' output parameter. 'anntype' will either be an integer key(0), 
      a shorthand display symbol(1), or a longer annotation code(2).

    Output argument:
    - annotation: The annotation object with the following fields:
        - annsamp: The annotation location in samples relative to the beginning of the record.
        - anntype: The annotation type according the the standard WFDB keys.
        - subtype: The marked class/category of the annotation.
        - chan: The signal channel associated with the annotations.
        - num: The labelled annotation number. 
        - aux: The auxiliary information string for the annotation.
        - fs: The sampling frequency written into the annotation file if present.

    *NOTE: Every annotation sample contains the 'annsamp' and 'anntype' fields. All 
           other fields default to 0 or empty if not present.
    """

    if sampto and sampto <= sampfrom:
        raise ValueError("sampto must be greater than sampfrom")
    if sampfrom < 0:
        raise ValueError("sampfrom must be a non-negative integer")

    # Read the file in byte pairs
    filebytes = loadbytepairs(recordname, annot)

    # The maximum number of annotations potentially contained
    annotlength = filebytes.shape[0]

    # Initialise arrays to the total potential number of annotations in the file
    annsamp, anntype, subtype, chan, num, aux = init_arrays(annotlength)

    # Indexing Variables
    ts = 0 # Total number of samples from beginning of record. Annotation bytes only store dt.
    ai = 0 # Annotation index, the number of annotations processed.

    # Check the beginning of the file for a potential fs field
    fs, bpi = get_fs(filebytes, bpi)

    # Process annotations. Iterate across byte pairs. 
    # Sequence for one ann is: 
    #   SKIP pair (if any)
    #   samp + anntype pair
    #   other pairs (if any)
    # The last byte pair is 0 indicating eof.
    while (bpi < annotlength - 1):

        # The first byte pair will either store the actual samples + anntype,
        # or 0 + SKIP.
        AT = filebytes[bpi, 1] >> 2  # anntype

        # flags that specify whether to copy the previous channel/num value for
        # the current annotation. Set default. 
        cpychan, cpynum = 1, 1
        ts, annsamp, anntype, bpi = copy_prev(AT,ts,filebytes,bpi,annsamp,anntype,ai)

        AT = filebytes[bpi, 1] >> 2

        while (AT > 59):  # Process any other fields belonging to this annotation

            subtype,bpi,num,chan,cpychan,cpynum,aux = proc_extra_fields(AT,
                subtype,ai,filebytes,bpi,num,chan,cpychan,cpynum,aux)

            # Only aux and sub are reset between annotations. Chan and num keep
            # previous value if missing.
            AT = filebytes[bpi, 1] >> 2

        if (ai > 0):  # Fill in previous values of chan and num.
            if cpychan:
                chan[ai] = chan[ai - 1]
            if cpynum:
                num[ai] = num[ai - 1]

        # Finished processing current annotation. Move onto next.
        ai = ai + 1

        if sampto and sampto<ts:
            break;

    # Snip the unallocated end of the arrays
    annsamp,anntype,num,subtype,chan,aux = snip_arrays(annsamp,anntype,num,subtype,chan,aux,ai)

    # Apply annotation range (from X to Y)
    annsamp,anntype,num,subtype,chan,aux = apply_annotation_range(annsamp,
        sampfrom,sampto,anntype,num,subtype,chan,aux)

    # Format the annotation types as symbols or strings
    anntype = format_anntype(anndisp,anntype)

    # Store fields in an Annotation object
    annotation = Annotation(annsamp = annsamp, anntype = anntype, subtype = subtype,
        chan = chan, num = num, aux = aux, fs = fs)

    return annotation

# Load the annotation file 1 byte at a time and arrange in pairs
def loadbytepairs(recordname, annot):
    with open(recordname + '.' + annot, 'rb') as f:
        filebytes = np.fromfile(f, '<u1').reshape([-1, 2])
    return filebytes

# Initialize arrays for storing content
def init_arrays(annotlength):
    annsamp = np.zeros(annotlength)
    anntype = np.zeros(annotlength)
    subtype = np.zeros(annotlength)
    chan = np.zeros(annotlength)
    num = np.zeros(annotlength)
    aux = [''] * annotlength
    return (annotlength, annsamp, anntype, subtype, chan, num, aux)

# Check the beginning of the annotation file for an fs
def get_fs(filebytes):

    fs = None # fs potentially stored in the file
    bpi = 0 # Byte pair index for searching the annotation file

    if filebytes.size > 24:
        testbytes = filebytes[:12, :].flatten()
        # First 2 bytes indicate dt=0 and anntype=NOTE. Next 2 indicate auxlen
        # and anntype=AUX. Then follows "## time resolution: "
        if [testbytes[i] for i in [ 0,1] + list(range(3,24))] == 
        [0,88,252,35,35,32,116,105,109,101,32,114,101,115,111,108,117,116,105,111,110,58,32]:  
            # The file's leading bytes match the expected pattern for encoding fs.
            # Length of the auxilliary string that includes the fs written into
            # the file.
            auxlen = testbytes[2]
            testbytes = filebytes[:(12 + int(np.ceil(auxlen / 2.))), :].flatten()
            fs = int("".join([chr(char)
                                 for char in testbytes[24:auxlen + 4]]))
            # byte pair index to start reading actual annotations.
            bpi = 0.5 * (auxlen + 12 + (auxlen & 1))
    return (fs, bpi)

def copy_prev(AT,ts,filebytes,bpi,annsamp,anntype,ai):    
    if AT == 59:  # Skip.
        ts = ts + 65536 * filebytes[bpi + 1,0] + \
               16777216 * filebytes[bpi + 1,1] + \
                          filebytes[bpi + 2,0] + 256 * filebytes[bpi + 2,1]  # 4 bytes storing dt
        annsamp[ai] = ts
        # The anntype is stored after the 4 bytes. Samples here should be 0.
        anntype[ai] = filebytes[bpi + 3, 1] >> 2
        bpi = bpi + 4
    # Not a skip so it should be the actual samples + anntype. Should not
    # need to check for alternatives.
    else:
        # total samples = previous + delta samples stored in current byte
        # pair
        ts = ts + filebytes[bpi, 0] + 256 * (filebytes[bpi, 1] & 3)
        annsamp[ai] = ts
        anntype[ai] = AT
        bpi = bpi + 1
    return ts,annsamp,anntype,bpi

def proc_extra_fields(AT,subtype,ai,filebytes,bpi,num,chan,cpychan,cpynum,aux):
    if AT == 61:  # SUB
        # sub is interpreted as signed char. Remember to limit writing
        # range.
        subtype[ai] = filebytes[bpi, 0].astype('i1')
        bpi = bpi + 1
    elif AT == 62:  # CHAN
        # chan is interpreted as unsigned char
        chan[ai] = filebytes[bpi, 0]
        cpychan = 0
        bpi = bpi + 1
    elif AT == 60:  # NUM
        # num is interpreted as signed char
        num[ai] = filebytes[bpi, 0].astype('i1')
        cpynum = 0
        bpi = bpi + 1
    elif AT == 63:  # AUX
        # length of aux string. Max 256? No need to check other bits of
        # second byte?
        auxlen = filebytes[bpi, 0]
        auxbytes = filebytes[bpi + 1:bpi + 1 + int(np.ceil(auxlen / 2.)),:].flatten()
        if auxlen & 1:
            auxbytes = auxbytes[:-1]
        aux[ai] = "".join([chr(char) for char in auxbytes])  # The aux string
        bpi = bpi + 1 + int(np.ceil(auxlen / 2.))
    return subtype,bpi,num,chan,cpychan,cpynum,aux

# Remove unallocated part of array
def snip_arrays(annsamp,anntype,num,subtype,chan,aux,ai):
    annsamp = annsamp[0:ai].astype(int)
    anntype = anntype[0:ai].astype(int)
    num = num[0:ai].astype(int)
    subtype = subtype[0:ai].astype(int)
    chan = chan[0:ai].astype(int)
    aux = aux[0:ai]
    return annsamp,anntype,num,subtype,chan,aux

# Keep only the specified annotations
def apply_annotation_range(annsamp,sampfrom,sampto,anntype,num,subtype,chan,aux):
    
    returnempty = 0

    afterfrom = np.where(annsamp >= sampfrom)[0]
    if len(afterfrom) > 0:
        ik0 = afterfrom[0]  # index keep start
    else:  # No annotations in specified range.
        returnempty = 1

    if not sampto:
        sampto = annsamp[-1]

    beforeto = np.where(annsamp <= sampto)[0]

    if len(beforeto) > 0:
        ik1 = beforeto[-1]
    else:
        returnempty = 1

    if returnempty:
        annsamp = []
        anntype = []
        num = []
        subtype = []
        chan = []
        aux = []
        print("No annotations in specified sample range")
    else:
        annsamp = annsamp[ik0:ik1 + 1]
        anntype = anntype[ik0:ik1 + 1]
        num = num[ik0:ik1 + 1]
        subtype = subtype[ik0:ik1 + 1]
        chan = chan[ik0:ik1 + 1]
        aux = aux[ik0:ik1 + 1]
    return annsamp,anntype,num,subtype,chan,aux

def format_anntype(anndisp,anntype):
    if anndisp == 1:
        anntype = [annsyms[code] for code in anntype]
    elif anndisp == 2:
        anntype = [anncodes[code] for code in anntype]
    return anntype




## ------------- /Reading Annotations ------------- ##


# Annotation mnemonic symbols for the 'anntype' field as specified in annot.c
# from wfdb software library 10.5.24. At this point, several values are blank.
annsyms = {
    0: ' ',  # not-QRS (not a getann/putann codedict) */
    1: 'N',  # normal beat */
    2: 'L',  # left bundle branch block beat */
    3: 'R',  # right bundle branch block beat */
    4: 'a',  # aberrated atrial premature beat */
    5: 'V',  # premature ventricular contraction */
    6: 'F',  # fusion of ventricular and normal beat */
    7: 'J',  # nodal (junctional) premature beat */
    8: 'A',  # atrial premature contraction */
    9: 'S',  # premature or ectopic supraventricular beat */
    10: 'E',  # ventricular escape beat */
    11: 'j',  # nodal (junctional) escape beat */
    12: '/',  # paced beat */
    13: 'Q',  # unclassifiable beat */
    14: '~',  # signal quality change */
    #15: '[15]',
    16: '|',  # isolated QRS-like artifact */
    #17: '[17]',
    18: 's',  # ST change */
    19: 'T',  # T-wave change */
    20: '*',  # systole */
    21: 'D',  # diastole */
    22: '"',  # comment annotation */
    23: '=',  # measurement annotation */
    24: 'p',  # P-wave peak */
    25: 'B',  # left or right bundle branch block */
    26: '^',  # non-conducted pacer spike */
    27: 't',  # T-wave peak */
    28: '+',  # rhythm change */
    29: 'u',  # U-wave peak */
    30: '?',  # learning */
    31: '!',  # ventricular flutter wave */
    32: '[',  # start of ventricular flutter/fibrillation */
    33: ']',  # end of ventricular flutter/fibrillation */
    34: 'e',  # atrial escape beat */
    35: 'n',  # supraventricular escape beat */
    36: '@',  # link to external data (aux contains URL) */
    37: 'x',  # non-conducted P-wave (blocked APB) */
    38: 'f',  # fusion of paced and normal beat */
    39: '(',  # waveform onset */
    # 39: 'PQ', # PQ junction (beginning of QRS) */
    40: ')',  # waveform end */
    # 40: 'JPT', # J point (end of QRS) */
    41: 'r',  # R-on-T premature ventricular contraction */
    #42: '[42]',
    #43: '[43]',
    #44: '[44]',
    #45: '[45]',
    #46: '[46]',
    #47: '[47]',
    #48: '[48]',
    #49: '[49]',
}

# Annotation codes for 'anntype' field as specified in ecgcodes.h from
# wfdb software library 10.5.24
anncodes = {
    0: 'NOTQRS',  # not-QRS (not a getann/putann codedict) */
    1: 'NORMAL',  # normal beat */
    2: 'LBBB',  # left bundle branch block beat */
    3: 'RBBB',  # right bundle branch block beat */
    4: 'ABERR',  # aberrated atrial premature beat */
    5: 'PVC',  # premature ventricular contraction */
    6: 'FUSION',  # fusion of ventricular and normal beat */
    7: 'NPC',  # nodal (junctional) premature beat */
    8: 'APC',  # atrial premature contraction */
    9: 'SVPB',  # premature or ectopic supraventricular beat */
    10: 'VESC',  # ventricular escape beat */
    11: 'NESC',  # nodal (junctional) escape beat */
    12: 'PACE',  # paced beat */
    13: 'UNKNOWN',  # unclassifiable beat */
    14: 'NOISE',  # signal quality change */
    16: 'ARFCT',  # isolated QRS-like artifact */
    18: 'STCH',  # ST change */
    19: 'TCH',  # T-wave change */
    20: 'SYSTOLE',  # systole */
    21: 'DIASTOLE',  # diastole */
    22: 'NOTE',  # comment annotation */
    23: 'MEASURE',  # measurement annotation */
    24: 'PWAVE',  # P-wave peak */
    25: 'BBB',  # left or right bundle branch block */
    26: 'PACESP',  # non-conducted pacer spike */
    27: 'TWAVE',  # T-wave peak */
    28: 'RHYTHM',  # rhythm change */
    29: 'UWAVE',  # U-wave peak */
    30: 'LEARN',  # learning */
    31: 'FLWAV',  # ventricular flutter wave */
    32: 'VFON',  # start of ventricular flutter/fibrillation */
    33: 'VFOFF',  # end of ventricular flutter/fibrillation */
    34: 'AESC',  # atrial escape beat */
    35: 'SVESC',  # supraventricular escape beat */
    36: 'LINK',  # link to external data (aux contains URL) */
    37: 'NAPC',  # non-conducted P-wave (blocked APB) */
    38: 'PFUS',  # fusion of paced and normal beat */
    39: 'WFON',  # waveform onset */
    # 39: 'PQ', # PQ junction (beginning of QRS) */
    40: 'WFOFF',  # waveform end */
    # 40: 'JPT', # J point (end of QRS) */
    41: 'RONT'  # R-on-T premature ventricular contraction */
}

# Mapping annotation symbols to the annotation codes
# For printing/user guidance
symcodes = pd.DataFrame({'Ann Symbol': list(annsyms.values()), 'Ann Code/Meaning': list(anncodes.values())})
symcodes = symcodes.set_index('Ann Symbol', list(annsyms.values()))

annfields = ['annsamp', 'anntype', 'num', 'subtype', 'chan', 'aux', 'fs']

annfieldtypes = {'annsamp': _headers.inttypes, 'anntypes': [str], 
              'num':_headers.inttypes, 'subtype': _headers.inttypes, 
              'chan' = _headers.inttypes, 'aux': [str], 
              'fs': _headers.floattypes}