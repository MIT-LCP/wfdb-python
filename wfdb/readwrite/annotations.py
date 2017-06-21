import numpy as np
import pandas as pd
import re
import os
import copy
from . import records
from . import _headers
from . import downloads

# Class for WFDB annotations
class Annotation(object):
    """
    The class representing WFDB annotations. 

    Annotation objects can be created using the constructor, or by reading a WFDB annotation
    file with 'rdann'. 

    The attributes of the Annotation object give information about the annotation as specified
    by https://www.physionet.org/physiotools/wag/annot-5.htm:
    - annsamp: The annotation location in samples relative to the beginning of the record.
    - anntype: The annotation type according the the standard WFDB codes.
    - subtype: The marked class/category of the annotation.
    - chan: The signal channel associated with the annotations.
    - num: The labelled annotation number.
    - aux: The auxiliary information string for the annotation.
    - fs: The sampling frequency of the record if contained in the annotation file.
    - custom_anntypes: The custom annotation types defined in the annotation file.
      A dictionary with {key:value} corresponding to {anntype:description}.
      eg. {'#': 'lost connection', 'C': 'reconnected'}

    Constructor function:
    def __init__(self, recordname, annotator, annsamp, anntype, subtype = None, 
                 chan = None, num = None, aux = None, fs = None, custom_anntypes = None)

    Call 'showanncodes()' to see the list of standard annotation codes. Any text used to label 
    annotations that are not one of these codes should go in the 'aux' field rather than the 
    'anntype' field.

    Example usage:
    import wfdb
    ann1 = wfdb.Annotation(recordname='ann1', annotator='atr', annsamp=[10,20,400],
                           anntype = ['N','N','['], aux=[None, None, 'Serious Vfib'])
    """
    def __init__(self, recordname, annotator, annsamp, anntype, subtype = None, 
                 chan = None, num = None, aux = None, fs = None, custom_anntypes=None):
        self.recordname = recordname
        self.annotator = annotator

        self.annsamp = annsamp
        self.anntype = anntype
        self.subtype = subtype
        self.chan = chan
        self.num = num
        self.aux = aux
        self.fs = fs
        self.custom_anntypes = custom_anntypes

    # Equal comparison operator for objects of this type
    def __eq__(self, other):
        att1 = self.__dict__
        att2 = other.__dict__

        if set(att1.keys()) != set(att2.keys()):
            return False

        for k in att1.keys():
            v1 = att1[k]
            v2 = att2[k]

            if type(v1) != type(v2):
                return False

            if type(v1) == np.ndarray:
                if not np.array_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False

        return True

    # Write an annotation file
    def wrann(self, writefs=False):
        """
        Instance method to write a WFDB annotation file from an Annotation object.
        
        def wrann(self, writefs=False)
        
        Input Parameters:
        - writefs (default=False): Flag specifying whether to write the fs
          attribute to the file.

        """
        # Check the validity of individual fields used to write the annotation file
        self.checkfields() 

        # Check the cohesion of fields used to write the annotation
        self.checkfieldcohesion()
        
        # Write the header file using the specified fields
        self.wrannfile(writefs)

    # Check the mandatory and set fields of the annotation object
    # Return indices of anntype field which are not encoded, and thus need
    # to be moved to the aux field.
    def checkfields(self):
        # Enforce the presence of mandatory write fields
        for field in ['recordname', 'annotator', 'annsamp', 'anntype']:
            if getattr(self, field) is None:
                print('The ', field, ' field is mandatory for writing annotation files')
                raise Exception('Missing required annotation field')

        # Check all set fields
        for field in annfields:
            if getattr(self, field) is not None:
                # Check the type of the field's elements
                self.checkfield(field)

    # Check a particular annotation field
    def checkfield(self, field):

        # Non list/array fields
        if field in ['recordname', 'annotator', 'fs', 'custom_anntypes']:
            # Check the field type
            if type(getattr(self, field)) not in annfieldtypes[field]:
                if len(annfieldtypes[field]>1):
                    raise TypeError('The '+field+' field must be one of the following types:', annfieldtypes)
                else:
                    raise TypeError('The '+field+' field must be the following type:', annfieldtypes[0])
            # Field specific checks
            if field == 'recordname':
                # Allow letters, digits, hyphens, and underscores.
                acceptedstring = re.match('[-\w]+', self.recordname)
                if not acceptedstring or acceptedstring.string != self.recordname:
                    raise ValueError('recordname must only comprise of letters, digits, hyphens, and underscores.')
            elif field == 'annotator':
                # Allow letters only
                acceptedstring = re.match('[a-zA-Z]+', self.annotator)
                if not acceptedstring or acceptedstring.string != self.annotator:
                    raise ValueError('annotator must only comprise of letters')
            elif field == 'fs':
                if self.fs <=0:
                    raise ValueError('The fs field must be a non-negative number')
            elif field == 'custom_anntypes':
                # All key/values must be strings
                for key in self.custom_anntypes.keys():
                    if type(key)!= str:
                        raise ValueError('All custom_anntypes keys must be strings')
                    if len(key)>1:
                        raise ValueError('All custom_anntypes keys must be single characters')
                    # Discourage (but not prevent) using existing codes
                    if key in annsyms.values():
                        print('It is discouraged to define the custom annotation code: '+key+' that already has an entry in the wfdb library.')
                        print('To see existing annotation codes and their meanings, call: showanncodes(). Continuing...')

                for value in self.custom_anntypes.values():
                    if type(key)!= str:
                        raise ValueError('All custom_anntypes dictionary values must be strings')
                    # No pointless characters
                    acceptedstring = re.match('[\w -]+', value)
                    if not acceptedstring or acceptedstring.string != value:
                        raise ValueError('custom_anntypes dictionary values must only contain alphanumerics, spaces, underscores, and dashes')

        else:
            fielditem = getattr(self, field)

            # annsamp must be a numpy array, not a list.
            if field == 'annsamp':
                if type(fielditem) != np.ndarray:
                    raise TypeError('The '+field+' field must be a numpy array')
            # Ensure the field item is a list or array.
            else:
                if type(fielditem) not in [list, np.ndarray]:
                    raise TypeError('The '+field+' field must be a list or numpy array')

            # Check the data types of the elements.
            # If the field is a numpy array, just check dtype. If list, check individual elements.
            # annsamp and anntype may NOT have nones. Others may.
            if type(fielditem) == np.ndarray:
                if fielditem.dtype not in intdtypes:
                    raise TypeError('The '+field+' field must have one of the following dtypes:', intdtypes)
            else:
                if field =='anntype':
                    for item in fielditem:
                        if type(item) not in annfieldtypes[field]:
                            print("All elements of the '"+field+"' field must be one of the following types:", annfieldtypes[field])
                            print("All elements must be present")
                            raise Exception()
                else:
                    for item in fielditem:
                        if item is not None and type(item) not in annfieldtypes[field]:
                            print("All elements of the '", field, "' field must be one of the following types:")
                            print(annfieldtypes[field])
                            print("Elements may also be set to 'None'")
                            raise Exception()
        
            # Field specific checks
            # The C WFDB library stores num/sub/chan as chars. 
            if field == 'annsamp':
                sampdiffs = np.concatenate(([self.annsamp[0]], np.diff(self.annsamp)))
                if min(self.annsamp) < 0 :
                    raise ValueError("The 'annsamp' field must only contain non-negative integers")
                if min(sampdiffs) < 0 :
                    raise ValueError("The 'annsamp' field must contain monotonically increasing sample numbers")
                if max(sampdiffs) > 2147483648:
                    raise ValueError('WFDB annotation files cannot store sample differences greater than 2**31')
            elif field == 'anntype':
                # Ensure all fields lie in standard WFDB annotation codes or custom codes
                if set(self.anntype) - set(annsyms.values()).union() != set():
                    print("The 'anntype' field contains items not encoded in the WFDB library, or in this object's custom defined anntypes.")
                    print('To see the valid annotation codes call: showanncodes()')
                    print('To transfer non-encoded anntype items into the aux field call: self.type2aux()')
                    print("To define custom codes, set the custom_anntypes field as a dictionary with format: {custom anntype character:description}")
                    raise Exception()
            elif field == 'subtype':
                # signed character
                if min(self.subtype) < 0 or max(self.subtype) >127:
                    raise ValueError("The 'subtype' field must only contain non-negative integers up to 127")
            elif field == 'chan':
                # unsigned character
                if min(self.chan) < 0 or max(self.chan) >255:
                    raise ValueError("The 'chan' field must only contain non-negative integers up to 255")
            elif field == 'num':
                # signed character
                if min(self.num) < 0 or max(self.num) >127:
                    raise ValueError("The 'num' field must only contain non-negative integers up to 127")
            #elif field == 'aux': # No further conditions for aux field.
                

    # Ensure all written annotation fields have the same length
    def checkfieldcohesion(self):

        # Number of annotation samples
        nannots = len(self.annsamp)
        for field in ['annsamp', 'anntype', 'num', 'subtype', 'chan', 'aux']:
            if getattr(self, field) is not None:
                if len(getattr(self, field)) != nannots:
                    raise ValueError("All written annotation fields: ['annsamp', 'anntype', 'num', 'subtype', 'chan', 'aux'] must have the same length")

    # Write an annotation file
    def wrannfile(self, writefs):

        # Calculate the fs bytes to write if present and desired to write
        if self.fs is not None and writefs:
            fsbytes = fs2bytes(self.fs)
        else:
            fsbytes = []

        # Calculate the custom_anntypes bytes to write if present
        if self.custom_anntypes is not None:
            cabytes = ca2bytes(self.custom_anntypes)
        else:
            cabytes = []

        # Calculate the main bytes to write
        databytes = self.fieldbytes()

        # Combine all bytes to write: fs (if any), custom annotations(if any), main content, file terminator
        databytes = np.concatenate((fsbytes, cabytes, databytes, np.array([0,0]))).astype('u1')

        # Write the file
        with open(self.recordname+'.'+self.annotator, 'wb') as f:
            databytes.tofile(f)

    # Convert all used annotation fields into bytes to write
    def fieldbytes(self):

        # The difference samples to write
        annsampdiff = np.concatenate(([self.annsamp[0]], np.diff(self.annsamp)))


        # Create a copy of the annotation object with a
        # compact version of fields to write
        compact_annotation = copy.deepcopy(self)
        compact_annotation.compact_fields()


        # The optional fields to be written. Write if they are not None or all empty
        extrawritefields = []

        for field in ['num', 'subtype', 'chan', 'aux']:
            if not isblank(getattr(compact_annotation, field)):
                extrawritefields.append(field)

        databytes = []

        # Iterate across all fields one index at a time
        for i in range(len(annsampdiff)):

            # Process the annsamp (difference) and anntype items
            databytes.append(field2bytes('samptype', [annsampdiff[i], self.anntype[i]]))

            # Process the extra optional fields
            for field in extrawritefields:
                value = getattr(compact_annotation, field)[i]
                if value is not None:
                    databytes.append(field2bytes(field, value))

        # Flatten and convert to correct format
        databytes = np.array([item for sublist in databytes for item in sublist]).astype('u1')

        return databytes

    # Compact all of the object's fields so that the output
    # writing annotation file writes as few bytes as possible
    def compact_fields(self):

        # Number of annotations
        nannots = len(self.annsamp)

        # Chan and num carry over previous fields. Get lists of as few
        # elements to write as possible
        self.chan = compact_carry_field(self.chan)
        self.num = compact_carry_field(self.num)

        # Elements of 0 (default) do not need to be written for subtype.
        # num and sub are signed in original c package...
        if self.subtype is not None:
            if type(self.subtype) == list:
                for i in range(nannots):
                    if self.subtype[i] == 0:
                        self.subtype[i] = None
                if np.array_equal(self.subtype, [None]*nannots):
                    self.subtype = None
            else:
                zero_inds = np.where(self.subtype==0)[0]
                if len(zero_inds) == nannots:
                    self.subtype = None
                else:
                    self.subtype = list(self.subtype)
                    for i in zero_inds:
                        self.subtype[i] = None
            
        # Empty aux strings are not written
        if self.aux is not None:
            for i in range(nannots):
                if self.aux[i] == '':
                    self.aux[i] = None
            if np.array_equal(self.aux, [None]*nannots):
                self.aux = None


    # Move non-encoded anntype elements into the aux field 
    def type2aux(self):

        # Ensure that anntype is a list of strings
        if type(self.anntype)!= list:
            raise TypeError('anntype must be a list')
        for at in self.anntype:
            if type(at) != str:
                raise TypeError('anntype elements must all be strings')

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

            for i in [i for i,x in enumerate(self.anntype) if x == ext]:
                if not self.aux[i]:
                    self.aux[i] = self.anntype[i]
                    self.anntype[i] = '"'
                else:
                    self.aux[i] = self.anntype[i]+' '+self.aux[i]
                    self.anntype[i] = '"'

# Calculate the bytes written to the annotation file for the fs field
def fs2bytes(fs):

    # Initial indicators of encoding fs
    databytes = [0,88, None, 252,35,35,32,116,105,109,101,32,114,101,115,111,108,117,116,105,111,110,58,32]

    # Be aware of potential float and int

    # Check if fs is close enough to int
    if type(fs) == float:
        if round(fs,8) == float(int(fs)):
            fs = int(fs)

    fschars = str(fs)
    ndigits = len(fschars)

    for i in range(ndigits):
        databytes.append(ord(fschars[i]))

    # Fill in the aux length
    databytes[2] = ndigits + 20

    # odd number of digits
    if ndigits % 2:
        databytes.append(0)

    # Add the extra -1 0 notqrs filler
    databytes = databytes+[0, 236, 255, 255, 255, 255, 1, 0] 

    return np.array(databytes).astype('u1')

# Calculate the bytes written to the annotation file for the custom_anntypes field
def ca2bytes(custom_anntypes):

    # The start wrapper: '0 NOTE length AUX ## annotation type definitions'
    headbytes = [0,88,30,252,35,35,32,97,110,110,111,116,97,116,105,111,110,32,116,
                 121,112,101,32,100,101,102,105,110,105,116,105,111,110,115]

    # The end wrapper: '0 NOTE length AUX ## end of definitions' followed by SKIP -1, +1
    tailbytes =  [0,88,21,252,35,35,32,101,110,100,32,111,102,32,100,101,102,105,110,
                  105,116,105,111,110,115,0,0,236,255,255,255,255,1,0]

    # Annotation codes range from 0-49.
    freenumbers = list(set(range(50)) - set(annsyms.keys()))

    if len(custom_anntypes) > len(freenumbers):
        raise Exception('There can only be a maximum of '+len(freenumbers)+' custom annotation codes.')

    # Allocate a number to each custom anntype.
    # List sublists: [number, code, description]
    writecontent = []
    for i in range(len(custom_anntypes)):
        writecontent.append([freenumbers[i],list(custom_anntypes.keys())[i],list(custom_anntypes.values())[i]])

    custombytes = [customcode2bytes(triplet) for triplet in writecontent]
    custombytes = [item for sublist in custombytes for item in sublist]

    return np.array(headbytes+custombytes+tailbytes).astype('u1')

# Convert triplet of [number, codesymbol (character), description] into annotation bytes
# Helper function to ca2bytes
def customcode2bytes(c_triplet):

    # Structure: 0, NOTE, len(aux), AUX, codenumber, space, codesymbol, space, description, (0 null if necessary)
    # Remember, aux string includes 'number(s)<space><symbol><space><description>''
    annbytes = [0, 88, len(c_triplet[2]) + 3 + len(str(c_triplet[0])), 252] + [ord(c) for c in str(c_triplet[0])] \
               + [32] + [ord(c_triplet[1])] + [32] + [ord(c) for c in c_triplet[2]] 

    if len(annbytes) % 2:
        annbytes.append(0)

    return annbytes

# Tests whether the item is blank
def isblank(x):
    if x is None:
        return True
    elif type(x) == list:
        if np.array_equal(x, [None]*len(x)):
            return True
    return False


def compact_carry_field(full_field):
    """
    Return the compact list version of a list/array of an
    annotation field that has previous values carried over
    (chan or num)
    - The first sample is 0 by default. Only set otherwise
      if necessary.
    - Only set fields if they are different from their prev
      field
    """

    # Keep in mind that the field may already be compact or None

    if full_field is None:
        return None

    # List of same length. Place None where element
    # does not need to be written
    compact_field = [None]*len(full_field)

    prev_field = 0

    for i in range(len(full_field)):
        current_field = full_field[i]
        if current_field != prev_field:
            compact_field[i] = current_field
            prev_field = current_field

    # May further simplify
    if np.array_equal(compact_field, [None]*len(full_field)):
        compact_field = None

    return compact_field


# Convert an annotation field into bytes to write
def field2bytes(field, value):

    databytes = []

    # annsamp and anntype bytes come together
    if field == 'samptype':

        # Numerical value encoding annotation symbol
        typecode = revannsyms[value[1]]
        # sample difference
        sd = value[0]

        # Add SKIP element if value is too large for single byte
        if sd>1023:
            # 8 bytes in total:
            # - [0, 59>>2] indicates SKIP
            # - Next 4 gives sample difference
            # - Final 2 give 0 and anntype
            databytes = [0, 236, (sd&16711680)>>16, (sd&4278190080)>>24, sd&255, (sd&65280)>>8, 0, 4*typecode]
        # Just need annsamp and anntype
        else:
            # - First byte stores low 8 bits of annsamp
            # - Second byte stores high 2 bits of annsamp
            #   and anntype
            databytes = [sd & 255, ((sd & 768) >> 8) + 4*typecode]

    elif field == 'num':
        # First byte stores num
        # second byte stores 60*4 indicator
        databytes = [value, 240]
    elif field == 'subtype':
        # First byte stores subtype
        # second byte stores 61*4 indicator
        databytes = [value, 244]
    elif field == 'chan':
        # First byte stores num
        # second byte stores 62*4 indicator
        databytes = [value, 248]
    elif field == 'aux':
        # - First byte stores length of aux field
        # - Second byte stores 63*4 indicator
        # - Then store the aux string characters
        databytes = [len(value), 252] + [ord(i) for i in value]    
        # Zero pad odd length aux strings
        if len(value) % 2: 
            databytes.append(0)

    return databytes


# Function for writing annotations
def wrann(recordname, annotator, annsamp, anntype, subtype = None, chan = None, num = None, aux = None, fs = None):
    """Write a WFDB annotation file.

    Usage:
    wrann(recordname, annotator, annsamp, anntype, num = None, subtype = None, chan = None, aux = None, fs = None)

    Input arguments:
    - recordname (required): The string name of the WFDB record to be written (without any file extensions). 
    - annotator (required): The string annotation file extension.
    - annsamp (required): The annotation location in samples relative to the beginning of the record. List or numpy array.
    - anntype (required): The annotation type according the the standard WFDB codes. List or numpy array.
    - subtype (default=None): The marked class/category of the annotation. List or numpy array.
    - chan (default=None): The signal channel associated with the annotations. List or numpy array.
    - num (default=None): The labelled annotation number. List or numpy array.
    - aux (default=None): The auxiliary information string for the annotation. List or numpy array.
    - fs (default=None): The numerical sampling frequency of the record to be written to the file.

    Note: This gateway function was written to enable a simple way to write WFDB annotation files without
          needing to explicity create an Annotation object beforehand. 
          
          You may also create an Annotation object, manually set its attributes, and call its wrann() instance method. 
          
    Note: Each annotation stored in a WFDB annotation file contains an annsamp and an anntype field. All other fields
          may or may not be present. Therefore in order to save space, when writing additional features such
          as 'aux' that are not present for every annotation, it is recommended to make the field a list, with empty 
          indices set to None so that they are not written to the file.

    Example Usage: 
    import wfdb
    # Read an annotation as an Annotation object
    annotation = wfdb.rdann('b001', 'atr', pbdir='cebsdb')
    # Call the gateway wrann function, manually inserting fields as function input parameters
    wfdb.wrann('b001', 'cpy', annotation.annsamp, annotation.anntype)
    """    

    # Create Annotation object
    annotation = Annotation(recordname, annotator, annsamp, anntype, subtype, chan, num, aux, fs)
    # Perform field checks and write the annotation file
    annotation.wrann(writefs = True)

# Display the annotation symbols and the codes they represent
def showanncodes():
    """
    Display the annotation symbols and the codes they represent according to the 
    standard WFDB library 10.5.24
    
    Usage: 
    showanncodes()
    """
    print(symcodes)

## ------------- Reading Annotations ------------- ##


def rdann(recordname, annotator, sampfrom=0, sampto=None, shiftsamps=False, pbdir=None):
    """ Read a WFDB annotation file recordname.annotator and return an
    Annotation object.

    Usage: 
    annotation = rdann(recordname, annotator, sampfrom=0, sampto=None, pbdir=None)

    Input arguments:
    - recordname (required): The record name of the WFDB annotation file. ie. for 
      file '100.atr', recordname='100'
    - annotator (required): The annotator extension of the annotation file. ie. for 
      file '100.atr', annotator='atr'
    - sampfrom (default=0): The minimum sample number for annotations to be returned.
    - sampto (default=None): The maximum sample number for annotations to be returned.
    - shiftsamps (default=False): Boolean flag that specifies whether to return the
      sample indices relative to 'sampfrom' (True), or sample 0 (False). Annotation files
      store exact sample locations.
    - pbdir (default=None): Option used to stream data from Physiobank. The Physiobank database 
      directory from which to find the required annotation file.
      eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb', pbdir = 'mitdb'.

    Output argument:
    - annotation: The Annotation object. Call help(wfdb.Annotation) for the attribute
      descriptions.

    Note: For every annotation sample, the annotation file explictly stores the 'annsamp' 
    and 'anntype' fields but not necessarily the others. When reading annotation files
    using this function, fields which are not stored in the file will either take their
    default values of 0 or None, or will be carried over from their previous values if any.

    Example usage:
    import wfdb
    ann = wfdb.rdann('sampledata/100', 'atr', sampto = 300000)
    """

    if sampto and sampto <= sampfrom:
        raise ValueError("sampto must be greater than sampfrom")
    if sampfrom < 0:
        raise ValueError("sampfrom must be a non-negative integer")

    # Read the file in byte pairs
    filebytes = loadbytepairs(recordname, annotator, pbdir)

    # The maximum number of annotations potentially contained
    maxannlen = filebytes.shape[0]

    # Initialise arrays to the total potential number of annotations in the file
    annsamp, anntype, subtype, chan, num, aux = init_arrays(maxannlen)

    # Check the beginning of the file for a potential fs field
    # bpi = byte pair index. The index at which to continue processing the bytes
    fs, bpi = get_fs(filebytes)

    # Get the main annotation fields from the annotation bytes
    annsamp,anntype,num,subtype,chan,aux,ai  = proc_ann_bytes(annsamp,anntype,num,
                                                              subtype,chan,aux,
                                                              filebytes,bpi,maxannlen,
                                                              sampto)

    # Snip the unallocated end of the arrays
    annsamp,anntype,num,subtype,chan,aux = snip_arrays(annsamp,anntype,num,subtype,chan,aux,ai)

    # Process and remove annotations with special types - custom anntype definition and notqrs
    allannsyms,annsamp,anntype,num,subtype,chan,aux,custom_anntypes = proc_special_types(annsamp,anntype,num,
                                                                                         subtype,chan,aux)

    # Keep annotations within the user specified sample range
    annsamp,anntype,num,subtype,chan,aux = apply_annotation_range(annsamp,
        sampfrom,sampto,anntype,num,subtype,chan,aux)

    # Convert annsamp to numpy array
    annsamp = np.array(annsamp, dtype='int64')

    # Set the annotation type to the annotation codes
    anntype = [allannsyms[code] for code in anntype]

    # If the fs field was not present in the file, try to read a wfdb header
    # the annotation is associated with to get the fs
    if fs is None:
        # Use try except to not make rdann dependent on the validity of the header
        try:
            rec = records.rdheader(recordname, pbdir)
            fs = rec.fs
        except:
            pass

    # Return annotation samples relative to starting signal index
    if shiftsamps and annsamp!=[] and sampfrom:
        annsamp = annsamp - sampfrom

    # Store fields in an Annotation object
    annotation = Annotation(os.path.split(recordname)[1], annotator, annsamp, anntype, 
        subtype, chan, num, aux, fs,custom_anntypes)

    return annotation


# Load the annotation file 1 byte at a time and arrange in pairs
def loadbytepairs(recordname, annot, pbdir):
    # local file
    if pbdir is None:
        with open(recordname + '.' + annot, 'rb') as f:
            filebytes = np.fromfile(f, '<u1').reshape([-1, 2])
    # physiobank file
    else:
        filebytes = downloads.streamannotation(recordname+'.'+annot, pbdir).reshape([-1, 2])

    return filebytes

# Initialize arrays for storing content
def init_arrays(maxannlen):
    annsamp = np.zeros(maxannlen)
    anntype = np.zeros(maxannlen)
    subtype = np.zeros(maxannlen)
    chan = np.zeros(maxannlen)
    num = np.zeros(maxannlen)
    aux = [''] * maxannlen
    return (annsamp, anntype, subtype, chan, num, aux)

# Check the beginning of the annotation file for an fs
def get_fs(filebytes):

    # fs potentially stored in the file
    fs = None
    # Byte pair index for continuing to process the annotation file
    # Set default for case where fs is not present
    bpi = 0

    if filebytes.size > 24:
        testbytes = filebytes[:12, :].flatten()
        # First 2 bytes indicate dt=0 and anntype=NOTE. Next 2 indicate auxlen
        # and anntype=AUX. Then follows "## time resolution: "
        if [testbytes[i] for i in [ 0,1] + list(range(3,24))] == [0,88,252,35,35,32,116,105,109,101,32,114,101,115,111,108,117,116,105,111,110,58,32]:  
            # The file's leading bytes match the expected pattern for encoding fs.
            # Length of the auxilliary string that includes the fs written into
            # the file.
            auxlen = testbytes[2]
            testbytes = filebytes[:(12 + int(np.ceil(auxlen / 2.))), :].flatten()
            fs = float("".join([chr(char) for char in testbytes[24:auxlen + 4]]))
            # fs may be int
            if round(fs, 8) == float(int(fs)):
                fs = int(fs)
            # byte pair index to start reading actual annotations.
            bpi = int(0.5 * (auxlen + 12 + (auxlen & 1)))
    return (fs, bpi)

#  Get the main annotation fields from the annotation bytes
def proc_ann_bytes(annsamp,anntype,num,subtype,chan,aux,filebytes,bpi,maxannlen,sampto):

    # Indexing Variables
    # Total number of samples from beginning of record. Annotation bytes only store dt
    ts = 0
    # Annotation index, the number of annotations processed.
    ai = 0

    # Process annotations. Iterate across byte pairs.
    # Sequence for one ann is:
    # - SKIP pair (if any)
    # - samp + anntype pair
    # - other pairs (if any)
    # The last byte pair is 0 indicating eof.
    while (bpi < maxannlen - 1):

        # anntype - use this to get final anntype and annsamp
        AT = filebytes[bpi, 1] >> 2

        # Get the annsamp and anntype fields of the current annotation
        ts, annsamp, anntype, bpi = get_samp_type(AT,ts,filebytes,bpi,annsamp,anntype,ai)

        # Flags that specify whether to copy the previous channel/num value for
        # the current annotation. Set default.
        cpychan, cpynum = True, True

        # get the next anntype value - it may indicate additional
        # fields for this annotation, or the values of the next annotation.
        AT = filebytes[bpi, 1] >> 2

        # Process any other fields belonging to this annotation
        while (AT > 59):

            subtype,bpi,num,chan,cpychan,cpynum,aux = proc_extra_fields(AT,
                subtype,ai,filebytes,bpi,num,chan,cpychan,cpynum,aux)

            AT = filebytes[bpi, 1] >> 2

        # Carry over previous values of chan and num if necessary
        chan, num = carry_fields(ai, cpychan, cpynum, chan, num)

        # Finished processing current annotation. Move onto next.
        ai = ai + 1

        if sampto and sampto<ts:
            break

    return annsamp,anntype,num,subtype,chan,aux,ai 

# Get the annsamp and anntype fields of the current annotation
def get_samp_type(AT,ts,filebytes,bpi,annsamp,anntype,ai):
    # The first byte pair will either store the actual samples + anntype,
    # or 0 + SKIP.
    
    # Skip.
    if AT == 59:
        # 4 bytes storing dt
        dt = 65536 * filebytes[bpi + 1,0] + 16777216 * filebytes[bpi + 1,1] \
             + filebytes[bpi + 2,0] + 256 * filebytes[bpi + 2,1]

        # Data type is long integer (stored in two's complement). Range -2**31 to 2**31 - 1
        if dt > 2147483647:
            dt = dt - 4294967296

        # After the 4 bytes, the next pair's annsamp is also added
        dt = dt + filebytes[bpi + 3, 0] + 256 * (filebytes[bpi + 3, 1] & 3)

        ts = ts + dt
        annsamp[ai] = ts
        # The anntype is stored after the 4 bytes. Samples here should be 0.
        # Note: Could there be another skip?
        anntype[ai] = filebytes[bpi + 3, 1] >> 2
        bpi = bpi + 4
    # Not a skip - it is the actual sample number + anntype.
    else:
        # total samples = previous + delta samples stored in current byte pair
        ts = ts + filebytes[bpi, 0] + 256 * (filebytes[bpi, 1] & 3)
        annsamp[ai] = ts
        anntype[ai] = AT
        bpi = bpi + 1

    return ts,annsamp,anntype,bpi


def proc_extra_fields(AT,subtype,ai,filebytes,bpi,num,chan,cpychan,cpynum,aux):
    # Process extra fields belonging to the current annotation

    # aux and sub are reset between annotations. chan and num copy over
    # previous value if missing.

    # SUB
    if AT == 61:
        # sub is interpreted as signed char.
        # range.
        subtype[ai] = filebytes[bpi, 0].astype('i1')
        bpi = bpi + 1
    # CHAN
    elif AT == 62:
        # chan is interpreted as unsigned char
        chan[ai] = filebytes[bpi, 0]
        cpychan = False
        bpi = bpi + 1
    # NUM
    elif AT == 60:
        # num is interpreted as signed char
        num[ai] = filebytes[bpi, 0].astype('i1')
        cpynum = False
        bpi = bpi + 1
    # AUX
    elif AT == 63:
        # length of aux string. Max 256? No need to check other bits of
        # second byte?
        auxlen = filebytes[bpi, 0]
        auxbytes = filebytes[bpi + 1:bpi + 1 + int(np.ceil(auxlen / 2.)),:].flatten()
        if auxlen & 1:
            auxbytes = auxbytes[:-1]
        # The aux string
        aux[ai] = "".join([chr(char) for char in auxbytes])
        bpi = bpi + 1 + int(np.ceil(auxlen / 2.))
    return subtype,bpi,num,chan,cpychan,cpynum,aux


def carry_fields(ai, cpychan, cpynum, chan, num):
    # Carry over previous values of chan and num if necessary
    if (ai > 0):
        if cpychan:
            chan[ai] = chan[ai - 1]
        if cpynum:
            num[ai] = num[ai - 1]

    return chan, num 



# Remove unallocated part of array
def snip_arrays(annsamp,anntype,num,subtype,chan,aux,ai):
    annsamp = annsamp[0:ai].astype(int)
    anntype = anntype[0:ai].astype(int)
    num = num[0:ai].astype(int)
    subtype = subtype[0:ai].astype(int)
    chan = chan[0:ai].astype(int)
    aux = aux[0:ai]
    return annsamp,anntype,num,subtype,chan,aux

# Keep annotations within the user specified sample range
def apply_annotation_range(annsamp,sampfrom,sampto,anntype,num,subtype,chan,aux):
    
    # No need to do anything if no custom user input range
    if not (sampfrom == 0 and sampto is None):

        returnempty = False

        afterfrom = np.where(annsamp >= sampfrom)[0]
        if len(afterfrom) > 0:
            # index keep start
            ik0 = afterfrom[0]
        # No annotations in specified range.
        else:
            returnempty = True

        if not sampto:
            sampto = annsamp[-1]

        beforeto = np.where(annsamp <= sampto)[0]

        if len(beforeto) > 0:
            ik1 = beforeto[-1]
        else:
            returnempty = True

        if returnempty:
            annsamp = []
            anntype = []
            num = []
            subtype = []
            chan = []
            aux = []
        else:
            annsamp = annsamp[ik0:ik1 + 1]
            anntype = anntype[ik0:ik1 + 1]
            num = num[ik0:ik1 + 1]
            subtype = subtype[ik0:ik1 + 1]
            chan = chan[ik0:ik1 + 1]
            aux = aux[ik0:ik1 + 1]

    return annsamp,anntype,num,subtype,chan,aux


# Process and remove annotations with special types - custom anntype definition and notqrs
def proc_special_types(annsamp,anntype,num,subtype,chan,aux):
    # Custom anncodes appear as regular annotations in the form: 
    # sample = 0, anntype = 22 (note annotation '"'), aux = "<number>[ \t]<customanncode>[ \t]<customannstring>"
    s0_inds = np.where(annsamp == 0)[0]
    t22_inds = np.where(anntype == 22)[0]

    # Annotations with anntype == notqrs are also to be removed
    notqrs_inds = np.where(anntype == 0)[0]

    special_inds = list(set(s0_inds).intersection(t22_inds).union(notqrs_inds))

    # Custom annotation types
    custom_anntypes = {}
    # The annotation symbol dictionary to modify and use
    allannsyms = annsyms.copy()

    if special_inds != []:
        # The annotation indices to be removed
        rm_inds = []

        # Check aux for custom codes
        for i in special_inds:
            # notqrs annotations
            if anntype[i] == 0:
                rm_inds.append(i)
            # Encasing annotations for custom annotation types
            elif aux[i] == "## annotation type definitions" or aux[i] == '## end of definitions':
                rm_inds.append(i)
            # Custom annotation types
            else:
                acceptedstring = re.match('(\d+)[ \t](\S)[ \t](.+)', aux[i])
                # Found custom annotation code. 
                if acceptedstring is not None and acceptedstring.string==aux[i]:
                    rm_inds.append(i)
                    # Code number, symbol, and description
                    codenum, codesym, codedesc = acceptedstring.group(1, 2, 3)
                    allannsyms[int(codenum)] = codesym
                    custom_anntypes[codesym] = codedesc
            
        # Remove the 'annotations' with the special anncode indices
        if rm_inds != []:
            keepinds = [i for i in range(len(annsamp)) if i not in rm_inds]

            annsamp = annsamp[keepinds]
            anntype = anntype[keepinds]
            num = num[keepinds]
            subtype = subtype[keepinds]
            chan = chan[keepinds]
            aux = [aux[i] for i in keepinds]

    if custom_anntypes == {}:
        custom_anntypes = None

    return (allannsyms,annsamp,anntype,num,subtype,chan,aux,custom_anntypes)
    

## ------------- /Reading Annotations ------------- ##


# Annotation mnemonic symbols for the 'anntype' field as specified in annot.c
# from wfdb software library 10.5.24. At this point, several values are blank.
# Commented out values are present in original file but have no meaning.
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
#    15: '[15]',
    16: '|',  # isolated QRS-like artifact */
#    17: '[17]',
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
    40: ')',  # waveform end */
    41: 'r',  # R-on-T premature ventricular contraction */
#    42: '[42]',
#    43: '[43]',
#    44: '[44]',
#    45: '[45]',
#    46: '[46]',
#    47: '[47]',
#    48: '[48]',
#    49: '[49]',
}
# Reverse ann symbols for mapping symbols back to numbers
revannsyms = {v: k for k, v in annsyms.items()}

# Annotation codes for 'anntype' field as specified in ecgcodes.h from
# wfdb software library 10.5.24. Commented out values are present in
# original file but have no meaning.
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
#    15: '',
    16: 'ARFCT',  # isolated QRS-like artifact */
#    17: '',
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
    40: 'WFOFF',  # waveform end */
    41: 'RONT',  # R-on-T premature ventricular contraction */  
#    42: '',
#    43: '', 
#    44: '', 
#    45: '', 
#    46: '',
#    47: '',
#    48: '', 
#    49: ''
}

# Mapping annotation symbols to the annotation codes
# For printing/user guidance
symcodes = pd.DataFrame({'Ann Symbol': list(annsyms.values()), 'Ann Code Meaning': list(anncodes.values())})
symcodes = symcodes.set_index('Ann Symbol', list(annsyms.values()))

# All annotation fields. Note: custom_anntypes placed first to check field before anntype
annfields = ['recordname', 'annotator', 'custom_anntypes', 'annsamp', 'anntype', 'num', 'subtype', 'chan', 'aux', 'fs']

annfieldtypes = {'recordname': [str], 'annotator': [str], 'annsamp': _headers.inttypes, 
                 'anntype': [str], 'num':_headers.inttypes, 'subtype': _headers.inttypes, 
                 'chan': _headers.inttypes, 'aux': [str], 'fs': _headers.floattypes,
                 'custom_anntypes': [dict]}

# Acceptable numpy integer dtypes
intdtypes = ['int64', 'uint64', 'int32', 'uint32','int16','uint16']