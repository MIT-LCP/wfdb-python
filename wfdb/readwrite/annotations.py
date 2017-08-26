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
    - samp: The annotation location in samples relative to the beginning of the record.
    - sym: The annotation type according the the standard WFDB codes.
    - subtype: The marked class/category of the annotation.
    - chan: The signal channel associated with the annotations.
    - num: The labelled annotation number.
    - aux_note: The aux_noteiliary information string for the annotation.
    - fs: The sampling frequency of the record if contained in the annotation file.
    - custom_labels: The custom annotation types defined in the annotation file.
      A dictionary with {key:value} corresponding to {sym:description}.
      eg. {'#': 'lost connection', 'C': 'reconnected'}
    - label_map: (storevalue, symbol, description)

    Constructor function:
    def __init__(self, recordname, annotator, samp, sym, subtype = None, 
                 chan = None, num = None, aux_note = None, fs = None, custom_labels = None)

    Call 'show_ann_labels()' to see the list of standard annotation codes. Any text used to label 
    annotations that are not one of these codes should go in the 'aux_note' field rather than the 
    'sym' field.

    Example usage:
    import wfdb
    ann1 = wfdb.Annotation(recordname='ann1', annotator='atr', samp=[10,20,400],
                           sym = ['N','N','['], aux_note=[None, None, 'Serious Vfib'])
    """
    def __init__(self, recordname, annotator, samples, symbols=None, subtype=None, 
                 chan=None, num=None, aux_note=None, fs=None, label_stores=None,
                 descriptions=None, custom_label_map=None, label_map=None):
        self.recordname = recordname
        self.annotator = annotator

        self.samples = samples

        self.symbols = symbols

        self.subtype = subtype
        self.chan = chan
        self.num = num
        self.aux_note = aux_note
        self.fs = fs

        self.label_stores = label_stores
        self.descriptions = descriptions

        self.custom_label_map = custom_label_map
        self.label_map = label_map



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
    # Return indices of sym field which are not encoded, and thus need
    # to be moved to the aux_note field.
    def checkfields(self):
        # Enforce the presence of mandatory write fields
        for field in ['recordname', 'annotator', 'samp', 'sym']:
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
        if field in ['recordname', 'annotator', 'fs', 'custom_labels']:
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
            elif field == 'custom_labels':
                # All key/values must be strings
                for key in self.custom_labels.keys():
                    if type(key)!= str:
                        raise ValueError('All custom_labels keys must be strings')
                    if len(key)>1:
                        raise ValueError('All custom_labels keys must be single characters')

                for value in self.custom_labels.values():
                    if type(key)!= str:
                        raise ValueError('All custom_labels dictionary values must be strings')
                    # No pointless characters
                    acceptedstring = re.match('[\w -]+', value)
                    if not acceptedstring or acceptedstring.string != value:
                        raise ValueError('custom_labels dictionary values must only contain alphanumerics, spaces, underscores, and dashes')

        else:
            fielditem = getattr(self, field)

            # samp must be a numpy array, not a list.
            if field == 'samp':
                if type(fielditem) != np.ndarray:
                    raise TypeError('The '+field+' field must be a numpy array')
            # Ensure the field item is a list or array.
            else:
                if type(fielditem) not in [list, np.ndarray]:
                    raise TypeError('The '+field+' field must be a list or numpy array')

            # Check the data types of the elements.
            # If the field is a numpy array, just check dtype. If list, check individual elements.
            # samp and sym may NOT have nones. Others may.
            if type(fielditem) == np.ndarray:
                if fielditem.dtype not in intdtypes:
                    raise TypeError('The '+field+' field must have one of the following dtypes:', intdtypes)
            else:
                if field =='sym':
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
            if field == 'samp':
                sampdiffs = np.concatenate(([self.samp[0]], np.diff(self.samp)))
                if min(self.samp) < 0 :
                    raise ValueError("The 'samp' field must only contain non-negative integers")
                if min(sampdiffs) < 0 :
                    raise ValueError("The 'samp' field must contain monotonically increasing sample numbers")
                if max(sampdiffs) > 2147483648:
                    raise ValueError('WFDB annotation files cannot store sample differences greater than 2**31')
            elif field == 'sym':
                # Ensure all fields lie in standard WFDB annotation codes or custom codes
                if set(self.sym) - set(ann_label_table['Symbol'].values).union() != set():
                    print("The 'sym' field contains items not encoded in the WFDB library, or in this object's custom defined syms.")
                    print('To see the valid annotation codes call: show_ann_labels()')
                    print('To transfer non-encoded sym items into the aux_note field call: self.type2aux_note()')
                    print("To define custom codes, set the custom_labels field as a dictionary with format: {custom sym character:description}")
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
            #elif field == 'aux_note': # No further conditions for aux_note field.
                

    # Ensure all written annotation fields have the same length
    def checkfieldcohesion(self):

        # Number of annotation samples
        nannots = len(self.samp)
        for field in ['samp', 'sym', 'num', 'subtype', 'chan', 'aux_note']:
            if getattr(self, field) is not None:
                if len(getattr(self, field)) != nannots:
                    raise ValueError("All written annotation fields: ['samp', 'sym', 'num', 'subtype', 'chan', 'aux_note'] must have the same length")

    # Write an annotation file
    def wrannfile(self, writefs):

        # Calculate the fs bytes to write if present and desired to write
        if self.fs is not None and writefs:
            fsbytes = fs2bytes(self.fs)
        else:
            fsbytes = []

        # Calculate the custom_labels bytes to write if present
        if self.custom_labels is not None:
            cabytes = ca2bytes(self.custom_labels)
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
        sampdiff = np.concatenate(([self.samp[0]], np.diff(self.samp)))


        # Create a copy of the annotation object with a
        # compact version of fields to write
        compact_annotation = copy.deepcopy(self)
        compact_annotation.compact_fields()


        # The optional fields to be written. Write if they are not None or all empty
        extrawritefields = []

        for field in ['num', 'subtype', 'chan', 'aux_note']:
            if not isblank(getattr(compact_annotation, field)):
                extrawritefields.append(field)

        databytes = []

        # Iterate across all fields one index at a time
        for i in range(len(sampdiff)):

            # Process the samp (difference) and sym items
            databytes.append(field2bytes('samptype', [sampdiff[i], self.sym[i]]))

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
        nannots = len(self.samp)

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
            
        # Empty aux_note strings are not written
        if self.aux_note is not None:
            for i in range(nannots):
                if self.aux_note[i] == '':
                    self.aux_note[i] = None
            if np.array_equal(self.aux_note, [None]*nannots):
                self.aux_note = None


    # Move non-encoded sym elements into the aux_note field 
    def type2aux_note(self):

        # Ensure that sym is a list of strings
        if type(self.sym)!= list:
            raise TypeError('sym must be a list')
        for at in self.sym:
            if type(at) != str:
                raise TypeError('sym elements must all be strings')

        external_syms = set(self.sym) - set(ann_label_table['Symbol'].values)

        # Nothing to do
        if external_syms == set():
            return

        # There is at least one external value.

        # Initialize aux_note field if necessary 
        if self.aux_note == None:
            self.aux_note = [None]*len(self.samp)

        # Move the sym fields
        for ext in external_syms:

            for i in [i for i,x in enumerate(self.sym) if x == ext]:
                if not self.aux_note[i]:
                    self.aux_note[i] = self.sym[i]
                    self.sym[i] = '"'
                else:
                    self.aux_note[i] = self.sym[i]+' '+self.aux_note[i]
                    self.sym[i] = '"'

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

    # Fill in the aux_note length
    databytes[2] = ndigits + 20

    # odd number of digits
    if ndigits % 2:
        databytes.append(0)

    # Add the extra -1 0 notqrs filler
    databytes = databytes+[0, 236, 255, 255, 255, 255, 1, 0] 

    return np.array(databytes).astype('u1')

# Calculate the bytes written to the annotation file for the custom_labels field
def ca2bytes(custom_labels):

    # The start wrapper: '0 NOTE length aux_note ## annotation type definitions'
    headbytes = [0,88,30,252,35,35,32,97,110,110,111,116,97,116,105,111,110,32,116,
                 121,112,101,32,100,101,102,105,110,105,116,105,111,110,115]

    # The end wrapper: '0 NOTE length aux_note ## end of definitions' followed by SKIP -1, +1
    tailbytes =  [0,88,21,252,35,35,32,101,110,100,32,111,102,32,100,101,102,105,110,
                  105,116,105,111,110,115,0,0,236,255,255,255,255,1,0]

    # Annotation codes range from 0-49.
    freenumbers = list(set(range(50)) - set(ann_label_table['Store-Value'].values))

    if len(custom_labels) > len(freenumbers):
        raise Exception('There can only be a maximum of '+len(freenumbers)+' custom annotation codes.')

    # Allocate a number to each custom sym.
    # List sublists: [number, code, description]
    writecontent = []
    for i in range(len(custom_labels)):
        writecontent.append([freenumbers[i],list(custom_labels.keys())[i],list(custom_labels.values())[i]])

    custombytes = [customcode2bytes(triplet) for triplet in writecontent]
    custombytes = [item for sublist in custombytes for item in sublist]

    return np.array(headbytes+custombytes+tailbytes).astype('u1')

# Convert triplet of [number, codesymbol (character), description] into annotation bytes
# Helper function to ca2bytes
def customcode2bytes(c_triplet):

    # Structure: 0, NOTE, len(aux_note), aux_note, codenumber, space, codesymbol, space, description, (0 null if necessary)
    # Remember, aux_note string includes 'number(s)<space><symbol><space><description>''
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

    # samp and sym bytes come together
    if field == 'samptype':
        # Numerical value encoding annotation symbol
        typecode = ann_label_table.loc[ann_label_table['Symbol']==value[1], 'Store-Value'].values[0]

        # sample difference
        sd = value[0]

        # Add SKIP element if value is too large for single byte
        if sd>1023:
            # 8 bytes in total:
            # - [0, 59>>2] indicates SKIP
            # - Next 4 gives sample difference
            # - Final 2 give 0 and sym
            databytes = [0, 236, (sd&16711680)>>16, (sd&4278190080)>>24, sd&255, (sd&65280)>>8, 0, 4*typecode]
        # Just need samp and sym
        else:
            # - First byte stores low 8 bits of samp
            # - Second byte stores high 2 bits of samp
            #   and sym
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
    elif field == 'aux_note':
        # - First byte stores length of aux_note field
        # - Second byte stores 63*4 indicator
        # - Then store the aux_note string characters
        databytes = [len(value), 252] + [ord(i) for i in value]    
        # Zero pad odd length aux_note strings
        if len(value) % 2: 
            databytes.append(0)

    return databytes


# Function for writing annotations
def wrann(recordname, annotator, samp, sym, subtype = None, chan = None, num = None, aux_note = None, fs = None):
    """Write a WFDB annotation file.

    Usage:
    wrann(recordname, annotator, samp, sym, num = None, subtype = None, chan = None, aux_note = None, fs = None)

    Input arguments:
    - recordname (required): The string name of the WFDB record to be written (without any file extensions). 
    - annotator (required): The string annotation file extension.
    - samp (required): The annotation location in samples relative to the beginning of the record. List or numpy array.
    - sym (required): The annotation type according the the standard WFDB codes. List or numpy array.
    - subtype (default=None): The marked class/category of the annotation. List or numpy array.
    - chan (default=None): The signal channel associated with the annotations. List or numpy array.
    - num (default=None): The labelled annotation number. List or numpy array.
    - aux_note (default=None): The aux_noteiliary information string for the annotation. List or numpy array.
    - fs (default=None): The numerical sampling frequency of the record to be written to the file.

    Note: This gateway function was written to enable a simple way to write WFDB annotation files without
          needing to explicity create an Annotation object beforehand. 
          
          You may also create an Annotation object, manually set its attributes, and call its wrann() instance method. 
          
    Note: Each annotation stored in a WFDB annotation file contains an samp and an sym field. All other fields
          may or may not be present. Therefore in order to save space, when writing additional features such
          as 'aux_note' that are not present for every annotation, it is recommended to make the field a list, with empty 
          indices set to None so that they are not written to the file.

    Example Usage: 
    import wfdb
    # Read an annotation as an Annotation object
    annotation = wfdb.rdann('b001', 'atr', pbdir='cebsdb')
    # Call the gateway wrann function, manually inserting fields as function input parameters
    wfdb.wrann('b001', 'cpy', annotation.samp, annotation.sym)
    """    

    # Create Annotation object
    annotation = Annotation(recordname, annotator, samp, sym, subtype, chan, num, aux_note, fs)
    # Perform field checks and write the annotation file
    annotation.wrann(writefs = True)


def show_ann_labels():
    """
    Display the standard wfdb annotation label mapping
    
    Usage: 
    show_ann_labels()
    """
    print(ann_label_table)


def show_ann_classes():
    """
    Display the standard wfdb annotation classes
    """

    pass

## ------------- Reading Annotations ------------- ##


def rdann(recordname, annotator, sampfrom=0, sampto=None, shiftsamps=False,
          pbdir=None, return_label_types=['label_stores']):
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

    Note: For every annotation sample, the annotation file explictly stores the 'samp' 
    and 'sym' fields but not necessarily the others. When reading annotation files
    using this function, fields which are not stored in the file will either take their
    default values of 0 or None, or will be carried over from their previous values if any.

    Example usage:
    import wfdb
    ann = wfdb.rdann('sampledata/100', 'atr', sampto = 300000)
    """

    return_label_types = check_read_inputs(sampfrom, sampto, return_label_types)

    # Read the file in byte pairs
    filebytes = loadbytepairs(recordname, annotator, pbdir)

    # Get wfdb annotation fields from the file bytes
    samples, label_stores, subtype, chan, num, aux_note = proc_ann_bytes(filebytes)

    # Interpret information from special fields, and remove the corresponding raw wfdb
    # annotation fields they were initially stored as. Cases: custom label definitions, notqrs
    samples, label_stores, subtype, chan, num, aux_note, custom_label_map, fs = interpret_special_fields(samples, label_stores,
                                                                                        subtype, chan, num,
                                                                                        aux_note)

    
    # Convert samp to numpy array
    samp = np.array(samp, dtype='int64')

    # Set the annotation type to the annotation codes
    sym = [allsyms[code] for code in sym]

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
    if shiftsamps and samp!=[] and sampfrom:
        samp = samp - sampfrom

    # Store fields in an Annotation object
    annotation = Annotation(os.path.split(recordname)[1], annotator, samp, sym, 
        subtype, chan, num, aux_note, fs,custom_labels)

    return annotation


def check_read_inputs(sampfrom, sampto, return_label_types):

    label_types = ['label_stores', 'symbols', 'descriptions']

    if sampto and sampto <= sampfrom:
        raise ValueError("sampto must be greater than sampfrom")
    if sampfrom < 0:
        raise ValueError("sampfrom must be a non-negative integer")

    if isinstance(return_label_types, str):
        return_label_types = [return_label_types]

    if set.union(set(label_types), set(return_label_types))!=set(label_types):
        raise ValueError('return_label_types must be a list containing one or more of the following elements:',label_types)

    return return_label_type

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

# Check the beginning of the annotation file for an fs
def get_fs(filebytes):

    # fs potentially stored in the file
    fs = None
    # Byte pair index for continuing to process the annotation file
    # Set default for case where fs is not present
    bpi = 0

    if filebytes.size > 24:
        testbytes = filebytes[:12, :].flatten()
        # First 2 bytes indicate dt=0 and sym=NOTE. Next 2 indicate aux_notelen
        # and sym=aux_note. Then follows "## time resolution: "
        if [testbytes[i] for i in [ 0,1] + list(range(3,24))] == [0,88,252,35,35,32,116,105,109,101,32,114,101,115,111,108,117,116,105,111,110,58,32]:  
            # The file's leading bytes match the expected pattern for encoding fs.
            # Length of the aux_noteilliary string that includes the fs written into
            # the file.
            aux_notelen = testbytes[2]
            testbytes = filebytes[:(12 + int(np.ceil(aux_notelen / 2.))), :].flatten()
            fs = float("".join([chr(char) for char in testbytes[24:aux_notelen + 4]]))
            # fs may be int
            if round(fs, 8) == float(int(fs)):
                fs = int(fs)
            # byte pair index to start reading actual annotations.
            bpi = int(0.5 * (aux_notelen + 12 + (aux_notelen & 1)))
    return (fs, bpi)


#  Get regular annotation fields from the annotation bytes
def proc_ann_bytes(filebytes, sampto):

    # Base annotation fields
    samples, label_stores, subtype, chan, num, aux_note = [], [], [], [], [], []

    # Indexing Variables
    # Total number of samples from beginning of record. Annotation bytes only store sample_diff
    sample_total = 0
    # Byte pair index
    bpi = 0

    # Process annotations. Iterate across byte pairs.
    # Sequence for one ann is:
    # - SKIP pair (if any)
    # - samp + sym pair
    # - other pairs (if any)
    # The last byte pair is 0 indicating eof.
    while (bpi < filebytes.shape[0] - 1):

        # Get the sample and store_value fields of the current annotation
        sample_diff, label_store, bpi = proc_core_fields(filebytes, bpi)
        sample_total = sample_total + sample_diff
        samples.append(sample_total)
        label_stores.append(label_store)     


        # Process any other fields belonging to this annotation

        # Flags that specify whether the extra fields need to be updated
        update = {'subtype':True, 'chan':True, 'num':True, 'aux_note':True}
        # Get the next label store value - it may indicate additional
        # fields for this annotation, or the values of the next annotation.
        label_store = filebytes[bpi, 1] >> 2

        while (label_store > 59):
            subtype, chan, num, aux_note, update, bpi = proc_extra_field(label_store, filebytes, 
                                                                         bpi, subtype, chan, num,
                                                                         aux_note, update)

            label_store = filebytes[bpi, 1] >> 2

        # Set defaults or carry over previous values if necessary
        subtype, chan, num, aux_note = update_extra_fields(chan, num, cpychan, cpynum, )

        if sampto and sampto<sample_total:
            break

    return samples, label_stores, subtype, chan, num, aux_note

# Get the sample difference and store fields of the current annotation
def proc_core_fields(filebytes, bpi):
    
    label_store = filebytes[bpi, 1] >> 2

    # The current byte pair will contain either the actual d_samples + annotation store value,
    # or 0 + SKIP.
    
    # Skip. Note: Could there be another skip after the first?
    if label_store == 59:
        # 4 bytes storing dt
        sample_diff = 65536 * filebytes[bpi + 1,0] + 16777216 * filebytes[bpi + 1,1] \
             + filebytes[bpi + 2,0] + 256 * filebytes[bpi + 2,1]

        # Data type is long integer (stored in two's complement). Range -2**31 to 2**31 - 1
        if sample_diff > 2147483647:
            sample_diff = sample_diff - 4294967296

        # After the 4 bytes, the next pair's samp is also added
        sample_diff = sample_diff + filebytes[bpi + 3, 0] + 256 * (filebytes[bpi + 3, 1] & 3)

        # The label is stored after the 4 bytes. Samples here should be 0.
        label_store = filebytes[bpi + 3, 1] >> 2
        bpi = bpi + 4
    # Not a skip - it is the actual sample number + annotation type store value
    else:
        sample_diff = filebytes[bpi, 0] + 256 * (filebytes[bpi, 1] & 3)
        bpi = bpi + 1

    return sample_diff, label_store, bpi

def proc_extra_field(label_store, filebytes, bpi, subtype, chan, num, aux_note, update):
    """
    Process extra fields belonging to the current annotation.
    Potential updated fields: subtype, chan, num, aux_note
    """

    # aux_note and sub are reset between annotations. chan and num copy over
    # previous value if missing.

    # SUB
    if label_store == 61:
        # sub is interpreted as signed char.
        subtype.append(filebytes[bpi, 0].astype('i1'))
        update['subtype'] = False
        bpi = bpi + 1
    # CHAN
    elif label_store == 62:
        # chan is interpreted as unsigned char
        chan[ai] = filebytes[bpi, 0]
        update['chan'] = False
        bpi = bpi + 1
    # NUM
    elif label_store == 60:
        # num is interpreted as signed char
        num[ai] = filebytes[bpi, 0].astype('i1')
        update['num'] = False
        bpi = bpi + 1
    # aux_note
    elif label_store == 63:
        # length of aux_note string. Max 256? No need to check other bits of
        # second byte?
        aux_notelen = filebytes[bpi, 0]
        aux_notebytes = filebytes[bpi + 1:bpi + 1 + int(np.ceil(aux_notelen / 2.)),:].flatten()
        if aux_notelen & 1:
            aux_notebytes = aux_notebytes[:-1]
        # The aux_note string
        aux_note.append("".join([chr(char) for char in aux_notebytes]))
        update['aux_note'] = False
        bpi = bpi + 1 + int(np.ceil(aux_notelen / 2.))

    return subtype, chan, num, aux_note, update, bpi


def update_extra_fields(subtype, chan, num, aux_note, update):
    """
    Update the field if the current annotation did not
    provide a value.

    - aux_note and sub are set to default values if missing.
    - chan and num copy over previous value if missing.
    """
    
    if update['subtype']:
        subtype.append(0)

    if update['chan']:
        if chan == []:
            chan.append(0)
        else:
            chan.append(chane[-1])
    if update['num']:
        if num == []:
            num.append(0)
        else:
            num.append(chane[-1])

    if update['aux_note']:
        aux_note.append('')

    return subtype, chan, num, aux_note


# Process and remove annotations with special types - custom sym definition and notqrs
def interpret_special_fields(samp,sym,num,subtype,chan,aux_note):
    # Custom anncodes appear as regular annotations in the form: 
    # sample = 0, sym = 22 (note annotation '"'), aux_note = "<number>[ \t]<customanncode>[ \t]<customannstring>"
    s0_inds = np.where(samp == 0)[0]
    t22_inds = np.where(sym == 22)[0]
    
    # Annotations with sym == notqrs are also to be removed
    notqrs_inds = np.where(sym == 0)[0]

    special_inds = list(set(s0_inds).intersection(t22_inds).union(notqrs_inds))
    
    custom_labels = {}
    # The annotation symbol dictionary to modify and use
    allsyms = dict(zip(ann_label_table['Store-Value'].values, ann_label_table['Symbol'].values))

    if special_inds != []:
        # The annotation indices to be removed
        rm_inds = []

        # Check aux_note for custom codes
        for i in special_inds:
            # notqrs annotations
            if sym[i] == 0:
                rm_inds.append(i)
            # Encasing annotations for custom annotation types
            elif aux_note[i] == "## annotation type definitions" or aux_note[i] == '## end of definitions':
                rm_inds.append(i)
            # Custom annotation types
            else:
                acceptedstring = re.match('(\d+)[ \t](\S)[ \t](.+)', aux_note[i])
                # Found custom annotation code. 
                if acceptedstring is not None and acceptedstring.string==aux_note[i]:
                    rm_inds.append(i)
                    # Code number, symbol, and description
                    codenum, codesym, codedesc = acceptedstring.group(1, 2, 3)
                    allsyms[int(codenum)] = codesym
                    custom_labels[codesym] = codedesc
            
        # Remove the 'annotations' with the special anncode indices
        if rm_inds != []:
            keepinds = [i for i in range(len(samp)) if i not in rm_inds]

            samp = samp[keepinds]
            sym = sym[keepinds]
            num = num[keepinds]
            subtype = subtype[keepinds]
            chan = chan[keepinds]
            aux_note = [aux_note[i] for i in keepinds]


    return (allsyms,samp,sym,num,subtype,chan,aux_note,custom_labels)
    

## ------------- /Reading Annotations ------------- ##




# All annotation fields. Note: custom_labels placed first to check field before sym
annfields = ['recordname', 'annotator', 'custom_labels', 'samp', 'sym', 'num', 'subtype', 'chan', 'aux_note', 'fs']

annfieldtypes = {'recordname': [str], 'annotator': [str], 'samp': _headers.inttypes, 
                 'sym': [str], 'num':_headers.inttypes, 'subtype': _headers.inttypes, 
                 'chan': _headers.inttypes, 'aux_note': [str], 'fs': _headers.floattypes,
                 'custom_labels': [dict]}

# Acceptable numpy integer dtypes
intdtypes = ['int64', 'uint64', 'int32', 'uint32','int16','uint16']




# Classes = extensions
class AnnotationClass(object):
    def __init__(self, extension, description, human_reviewed):

        self.extension = extension
        self.description = description
        self.human_reviewed = human_reviewed


ann_classes = [
    AnnotationClass('atr', 'Reference ECG annotations', True),

    AnnotationClass('blh', 'Human reviewed beat labels', True),
    AnnotationClass('blm', 'Machine beat labels', False),

    AnnotationClass('alh', 'Human reviewed alarms', True),
    AnnotationClass('alm', 'Machine alarms', False),

    AnnotationClass('qrsc', 'Human reviewed qrs detections', True),
    AnnotationClass('qrs', 'Machine QRS detections', False),
    
    AnnotationClass('bph', 'Human reviewed BP beat detections', True),
    AnnotationClass('bpm', 'Machine BP beat detections', False),

    #AnnotationClass('alh', 'Human reviewed BP alarms', True),
    #AnnotationClass('alm', 'Machine BP alarms', False),
    # separate ecg and other signal category alarms?
    # Can we use signum to determine the channel it was triggered off?

    #ppg alarms?
    #eeg alarms
]

# Individual annotation labels
class AnnotationLabel(object):
    def __init__(self, store_value, symbol, short_description, description):
        self.store_value = store_value
        self.symbol = symbol
        self.short_description = short_description
        self.description = description

    def __str__(self):
        return str(self.store_value)+', '+str(self.symbol)+', '+str(self.short_description)+', '+str(self.description)

ann_labels = [
    AnnotationLabel(0, " ", 'NOTQRS', 'Not an actual annotation'),
    AnnotationLabel(1, "N", 'NORMAL', 'Normal beat'),
    AnnotationLabel(2, "L", 'LBBB', 'Left bundle branch block beat'),
    AnnotationLabel(3, "R", 'RBBB', 'Right bundle branch block beat'),
    AnnotationLabel(4, "a", 'ABERR', 'Aberrated atrial premature beat'),
    AnnotationLabel(5, "V", 'PVC', 'Premature ventricular contraction'),
    AnnotationLabel(6, "F", 'FUSION', 'Fusion of ventricular and normal beat'),
    AnnotationLabel(7, "J", 'NPC', 'Nodal (junctional) premature beat'),
    AnnotationLabel(8, "A", 'APC', 'Atrial premature contraction'),
    AnnotationLabel(9, "S", 'SVPB', 'Premature or ectopic supraventricular beat'),
    AnnotationLabel(10, "E", 'VESC', 'Ventricular escape beat'),
    AnnotationLabel(11, "j", 'NESC', 'Nodal (junctional) escape beat'),
    AnnotationLabel(12, "/", 'PACE', 'Paced beat'),
    AnnotationLabel(13, "Q", 'UNKNOWN', 'Unclassifiable beat'),
    AnnotationLabel(14, "~", 'NOISE', 'Signal quality change'),
    # AnnotationLabel(15, None, None, None),
    AnnotationLabel(16, "|", 'ARFCT',  'Isolated QRS-like artifact'),
    # AnnotationLabel(17, None, None, None),
    AnnotationLabel(18, "s", 'STCH',  'ST change'),
    AnnotationLabel(19, "T", 'TCH',  'T-wave change'),
    AnnotationLabel(20, "*", 'SYSTOLE',  'Systole'),
    AnnotationLabel(21, "D", 'DIASTOLE',  'Diastole'),
    AnnotationLabel(22, '"', 'NOTE',  'Comment annotation'),
    AnnotationLabel(23, "=", 'MEASURE',  'Measurement annotation'),
    AnnotationLabel(24, "p", 'PWAVE',  'P-wave peak'),
    AnnotationLabel(25, "B", 'BBB',  'Left or right bundle branch block'),
    AnnotationLabel(26, "^", 'PACESP',  'Non-conducted pacer spike'),
    AnnotationLabel(27, "t", 'TWAVE',  'T-wave peak'),
    AnnotationLabel(28, "+", 'RHYTHM',  'Rhythm change'),
    AnnotationLabel(29, "u", 'UWAVE',  'U-wave peak'),
    AnnotationLabel(30, "?", 'LEARN',  'Learning'),
    AnnotationLabel(31, "!", 'FLWAV',  'Ventricular flutter wave'),
    AnnotationLabel(32, "[", 'VFON',  'Start of ventricular flutter/fibrillation'),
    AnnotationLabel(33, "]", 'VFOFF',  'End of ventricular flutter/fibrillation'),
    AnnotationLabel(34, "e", 'AESC',  'Atrial escape beat'),
    AnnotationLabel(35, "n", 'SVESC',  'Supraventricular escape beat'),
    AnnotationLabel(36, "@", 'LINK',  'Link to external data (aux_note contains URL)'),
    AnnotationLabel(37, "x", 'NAPC',  'Non-conducted P-wave (blocked APB)'),
    AnnotationLabel(38, "f", 'PFUS',  'Fusion of paced and normal beat'),
    AnnotationLabel(39, "(", 'WFON',  'Waveform onset'),
    AnnotationLabel(40, ")", 'WFOFF',  'Waveform end'),
    AnnotationLabel(41, "r", 'RONT',  'R-on-T premature ventricular contraction'),
    # AnnotationLabel(42, None, None, None),
    # AnnotationLabel(43, None, None, None),
    # AnnotationLabel(44, None, None, None),
    # AnnotationLabel(45, None, None, None),
    # AnnotationLabel(46, None, None, None),
    # AnnotationLabel(47, None, None, None),
    # AnnotationLabel(48, None, None, None),
    # AnnotationLabel(49, None, None, None),
]


ann_label_table = pd.DataFrame({'Store-Value':[al.store_value for al in ann_labels], 'Symbol':[al.symbol for al in ann_labels], 
                               'Short-Description':[al.short_description for al in ann_labels], 'Description':[al.description for al in ann_labels]})
ann_label_table.set_index('Store-Value', list(ann_label_table['Store-Value'].values))
ann_label_table = ann_label_table[['Store-Value','Symbol','Short-Description','Description']]

