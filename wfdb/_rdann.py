## Written by: Chen Xie 2016 ##
# Please report bugs and suggestions to
# https://github.com/MIT-LCP/wfdb-python or cx1111@mit.edu ###

import numpy as np
import os
import math

def get_sample_freq(filebytes):
    """Check the beginning of the annotation file to see if it is storing the
    'time resolution' field.
    """
    annfs = [] # Store frequencies if they appear in the annotation file.
    bpi = 0 # Byte pair index, for searching through bytes of the annotation file.
    if filebytes.size > 24:
        testbytes = filebytes[:12, :].flatten()
        # First 2 bytes indicate dt=0 and anntype=NOTE. Next 2 indicate auxlen
        # and anntype=AUX. Then follows "## time resolution: "
        if [
                testbytes[i] for i in [
                    0,1] +
                list(
                    range(3,24))] == [0,88,252,35,35,32,116,105,109,101,32,114,
                                      101,115,111,108,117,116,105,111,110,58,32]:  
            # The file's leading bytes match the expected pattern for encoding fs.
            # Length of the auxilliary string that includes the fs written into
            # the file.
            auxlen = testbytes[2]
            testbytes = filebytes[:(12 + math.ceil(auxlen / 2.)), :].flatten()
            annfs = int("".join([chr(char)
                                 for char in testbytes[24:auxlen + 4]]))
            # byte pair index to start reading actual annotations.
            bpi = 0.5 * (auxlen + 12 + (auxlen & 1))
    return annfs,bpi

def rdann(recordname, annot, sampfrom=0, sampto=[], anndisp=1):
    """ Read a WFDB annotation file recordname.annot and return the fields as lists or arrays

    Usage: annsamp, anntype, num, subtype, chan, aux, annfs) = rdann(recordname, annot, sampfrom=0, sampto=[], anndisp=1)

    Input arguments:
    - recordname (required): The record name of the WFDB annotation file. ie. for file '100.atr', recordname='100'
    - annot (required): The annotator extension of the annotation file. ie. for file '100.atr', annot='atr'
    - sampfrom (default=0): The minimum sample number for annotations to be returned.
    - sampto (default=the final annotation sample): The maximum sample number for annotations to be returned.
    - anndisp (default = 1): The annotation display flag that controls the data type of the 'anntype' output parameter. 'anntype' will either be an integer key(0), a shorthand display symbol(1), or a longer annotation code.

    Output arguments:
    - annsamp: The annotation location in samples relative to the beginning of the record.
    - anntype: The annotation type according the the standard WFDB keys.
    - subtype: The marked class/category of the annotation.
    - chan: The signal channel associated with the annotations.
    - num: The marked annotation number. This is not equal to the index of the current annotation.
    - aux: The auxiliary information string for the annotation.
    - annfs: The sampling frequency written in the beginning of the annotation file if present.

    *NOTE: Every annotation contains the 'annsamp' and 'anntype' field. All other fields default to 0 or empty if not present.
    """

    if sampto and sampto <= sampfrom:
        raise ValueError("sampto must be greater than sampfrom")

    # Get the info from the header file
    # fields=readheader(recordname)
    dirname, baserecordname = os.path.split(recordname)

    # Read the file's byte pairs.
    with open(recordname + '.' + annot, 'rb') as f:
        filebytes = np.fromfile(f, '<u1').reshape([-1, 2])

    # Allocate for the maximum possible number of annotations contained in the
    # file.
    annsamp = np.zeros(filebytes.shape[0])
    anntype = np.zeros(filebytes.shape[0])
    subtype = np.zeros(filebytes.shape[0])
    chan = np.zeros(filebytes.shape[0])
    num = np.zeros(filebytes.shape[0])
    aux = [''] * filebytes.shape[0]

    # Annotation index, the number of annotations processed. Not to be
    # comfused with the 'num' field of an annotation.
    ai = 0

    # Check the beginning of the annotation file to see if it is storing the
    # 'time resolution' field.
    annfs,bpi = get_sample_freq(filebytes)

    # Total number of samples of current annotation from beginning of record.
    # Annotation bytes only store dt.
    ts = 0

    # Processing annotations. Sequence for one ann is: SKIP pair (if any) ->
    # samp + anntype pair -> other pairs
    # The last byte pair is 0 indicating eof.
    while bpi < filebytes.shape[0] - 1:

        # The first byte pair will either store the actual samples + anntype,
        # or 0 + SKIP.
        AT = filebytes[bpi, 1] >> 2  # anntype

        # flags that specify whether to copy the previous channel/num value for
        # the current annotation.
        cpychan = 1
        cpynum = 1
        if AT == 59:  # Skip.
            ts = ts + 65536 * filebytes[bpi + 1,
                                        0] + 16777216 * filebytes[bpi + 1,
                                                                  1] + filebytes[bpi + 2,
                                                                                 0] + 256 * filebytes[bpi + 2,
                                                                                                      1]  # 4 bytes storing dt
            annsamp[ai] = ts
            # The anntype is stored after the 4 bytes. Samples here should be
            # 0.
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

        AT = filebytes[bpi, 1] >> 2
        while (AT > 59):  # Process any other fields belonging to this annotation
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
                auxbytes = filebytes[
                    bpi +
                    1:bpi +
                    1 +
                    math.ceil(
                        auxlen /
                        2.),
                    :].flatten()  # The aux bytes
                if auxlen & 1:
                    auxbytes = auxbytes[:-1]
                aux[ai] = "".join([chr(char)
                                   for char in auxbytes])  # The aux string
                bpi = bpi + 1 + math.ceil(auxlen / 2.)
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

    # Get rid of the unallocated parts of the arrays
    annsamp = annsamp[0:ai].astype(int)
    anntype = anntype[0:ai].astype(int)
    num = num[0:ai].astype(int)
    subtype = subtype[0:ai].astype(int)
    chan = chan[0:ai].astype(int)
    aux = aux[0:ai]

    # Keep the annotations in the specified range
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

        # Return the annotation types as symbols or strings depending on input
        # parameter
        if anndisp == 1:
            anntype = [annsyms[code] for code in anntype]
        elif anndisp == 2:
            anntype = [anncodes[code] for code in anntype]

    return (annsamp, anntype, subtype, chan, num, aux, annfs)


# Annotation print symbols for 'anntype' field as specified in annot.c
# from wfdb software library 10.5.24
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
    15: '[15]',
    16: '|',  # isolated QRS-like artifact */
    17: '[17]',
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
    42: '[42]',
    43: '[43]',
    44: '[44]',
    45: '[45]',
    46: '[46]',
    47: '[47]',
    48: '[48]',
    49: '[49]',
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

if __name__ == '__main__':
    rdann(sys.argv)
