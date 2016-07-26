## Written by: Chen Xie 2016 ##


# rdann - Read a WFDB annotation file and return the annotation locations, annotation types, and additional fields. 

# Usage: rdann(recordname, annot)


import numpy as np
import os
import math

# Read a wfdb annotation file recordname.annot. returncodes==1 returns the annotation codes strings rather than numbers. 
def rdann(recordname, annot, returncodes=1):

    #fields=readheader(recordname) # Get the info from the header file
    dirname, baserecordname=os.path.split(recordname)
    
    if dirname:
        dirname=dirname+"/"

    f=open(recordname+'.'+annot, 'rb')
    filebytes=np.fromfile(f, '<u1').reshape([-1, 2]) # Read the file's byte pairs.
    f.close
    
    # Allocate for the maximum possible number of annotations contained in the file. 
    annsamp=np.empty(filebytes.shape[0])
    anntype=np.empty(filebytes.shape[0])
    num=np.empty(filebytes.shape[0])
    subtype=np.empty(filebytes.shape[0])
    chan=np.empty(filebytes.shape[0])
    aux=[None]*filebytes.shape[0] 
    
    annfs=[] # Stores the fs written in the annotation file if it exists. 
    ai=0 # Annotation index, the number of annotations processed. Not to be comfused with the 'num' field of an annotation.
    bpi=0 # Byte pair index, for searching through bytes of the annotation file. 
    
    # Check the beginning of the annotation file to see if it is storing the 'time resolution' field. 
    if filebytes.size>24:
        testbytes=filebytes[:12,:].flatten()
        # First 2 bytes indicate dt=0 and anntype=NOTE. Next 2 indicate auxlen and anntype=AUX. Then follows "## time resolution: "
        if [testbytes[i] for i in [0, 1]+list(range(3, 24))]==[0, 88, 252, 35, 35, 32, 116, 105, 109, 101, 32, 114, 101, 115, 111, 108, 117, 116, 105, 111, 110, 58, 32]: # The file's leading bytes match the expected pattern for encoding fs. 
            auxlen=testbytes[2] # Length of the auxilliary string that includes the fs written into the file. 
            testbytes=filebytes[:(12+math.ceil(auxlen/2)),:].flatten()
            annfs=int("".join([chr(char) for char in testbytes[24:auxlen+4]]))
            bpi=0.5*(auxlen+12+(auxlen & 1)) # byte pair index to start reading actual annotations. 
   
    ts=0 # Total number of samples of current annotation from beginning of record. Annotation bytes only store dt. 
    
    
    # Processing annotations. Sequence for one ann is: SKIP pair (if any) -> samp + anntype pair -> other pairs 
    while bpi<filebytes.shape[0]-1: # The last byte pair is 0 indicating eof. 
        
        AT=filebytes[bpi,1] >> 2 # anntype
        # The first byte pair will either store the actual samples + anntype, or 0 + SKIP.
        
        if AT==59: # Skip. 
            ts=ts+65536*filebytes[bpi+1,0]+16777216*filebytes[bpi+1,1]+filebytes[bpi+2,0]+256*filebytes[bpi+2,1] # 4 bytes storing dt
            annsamp[ai]=ts
            anntype[ai]=filebytes[bpi+3,1] >> 2 # The anntype is stored after the 4 bytes. Samples here should be 0.   
            bpi=bpi+4
        else: # Not a skip so it should be the actual samples + anntype. Should not need to check for alternatives. 
            ts=ts+filebytes[bpi, 0]+256*(filebytes[bpi,1] & 3) # total samples = previous + delta samples stored in current byte pair
            annsamp[ai]=ts 
            anntype[ai]=AT
            bpi=bpi+1
            
        AT=filebytes[bpi,1] >> 2     
        while (AT > 59): # Process any other fields belonging to this annotation 
            
            # Must prevent this shit from reading past the end of the file. 
            if AT==60: # NUM
                num[ai]= filebytes[bpi, 0] + 256*(filebytes[bpi,1] & 3)
                bpi=bpi+1
            elif AT==61: # SUB
                subtype[ai]= filebytes[bpi, 0] + 256*(filebytes[bpi,1] & 3)
                bpi=bpi+1
            elif AT==62: # CHAN
                print("we are in chan")
                chan[ai]= filebytes[bpi, 0] + 256*(filebytes[bpi,1] & 3)
                bpi=bpi+1
            elif AT==63: # AUX
                auxlen=filebytes[bpi,0] # length of aux string. Max 256? No need to check other bits of second byte? 
                auxbytes=filebytes[bpi+1:bpi+1+math.ceil(auxlen/2), :].flatten() # The aux bytes
                if auxlen&1:
                    auxbytes=auxbytes[:-1]
                aux[ai]="".join([chr(char) for char in auxbytes]) # The aux string
                bpi=bpi+1+math.ceil(auxlen/2)
                
            AT=filebytes[bpi,1] >> 2
            
        # Finished processing current annotation. Move onto next. 
        ai=ai+1
    print("chan before int conversion:", chan)
    print("annsamp before conversion: ", annsamp)
    annsamp=annsamp[0:ai].astype(int)
    anntype=anntype[0:ai].astype(int)
    num=num[0:ai].astype(int)
    subtype=subtype[0:ai].astype(int)
    chan=chan[0:ai].astype(int)
    aux=aux[0:ai]

    # Return the annotation strings. 
    if returncodes==1:
        anntype=[anncodes[code] for code in anntype]
            
    return (annsamp, anntype, num, subtype, chan, aux, annfs)
    
    
    
# Annotation codes for 'anntype' field as specified in ecgcodes.h from wfdb software library 10.5.24
anncodes = { 
    0: 'NOTQRS', # not-QRS (not a getann/putann codedict) */
    1: 'NORMAL', # normal beat */
    2: 'LBBB', # left bundle branch block beat */
    3: 'RBBB', # right bundle branch block beat */
    4: 'ABERR', # aberrated atrial premature beat */
    5: 'PVC', # premature ventricular contraction */
    6: 'FUSION', # fusion of ventricular and normal beat */
    7: 'NPC', # nodal (junctional) premature beat */
    8: 'APC', # atrial premature contraction */
    9: 'SVPB', # premature or ectopic supraventricular beat */
    10: 'VESC', # ventricular escape beat */
    11: 'NESC', # nodal (junctional) escape beat */
    12: 'PACE', # paced beat */
    13: 'UNKNOWN', # unclassifiable beat */
    14: 'NOISE', # signal quality change */
    16: 'ARFCT', # isolated QRS-like artifact */
    18: 'STCH', # ST change */
    19: 'TCH', # T-wave change */
    20: 'SYSTOLE', # systole */
    21: 'DIASTOLE', # diastole */
    22: 'NOTE', # comment annotation */
    23: 'MEASURE', # measurement annotation */
    24: 'PWAVE', # P-wave peak */
    25: 'BBB', # left or right bundle branch block */
    26: 'PACESP', # non-conducted pacer spike */
    27: 'TWAVE', # T-wave peak */
    28: 'RHYTHM', # rhythm change */
    29: 'UWAVE', # U-wave peak */
    30: 'LEARN', # learning */
    31: 'FLWAV', # ventricular flutter wave */
    32: 'VFON', # start of ventricular flutter/fibrillation */
    33: 'VFOFF', # end of ventricular flutter/fibrillation */
    34: 'AESC', # atrial escape beat */
    35: 'SVESC', # supraventricular escape beat */
    36: 'LINK', # link to external data (aux contains URL) */
    37: 'NAPC', # non-conducted P-wave (blocked APB) */
    38: 'PFUS', # fusion of paced and normal beat */
    39: 'WFON', # waveform onset */
    #39: 'PQ', # PQ junction (beginning of QRS) */
    40: 'WFOFF', # waveform end */
    #40: 'JPT', # J point (end of QRS) */
    41: 'RONT' # R-on-T premature ventricular contraction */
    }
    
# Annotation print symbols for 'anntype' field as specified in ecgmap.h from wfdb software library 10.5.24
annsyms={
    0:'O'

    }


#static char wfdb_mamp[] = {
#	'O',	'N',	'N',	'N',	'N',		/* 0 - 4 */
#	'V',	'F',	'N',	'N',	'N',		/* 5 - 9 */
#	'E',	'N',	'P',	'Q',	'U',		/* 10 - 14 */
#	'O',	'O',	'O',	'O',	'O',		/* 15 - 19 */
#	'O',	'O',	'O',	'O',	'O',		/* 20 - 24 */
#	'N',	'O',	'O',	'O',	'O',		/* 25 - 29 */
#	'Q',	'O',	'[',	']',	'N',		/* 30 - 34 */
#	'N',	'O',	'O',	'N',	'O',		/* 35 - 39 */
#	'O',	'R',	'O',	'O',	'O',		/* 40 - 44 */
#	'O',	'O',	'O',	'O',	'O'		/* 45 - 49 */
#};
    