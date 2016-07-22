## Written by: Chen Xie 2016 ##


# rdann - Read a WFDB annotation file and return the annotation locations as a numpy array and the additional fields as lists. 

# Usage: rdann(recordname, annot)



import numpy as np
import os
import math

# Read a wfdb annotation file recordname.annot. returncodes==1 returns the annotation codes strings rather than numbers. 
def rdann(recordname, annot, returncodes=1):
    
    
anncodes = { # Annotation codes for 'anntype' field as specified in ecgcodes.h from wfdb software library 10.5.24
    0 : 'NOTQRS', # not-QRS (not a getann/putann codedict) */
    1 : 'NORMAL', # normal beat */
    2 : 'LBBB', # left bundle branch block beat */
    3 : 'RBBB', # right bundle branch block beat */
    4 : 'ABERR', # aberrated atrial premature beat */
    5 : 'PVC', # premature ventricular contraction */
    6 : 'FUSION', # fusion of ventricular and normal beat */
    7 : 'NPC', # nodal (junctional) premature beat */
    8 : 'APC', # atrial premature contraction */
    9 : 'SVPB', # premature or ectopic supraventricular beat */
    10 : 'VESC', # ventricular escape beat */
    11 : 'NESC', # nodal (junctional) escape beat */
    12 : 'PACE', # paced beat */
    13 : 'UNKNOWN', # unclassifiable beat */
    14 : 'NOISE', # signal quality change */
    16 : 'ARFCT', # isolated QRS-like artifact */
    18 : 'STCH', # ST change */
    19 : 'TCH', # T-wave change */
    20 : 'SYSTOLE', # systole */
    21 : 'DIASTOLE', # diastole */
    22 : 'NOTE', # comment annotation */
    23 : 'MEASURE', # measurement annotation */
    24 : 'PWAVE', # P-wave peak */
    25 : 'BBB', # left or right bundle branch block */
    26 : 'PACESP', # non-conducted pacer spike */
    27 : 'TWAVE', # T-wave peak */
    28 : 'RHYTHM', # rhythm change */
    29 : 'UWAVE', # U-wave peak */
    30 : 'LEARN', # learning */
    31 : 'FLWAV', # ventricular flutter wave */
    32 : 'VFON', # start of ventricular flutter/fibrillation */
    33 : 'VFOFF', # end of ventricular flutter/fibrillation */
    34 : 'AESC', # atrial escape beat */
    35 : 'SVESC', # supraventricular escape beat */
    36 : 'LINK', # link to external data (aux contains URL) */
    37 : 'NAPC', # non-conducted P-wave (blocked APB) */
    38 : 'PFUS', # fusion of paced and normal beat */
    39 : 'WFON', # waveform onset */
    #39 : 'PQ', # PQ junction (beginning of QRS) */
    40 : 'WFOFF', # waveform end */
    #40 : 'JPT', # J point (end of QRS) */
    41 : 'RONT' # R-on-T premature ventricular contraction */
    }
    
    
    #fields=readheader(recordname) # Get the info from the header file
    dirname, baserecordname=os.path.split(recordname)
    
    if dirname:
        dirname=dirname+"/"
    
    print()
    
    f=open(recordname+'.'+annot, 'rb')
    filebytes=np.fromfile(f, '<u1').reshape([-1, 2]) # The second (large) byte's 6 msbits in the byte pairs store the data type info and the first byte stores the actual info. 
    
   
    
    # Check if the beginning of the annotation file to see if it is storing the 'time resolution' field. 
    if (filebytes[0,1] >> 2 == 22) & ((filebytes[0,0]+(filebytes[0,1] & 3) )==0 ): # The first annotation is note and first annotation sample is 0 
    
    # if ((filebytes[0,0]==0) & filebytes[0,1]==88): # This is easier haha... 
    
        if filebytes[1,2] >> 2 == 63: # The next byte pair stores an aux
            auxlen=filebytes[1,1] # length of aux string
            if auxlen>19:  
                auxbytes=filebytes[2:2+math.ceil(auxlen/2), :].flatten() # The aux bytes
                aux="".join([chr(char) for char in aux]) # The aux string
                
                if '## time resolution: '==aux[0:20]: # The annotation file is storing 'time resolution' info. 
                    annfs=int(aux[20:])
                    bpi=4+auxlen
    
    
     # Allocate for the max possible number of annotations contained in the file. 
    annsamp=np.empty([filebytes.shape[0],1])
    anntype=np.empty([filebytes.shape[0],1])
    num=np.empty([filebytes.shape[0],1])
    subtype=np.empty([filebytes.shape[0],1])
    chan=np.empty([filebytes.shape[0],1])
    aux=[None]*filebytes.shape[0] 
    
    annfs=[]
    
    ai=0 # Annotation index, the number of annotations processed. Not to be comfused with the 'num' field of an annotation.
    bpi=0 # Byte pair index, for searching through bytes of the annotation file. 
    
    
    ts=0 # Total number of samples of current annotation from beginning of record
    
    # Go through the annotation bytes and process the byte/byte pairs. 
    while bpi<filebytes.shape[0]-1: # The last byte pair is 0 indicating eof. 
        
        AT=filebytes[bpi,1] >> 2 # anntype
        anntype[ai]=AT # First pair of the annotation is guaranteed to contain anntype. 
        
        ts=ts+filebytes[bpi, 0]+256*(filebytes[bpi,1] & 3) # total samples = previous + delta samples stored in current byte pair
        annsamp[ai]=ts 
         
        bpi=bpi+1
        AT=filebytes[bpi, 1] >> 2 # Move onto the next byte pair to see if they hold info for the current annotation. 
        
        while (AT >= 59): # This annotation contains more fields (other than 0) 
            if AT==59: # SKIP - Look at the next byte pair for the annotation sample.  
                ts=ts+65536*filebytes[bpi+1,0]+16777216*filebytes[bpi+1,1]+filebytes[bpi+2,0]+256*filebytes[bpi+2,1]
                annsamp[ai]=ts
                bpi=bpi+3
            elif AT==60: # NUM
                num[ai]= filebytes[bpi, 0] + 256*(filebytes[bpi,1] & 3)
                bpi=bpi+1
            elif AT==61: # SUB
                subtype[ai]= filebytes[bpi, 0] + 256*(filebytes[bpi,1] & 3)
                bpi=bpi+1
            elif AT==62: # CHAN
                chan[ai]= filebytes[bpi, 0] + 256*(filebytes[bpi,1] & 3)
                bpi=bpi+1
            elif AT==63: # AUX
                auxlen=filebytes[1,1] # length of aux string
                auxbytes=filebytes[2:2+math.ceil(auxlen/2), :].flatten() # The aux bytes
                aux[ai]="".join([chr(char) for char in aux]) # The aux string
                
            AT=filebytes[bpi,1] >> 2
        
        # No more fields for this annotation. Move on to next. 
        ai=ai+1
        
    for item in [annsamp, anntype, num, subtype, chan, aux]: # Discard the empty parts of the arrays/lists. 
        item=item[0:ai]
    
    # Return the annotation strings. 
    if returncodes==1:
        anntype=[anncodes[code] for code in anntype]
            
    
    return (annsamp, anntype, num, subtyp, chan, aux, annfs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    