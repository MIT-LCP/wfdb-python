## Written by: Chen Xie 2016 ##


# rdann - Read a WFDB annotation file and return the annotation locations as a numpy array and the additional fields as lists. 

# Usage: rdann(recordname, annot)



import numpy as np
import os
import math

# Read a wfdb annotation file recordname.annot
def rdann(recordname, annot):
    
    #fields=readheader(recordname) # Get the info from the header file
    dirname, baserecordname=os.path.split(recordname)
    
    if dirname:
        dirname=dirname+"/"
    
    print()
    
    f=open(recordname+'.'+annot, 'rb')
    filebytes=np.fromfile(f, '<u1').reshape([-1, 2]) # The second (large) byte's 6 msbits in the byte pairs store the data type info and the first byte stores the actual info. 
    
     # The max possible number of annotations stored. 
    annsamp=np.empty([filebytes.shape[0],1])
    anntype=np.empty([filebytes.shape[0],1])
    subtype=np.empty([filebytes.shape[0],1])
    chan=np.empty([filebytes.shape[0],1])
    aux=[None]*filebytes.shape[0]
    
    annfs=[]
    b=0
    # Check if the beginning of the annotation file to see if it is storing the 'time resolution' field. 
    if (filebytes[0,1] >> 2 == 22) & ((filebytes[0,0]+(filebytes[0,1] & 3) )==0 ): # The first annotation is note and first annotation sample is 0 
    
    # if ((filebytes[0,0]==0) & filebytes[0,1]==88): # This is easier haha... 
    
        if filebytes[1,2] >> 2 == 63: # The next byte pair stores an aux
            auxlen=filebytes[1,1] # length of aux string
            if auxlen>19:  
                aux=filebytes[2:2+math.ceil(auxlen/2), :].flatten() # The aux bytes
                aux="".join([chr(char) for char in aux]) # The aux string
                
                if '## time resolution: '==aux[0:20]: # The annotation file is storing 'time resolution' info. 
                    annfs=int(aux[20:])
                    
                    
            
            
            #auxstring=filebytes[2:(auxlen/2), :].flatten() # The aux string 
        
        
        
        
        
        
        
        
    
    
    # Go through the entire annotation bytes and process the byte/byte pairs. 
    #while b<filebytes.shape[0]:
        
            
    
    #return (annsamp, anntype, subtyp, chan, aux, annfs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    