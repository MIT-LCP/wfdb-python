import numpy as np
import re
import os
import math

def rdsamp(recordname, sampfrom=0, sampto=[], physical=1):
    # to do: add channel selection, to and from
    
    fields=readheader(recordname) 
    
    if fields["nseg"]==1: # single segment file
        if (len(set(fields["filename"]))==1): # single dat file to read
            sig=readdat(recordname, fields["fmt"][0], fields["byteoffset"], sampfrom, sampto, fields["nsig"], fields["nsamp"])
            
            if (physical==1):
                # Insert nans/invalid samples. 

                # Values that correspond to NAN (Format 8 has no NANs)
                wfdbInvalids={'16': -32768, '24': -8388608, '32': -2147483648, '61': -32768, 
                              '80': -128, '160':-32768, '212': -2048, '310': -512, '311': -512} 
                
                # In multi dat case, be careful about NANs!!!!!
                sig=sig.astype(float)
                sig[sig==wfdbInvalids[fields["fmt"][0]]]=np.nan 

                sig=np.subtract(sig, np.array([float(i) for i in fields["baseline"]]))
                sig=np.divide(sig, np.array([fields["gain"]]))
                #sig=np.subtract(np.array(fbaseline))
                #sig=np.divide(np.array(fgain))
            
        else: # Multiple dat files. Read different dats and glue them together channel wise. 
            # ACTUALLY MULTIPLE CHANNELS CAN BE GROUPED IN DAT FILES!!!!! 
            for i in range(0, nsig):
                
                print('this is hard')
                #singlesig=readdat(fields["filename"][i][0:end-4, info["fmt"][i], sampfrom, )
    else: # Multi-segment file
        
        # Determine if this record is fixed or variable layout:
        if fields["nsampseg"][0]==0: # variable layout - first segment is layout specification file 
            startline=1
            layoutfields=readheader(fields["filename"][0]) # Store the layout header info. 
        else: # fixed layout - no layout specification file. 
            startline=0
        
        # Read segments one at a time and stack them together. fs is ALWAYS constant between segments. 
        sigsegments=[]
        fieldsegments=[]
        for segrecordname in fields["filename"][startline:]: # NEED TO ADD CONDITION FOR SAMPFROM AND SAMPTO!!!!!!
            sig, fields = rdsamp(recordname=segrecordname, sampfrom=0, 
                                      sampto=[], physical=physical)[0] # Hey look, a recursive function. I knew this lesson would come in handy one day.
            sigsegments.append(sig)
            if fields:
                fieldsegments.append(fields)
            # How to return signal? List of numpy arrays? Give user input options. 
            
        #sig=np.vstack(sigsegments)
            
    return (sig, fields)

def readheader(recordname): # For reading signal headers

    # To do: Allow exponential input format for some fields 
    
    fid=open(recordname + ".hea", 'r')
    
    # Output dictionary
    fields = {'nseg':[], 'nsig':[], 'fs':[], 'nsamp':[], 'basetime':[], 'basedate':[],  
            'filename':[], 'fmt':[], 'byteoffset':[], 'gain':[], 'units':[], 'baseline':[], 
            'initvalue':[],'signame':[], 'nsampseg':[], 'comments':[]}
    #filename stores file names for both multi and single segment headers. nsampseg is only for multi-segment

    commentlines=[] # Store comments 
    headerlines=[] # Store record line followed by the signal lines if any 

    # RECORD LINE fields (o means optional, delimiter is space or tab unless specified):
    # record name, nsegments (o, delim=/), nsignals, fs (o), counter freq (o, delim=/, needs fs), 
    # base counter (o, delim=(), needs counter freq), nsamples (o, needs fs), base time (o), 
    # base date (o needs base time).   

    # Regexp object for record line
    rxRECORD=re.compile(''.join(["(?P<name>[\w]+)/*(?P<nseg>\d*)[ \t]+",
                                "(?P<nsig>\d+)[ \t]*",
                                "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<basecounter>\d*\.?\d*)\)?[ \t]*",
                                "(?P<nsamples>\d*)[ \t]*",
                                "(?P<basetime>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
                                "(?P<basedate>\d{,2}/?\d{,2}/?\d{,4})"]))
    # Watch out for potential floats: fs (and also exponent notation...), counterfs, basecounter


    # SIGNAL LINE fields (o means optional, delimiter is space or tab unless specified):
    # file name, format, samplesperframe(o, delim=x), skew(o, delim=:), byteoffset(o,delim=+),
    # ADCgain(o), baseline(o, delim=(), requires ADCgain), units(o, delim=/, requires baseline), 
    # ADCres(o, requires ADCgain), ADCzero(o, requires ADCres), initialvalue(o, requires ADCzero), 
    # checksum(o, requires initialvalue), blocksize(o, requires checksum), signame(o, requires block)

    # Regexp object for signal lines. Consider filenames - dat and mat or more?
    rxSIGNAL=re.compile(''.join(["(?P<filename>[\w]+\.[md]at)[ \t]+(?P<format>\d+)x?"
                                 "(?P<samplesperframe>\d*):?(?P<skew>\d*)\+?(?P<byteoffset>\d*)[ \t]*",
                                 "(?P<ADCgain>\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^/-]*)[ \t]*",
                                 "(?P<ADCres>\d*)[ \t]*(?P<ADCzero>-?\d*)[ \t]*(?P<initialvalue>-?\d*)[ \t]*",
                                 "(?P<checksum>-?\d*)[ \t]*(?P<blocksize>\d*)[ \t]*(?P<signame>[\S]*)"])) 

    # check regexp allowed characters of signame...
    
    # Units characters: letters, numbers, /, ^, -, 
    # Make sure \w doesn't trigger more errors than [0-9a-zA-z_]
    # Watch out for potentially negative fields: baseline, ADCzero, initialvalue, checksum, 
    # Watch out for potential float: ADCgain. There are no negative floats. 

    # Split comment and non-comment lines

    for line in fid:
        line=line.strip()
        if line.startswith('#'): # comment line
            commentlines.append(line)
        elif line: # Non-empty non-comment line = header line. 
            ci=line.find('#')
            if ci > 0: 
                headerlines.append(line[:ci]) # header line
                commentlines.append(line[ci:]) # comment on same line as header line
            else:
                headerlines.append(line)
                
    # Get record line parameters
    (_, nseg, nsig, fs, counterfs, 
     basecounter, nsamp, basetime, basedate)=rxRECORD.findall(headerlines[0])[0]

    if not nseg:
        nseg='1'
    if not fs:
        fs='250'
    fs=float(fs)
    
    fields['nseg']=int(nseg) # These fields are either mandatory or set to defaults. 
    fields['fs']=float(fs)
    fields['nsig']=int(nsig)

    fields['nsamp']=int(nsamp) # These fields might be empty.
    fields['basetime']=basetime
    fields['basedate']=basedate

    # Signal or Segment line paramters 

    if int(nseg) >1: # Multi segment header - Process segment spec lines in current master header.
        print("I'm working on it....")
        for i in range(0, int(nsig)):
            (filename, nsampseg)=re.findall('(?P<filename>\w*~?)[ \t]+(?P\d+)', headerlines[i+1])
            fields["filename"].append(filename) # filename might be ~ for null segment. 
            fields["nsampseg"].append(int(nsampseg)) # number of samples for the segment is mandatory. 
    else: # Single segment header - Process signal spec lines in current regular header. 
        
        for i in range(0,int(nsig)): # will not run if nsignals=0
            # get signal line parameters
            #print(rxSIGNAL.findall(headerlines[i+1]))
            (filename, fmt, sampsperframe, skew, byteoffset, adcgain, baseline, units, adcres,
             adczero, initvalue, checksum, blocksize, signame)=rxSIGNAL.findall(headerlines[i+1])[0]
            
            
            #print(rxSIGNAL.findall(headerlines[i+1])[0])
            
            # Setting defaults
            if not byteoffset:
                byteoffset='0'
            if not adcgain:
                adcgain='200'
            if not baseline:
                if not adczero:
                    baseline='0'
                else:
                    baseline=adczero # missing baseline actually takes adczero value if present
            if not units:
                units='mV'
            if not initvalue:
                initvalue='0'
            if not signame:
                signame="ch" + str(i+1)
            if not initvalue:
                initvalue='0'
                

            #'filename':[], 'fmt':[], 'byteoffset':[], 'gain':[], 'units':[], 'baseline':[], 
            #'initvalue':[],'signame':[], 'nsampseg':[]}                        
            fields["filename"].append(filename)
            fields["fmt"].append(fmt)
            fields['byteoffset'].append(int(byteoffset))
            fields["gain"].append(float(adcgain))
            fields["baseline"].append(int(baseline))
            fields["units"].append(units)
            fields["initvalue"].append(int(initvalue))
            fields["signame"].append(signame)
    
    if commentlines:
        for comment in commentlines:
            fields["comments"].append(comment.strip('\s#'))
            
    return fields


def readdat(recordname, fmt, byteoffset, sampfrom, sampto, nsig, siglen): 
    # nsig and nsamp define whole file, not selected inputs. nsamp is signal length. 
    
    # to do: allow channel choice too. Put that in rdsamp actually, not here.
    filename = recordname + ".dat"
    # FIX THIS STUPID FILENAME RECORDNAME 
    
    # Bytes required to hold each sample (including wasted space)
    bytespersample={'8': 1, '16': 2, '24': 3, '32': 4, '61': 2, 
                    '80': 1, '160':2, '212': 1.5, '310': 4/3, '311': 4/3}
    
    # Data type objects for each format to load. Doesn't directly correspond for final 3 formats. 
    datatypes={'8':'<i1', '16':'<i2', '24':'<i3', '32':'<i4', 
               '61':'>i2', '80':'<u1', '160':'<u2', 
               '212':'<u1', '310':'<u1', '311':'<u2'} 
    
    if not sampto:
        if not siglen: # Signal length was not obtained, calculate it from the file size. 
            filesize=os.path.getsize(filename) 
            siglen=filesize/nsig/bytespersample[fmt]
        sampto=siglen
    
    fp=open(filename,'rb')
    
    # Need to fix the byteoffset field for multi dat records.... 
    fp.seek(int(sampfrom*nsig*bytespersample[fmt])+int(byteoffset[0])) # Point to the starting sample 
    
    # Reading the dat file into np array and reshaping. 
    if fmt=='212': # 212, 310, and 311 need special loading and processing. 
        nbytesload=math.ceil((sampto-sampfrom)*nsig*1.5) # Loaded bytes
        nbytesstack=int(nbytesload/3)*3 # Most samples come in 3 byte (24 bit) blocks. Possibly remainder 
        sig=np.fromfile(fp, dtype=np.dtype(datatypes[fmt]), count=nbytesload) 
        s1=sig[0:nbytesstack:3]+256*np.bitwise_and(sig[1:nbytesstack:3], 0x0f)
        s2=sig[2:nbytesstack:3]+256*np.bitwise_and(sig[1:nbytesstack:3] >> 4, 0x0f)
        # Arrange and flatten samples
        if (nbytesload == nbytesstack): 
            sig=np.vstack((s1,s2)).flatten('F')
        else: # Append last sample that didn't completely fit into 3 byte blocks if any. Maximum 1
            sig=np.hcat((np.vstack((s1,s2)).flatten('F')), sig[nbytesstack]+256*np.bitwise_and(sig[nbytesstack+1], 0x0f))
        
        sig[sig>2047]-=4096 # Loaded values as unsigned. Convert to 2's complement form: values > 2^11-1 are negative.
    
    elif fmt=='310': # Three 10 bit samples packed into 4 bytes with 2 bits discarded
        nbytesload=math.ceil((sampto-sampfrom)*nsig*4/3)
        nbytesstack=int(nbytesload/4)*4 # Most samples come in 4 byte (32 bit) blocks. Possibly remainders
        if nbytesload%4 == 3: # Actually need to load 1 more byte because of separate format.  
            nbytesload+=1

        sig=np.fromfile(fp, dtype=np.dtype(datatypes[fmt]), count=nbytesload)
        # Loaded number of bytes = rounded up from required (will not surpass file limit). 
        # But process round down to nearest 4 bytes. Then get remainder if necessary. 
        s1=(sig[0:nbytesload:4] >> 1) +1024*np.bitwise_and(sig[1:nbytesload:4], 0x07)
        s2=(sig[2:nbytesload:4] >> 1) +1024*np.bitwise_and(sig[3:nbytesload:4], 0x07)
        s3=np.bitwise_and((sig[1:nbytesload:4] >> 3), 0x1f) +32*np.bitwise_and(sig[3:nbytesload:4] >> 3, 0x1f)
        # First signal is 7 msb of first byte and 3 lsb of second byte. 
        # Second signal is 7 msb of third byte and 3 lsb of forth byte
        # Third signal is 5 msb of second byte and 5 msb of forth byte
        
        if (nbytesload == nbytesstack):
            sig=np.vstack((s1,s2,s3)).flatten('F')
        else: # Append last samples that didn't completely fit into 4 byte blocks if any. Maximum 2. 
            if (nbytesload % 4 == 2): # Extra 1 sample 
                sig=np.hcat((np.vstack((s1,s2,s3)).flatten('F')), (sig[nbytesstack] >> 1) +1024*np.bitwise_and(sig[nbytesstack+1], 0x07))
            else: # Extra 2 samples
                sig=np.hcat((np.vstack((s1,s2,s3)).flatten('F')), (sig[nbytesstack] >> 1) +1024*np.bitwise_and(sig[nbytesstack+1], 0x07) , (sig[nbytesstack+2] >> 1) +1024*np.bitwise_and(sig[nbytesstack+3], 0x07))              
            
        sig[sig>511]-=1024 # convert to two's complement form (signed)
    elif fmt=='311': # Three 10 bit samples packed into 4 bytes with 2 bits discarded
        nbytesload=math.ceil((sampto-sampfrom)*nsig*4/3)
        nbytesstack=int(nbytesload/4)*4 # Most samples come in 4 byte (32 bit) blocks. Possibly remainders
        
        sig=np.fromfile(fp, dtype=np.dtype(datatypes[fmt]), count=nbytesload)
        s1=sig[0::4] +256*np.bitwise_and(sig[1::4], 0x03) 
        s2=np.bitwise_and(sig[1::4] >> 2, 0x3f) +64*np.bitwise_and(sig[2::4], 0x0f) 
        s3=np.bitwise_and((sig[2::4] >> 4), 0x0f) +16*np.bitwise_and(sig[3::4], 0x3f)
        # First signal is first byte and 2 lsb of second byte. 
        # Second signal is 6 msb of second byte and 4 lsb of third byte
        # Third signal is 4 msb of third byte and 6 lsb of forth byte
        if (nbytesload == nbytesstack):
            sig=np.vstack((s1,s2,s3)).flatten('F')
        else:
            if (nbytesload % 4 == 2): # Extra 1 sample 
                sig=np.hcat((np.vstack((s1,s2,s3)).flatten('F')), sig[nbytesstack] +256*np.bitwise_and(sig[nbytesstack+1], 0x03))
            else: # Extra 2 samples
                sig=np.hcat((np.vstack((s1,s2,s3)).flatten('F')), sig[nbytesstack] +256*np.bitwise_and(sig[nbytesstack+1], 0x03), np.bitwise_and(sig[nbytesstack+1] >> 2, 0x3f) +64*np.bitwise_and(sig[nbytesstack+2], 0x0f) )
                  
        sig[sig>511]-=1024 # convert to two's complement form (signed)
    else: # Simple format signals that can be loaded as they are stored. 
        sig=np.fromfile(fp, dtype=np.dtype(datatypes[fmt]), count=(sampto-sampfrom)*nsig)
        # Correct byte offset format data
        if fmt=='80':
            sig=sig.astype(int)
            sig=sig-128
        elif fmt=='160':
            sig=sig.astype(int)
            sig=sig-32768
            
    sig=sig.reshape(sampto-sampfrom, nsig)
        
        
    return sig

