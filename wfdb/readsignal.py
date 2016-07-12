import numpy as np
import re
import os
import math
import sys

def rdsamp(recordname, sampfrom=0, sampto=[], physical=1, stacksegments=0):
    # to do: add channel selection, to and from
    #print("rdsamp calling on: "+recordname)
    fields=readheader(recordname) # Get the info from the header file
    if fields["nsig"]==0:
        sys.exit("This record has no signals. Use rdann to read annotations")
    if sampfrom<0:
        sys.exit("sampfrom must be non-negative")
    dirname, baserecordname=os.path.split(recordname)
    if dirname:
        dirname=dirname+"/"
    
    if fields["nseg"]==1: # single segment file
        if (len(set(fields["filename"]))==1): # single dat file in the segment
            sig=readdat(dirname+fields["filename"][0], fields["fmt"][0], fields["byteoffset"][0], sampfrom, sampto, fields["nsig"], fields["nsamp"])
            
            if (physical==1):
                # Insert nans/invalid samples. 
                # Values that correspond to NAN (Format 8 has no NANs)
                wfdbInvalids={'16': -32768, '24': -8388608, '32': -2147483648, '61': -32768, 
                              '80': -128, '160':-32768, '212': -2048, '310': -512, '311': -512} 
                
                sig=sig.astype(float)
                sig[sig==wfdbInvalids[fields["fmt"][0]]]=np.nan

                sig=np.subtract(sig, np.array([float(i) for i in fields["baseline"]]))
                sig=np.divide(sig, np.array([fields["gain"]]))
            
        else: # Multiple dat files in the segment. Read different dats and glue them together channel wise. 
            
            # Get unique dat files to open, and the number of channels each of them contain
            filenames=[]
            channelsperfile={}
            for chanfile in fields["filename"]:
                if chanfile not in filenames:
                    filenames.append(chanfile)
                    channelsperfile[chanfile]=1
                else:
                    channelsperfile[chanfile]=channelsperfile[chanfile]+1
            
            # Allocate array for all channels
            if not fields["nsamp"]: # header doesn't have signal length. Figure it out from the first dat. 
                # Bytes required to hold each sample (including wasted space)
                bytespersample={'8': 1, '16': 2, '24': 3, '32': 4, '61': 2, 
                                '80': 1, '160':2, '212': 1.5, '310': 4/3, '311': 4/3}
                filesize=os.path.getsize(dirname+filenames[0]) 
                fields["nsamp"]=filesize/channelsperfile[fields["filename"][0]]/bytespersample[fields["fmt"][0]] # CONSIDER BYTE OFFSET! Using fields["nsamp"] later
            
            if not sampto: #if it is empty
                sampto=fields["nsamp"]
            sig=np.empty([sampto-sampfrom, fields["nsig"]])
            
            # Read signals and store them in array
            fillchannels=0
            for sigfiles in filenames:
                # def readdat(filename, fmt, byteoffset, sampfrom, sampto, nsig, siglen)
                sig[:,fillchannels:fillchannels+channelsperfile[sigfiles]]=readdat(dirname+sigfiles, fields["fmt"][fillchannels], fields["byteoffset"][fillchannels], sampfrom, sampto, channelsperfile[sigfiles], fields["nsamp"]) # Fix the byte offset
                fillchannels=fillchannels+channelsperfile[sigfiles]
                
            if (physical==1):
                # Insert nans/invalid samples. 
                # Values that correspond to NAN (Format 8 has no NANs)
                wfdbInvalids={'16': -32768, '24': -8388608, '32': -2147483648, '61': -32768, 
                              '80': -128, '160':-32768, '212': -2048, '310': -512, '311': -512} 
                sig=sig.astype(float)
                for ch in range(0, fields["nsig"]):
                    sig[sig[:,ch]==wfdbInvalids[fields["fmt"][ch]], ch] =np.nan
                sig=np.subtract(sig, np.array([float(b) for b in fields["baseline"]]))
                sig=np.divide(sig, np.array([fields["gain"]]))
                
    else: # Multi-segment file
        
        # Determine if this record is fixed or variable layout. startseg is the first signal segment.  
        if fields["nsampseg"][0]==0: # variable layout - first segment is layout specification file 
            startseg=1
            layoutfields=readheader(dirname+fields["filename"][0]) # Store the layout header info. 
        else: # fixed layout - no layout specification file. 
            startseg=0
        
        # Determine the segments and samples that have to be read based on sampfrom and sampto  
        cumsumlengths=list(np.cumsum(fields["nsampseg"][startseg:])) # Cumulative sum of segment lengths
        
        if not sampto:
            sampto=cumsumlengths[len(cumsumlengths)-1]
        
        if sampto>cumsumlengths[len(cumsumlengths)-1]: # Error check sampto
            sys.exit("sampto exceeds length of record: ", cumsumlengths[len(cumsumlengths)-1])
            
        readsegs=[[sampfrom<cs for cs in cumsumlengths].index(True)] # First segment
        
        if sampto==cumsumlengths[len(cumsumlengths)-1]: 
            readsegs.append(len(cumsumlengths)-1) # Final segment
        else:
            readsegs.append([sampto<cs for cs in cumsumlengths].index(True))
            
        if readsegs[1]==readsegs[0]: # Only one segment to read
            readsegs=[readsegs[0]]
            readsamps=[[sampfrom, sampto]] # The sampfrom and sampto for each segment
        else:
            readsegs=list(range(readsegs[0], readsegs[1]+1)) # Expand into list 
            readsamps=[[0, fields["nsampseg"][s+startseg]] for s in readsegs] # The sampfrom and sampto for each segment. Most are [0, nsampseg]

            readsamps[0][0]=sampfrom-([0]+cumsumlengths)[readsegs[0]] # Starting sample for first segment

            readsamps[len(readsamps)-1][1]=sampto-([0]+cumsumlengths)[readsegs[len(readsamps)-1]] # End sample for last segment 
         
        print("readsegs:", readsegs)  
        print("readsamps:", readsamps)
        print("\n\n\n")
        
        ################ Done processing segments and samples to read in each segment. 
        

        # Preprocess/preallocate according to chosen output format
        if stacksegments==1: # Output a single concatenated numpy array
            # Figure out the dimensions of the entire signal
            if sampto: # User inputs sampto
                nsamp=sampto
            elif fields["nsamp"]: # master header has number of samples
                nsamp=fields["nsamp"]
            else: # Have to figure out length by adding up all segment lengths
                nsamp=sum(fields["nsampseg"][startseg:])
            nsamp=nsamp-sampfrom
            # if user inputs channels:
                #nsig=len(channels)
            #else:
            nsig=fields["nsig"] # Number of signals read from master header
            sig=np.empty((nsamp, nsig))
            indstart=0 # The overall signal index of the start of the current segment for filling in the large np array
            
        else: # Output a list of numpy arrays
            sig=[None]*len(readsegs)  # Empty list for storing segment signals. 
        segmentfields=[None]*len(readsegs) # List for storing segment fields. 
        
        print("fields[filename]: " , fields["filename"])
        print("startseg: ", startseg, "\n\n\n")
        
        
        
        # Read segments one at a time.
        for i in [r+startseg for r in readsegs]: # i is the segment number. It accounts for the layout record if exists and skips past it.    
            segrecordname=fields["filename"][i] 

            if stacksegments==0: # Return list of np arrays
                if (segrecordname=='~'): # Empty segment. Store indicator and segment length. 
                    #sig[i-startseg-readsegs[0]]=fields["nsampseg"][i] # store the segment length
                    sig[i-startseg-readsegs[0]]=readsamps[i-startseg-readsegs[0]][1]-readsamps[i-startseg-readsegs[0]][0]
                    segmentfields[i-startseg-readsegs[0]]="Empty Segment" 
                else: # Non-empty segment. Read its signal and header fields
                    sig[i-startseg-readsegs[0]], segmentfields[i-startseg-readsegs[0]] = rdsamp(recordname=dirname+segrecordname, physical=physical, sampfrom=readsamps[i-startseg-readsegs[0]][0], sampto=readsamps[i-startseg-readsegs[0]][1])

            else: # Return single stacked np array of all (selected) channels 
                indend=indstart+readsamps[i-startseg-readsegs[0]][1]-readsamps[i-startseg-readsegs[0]][0] # end index of the large array for this segment
                
                if (segrecordname=='~'): # Empty segment: fill in nans
                    #sig[indstart:indstart+fields["nsampseg"][i], :] = np.nan # channel selection later... 
                    sig[indstart:indend, :] = np.nan # channel selection later... 
                    segmentfields[i-startseg-readsegs[0]]="Empty Segment"
                else: # Non-empty segment - Get samples
                    if startseg==1: # Variable layout format. Load data then rearrange channels. 
                        segmentsig, segmentfields[i-startseg-readsegs[0]]= rdsamp(recordname=dirname+segrecordname, physical=physical, sampfrom=readsamps[i-startseg-readsegs[0]][0], sampto=readsamps[i-startseg-readsegs[0]][1])
                        
                        for ch in range(0, layoutfields["nsig"]): # Fill each channel with signal or nan
                            
                            # The segment contains the channel
                            if layoutfields["signame"][ch] in segmentfields[i-startseg-readsegs[0]]["signame"]:
                                sig[indstart:indend, ch] = segmentsig[:, segmentfields[i-startseg-readsegs[0]]["signame"].index(layoutfields["signame"][ch])]  
                                
                            else: # The segment doesn't contain the channel. Fill in nans
                                sig[indstart:indend, ch] = np.nan
                                
                    else: # Fixed layout - channels already arranged                 
                        sig[indstart:indend, :] , segmentfields[i-startseg] = rdsamp(recordname=dirname+segrecordname, physical=physical, sampfrom=readsamps[i-startseg-readsegs[0]][0], sampto=readsamps[i-startseg-readsegs[0]][1]) 

                indstart=indend # Update the start index for filling in the next part of the array
                
                
        # Done reading individual segments
             
            

            
        # Return a list for the fields. The first element is the master, second is layout if any, last is a list of all the segment fields. 
        if startseg==1: # Variable layout format
            fields=[fields,layoutfields, segmentfields]
        else: # Fixed layout format. 
            fields=[fields, segmentfields] 
        
    return (sig, fields)



def readheader(recordname): # For reading signal headers

    # To do: Allow exponential input format for some fields 
    
    fid=open(recordname + ".hea", 'r')
    #print("readheader opening: "+recordname+".hea")
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
    rxRECORD=re.compile(''.join(["(?P<name>[\w]+)/?(?P<nseg>\d*)[ \t]+",
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

    # Regexp object for signal lines. Consider filenames - dat and mat or ~
    rxSIGNAL=re.compile(''.join(["(?P<filename>[\w]*\.?[\w]*~?)[ \t]+(?P<format>\d+)x?"
                                 "(?P<samplesperframe>\d*):?(?P<skew>\d*)\+?(?P<byteoffset>\d*)[ \t]*",
                                 "(?P<ADCgain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^/-]*)[ \t]*",
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
    
    # These fields are either mandatory or set to defaults. 
    fields['nseg']=int(nseg) 
    fields['fs']=float(fs)
    fields['nsig']=int(nsig) 
    
    # These fields might by empty
    fields['nsamp']=int(nsamp) 
    fields['basetime']=basetime
    fields['basedate']=basedate

    # Signal or Segment line paramters 

    if int(nseg)>1: # Multi segment header - Process segment spec lines in current master header.
        for i in range(0, int(nseg)):
            (filename, nsampseg)=re.findall('(?P<filename>\w*~?)[ \t]+(?P<nsampseg>\d+)', headerlines[i+1])[0]
            fields["filename"].append(filename) # filename might be ~ for null segment. 
            fields["nsampseg"].append(int(nsampseg)) # number of samples for the segment
    else: # Single segment header - Process signal spec lines in current regular header. 
        for i in range(0,int(nsig)): # will not run if nsignals=0
            # get signal line parameters
            (filename, fmt, sampsperframe, skew, byteoffset, adcgain, baseline, units, adcres,
             adczero, initvalue, checksum, blocksize, signame)=rxSIGNAL.findall(headerlines[i+1])[0]
            
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




def readdat(filename, fmt, byteoffset, sampfrom, sampto, nsig, siglen): 
    # nsig and nsamp define whole file, not selected inputs. nsamp is signal length. 
    
    # to do: allow channel choice too. Put that in rdsamp actually, not here.
    
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
            siglen=filesize/nsig/bytespersample[fmt] # NEED TO CONSIDER BYTE OFFSET
        sampto=siglen
    
    fp=open(filename,'rb')
    
    # Need to fix the byteoffset field for multi dat records.... 
    
    fp.seek(int(sampfrom*nsig*bytespersample[fmt])+int(byteoffset)) # Point to the starting sample 
    
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
    #print("Exiting readdat with: ")
    #print(sig)
        
    return sig

