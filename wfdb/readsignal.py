### Written by - Chen Xie 2016 ### 


# rdsamp - Read a WFDB file and return the signal as a numpy array and the metadata as a dictionary. 

## Usage: 
# sig, fields = rdsamp(recordname, sampfrom, sampto, channels, physical, stacksegments) 

#Input Arguments: 
# recordname (mandatory) - The name of the WFDB record to be read (without any file extensions).
# sampfrom (default=0) - The starting sample number to read for each channel.
# sampto (default=length of entire signal)- The final sample number to read for each channel.
# channels (default=all channels) - Indices specifying the channel to be returned.
# physical (default=1) - Flag that specifies whether to return signals in physical (1) or digital (0) units.
# stacksegments (default=1) - Flag used only for multi-segment files. Specifies whether to return the signal as a single stacked/concatenated numpy array (1) or as a list of one numpy array for each segment (0). 


# Output Arguments:
# sig - An nxm numpy array where n is the signal length and m is the number of channels. If the input record is a multi-segment record, depending on the input stacksegments flag, sig will either be a single stacked/concatenated numpy array (1) or a list of one numpy array for each segment (0). For empty segments, stacked format will contain Nan values, and non-stacked format will contain a single integer specifying the length of the empty segment.

# fields - A dictionary of metadata about the record extracted or deduced from the header/signal file. If the input record is a multi-segment record, the output argument will be a list of dictionaries:
#    -The first list element will be a dictionary of metadata about the master header.
#    -If the record is in variable layout format, the next list element will be a dictionary of metadata about the layout specification header.
#    -The last list element will be a list of dictionaries of metadata for each segment. For empty segments, the dictionary will be replaced by a single string: 'Empty Segment'


import numpy as np
import re
import os
import math
import sys

def rdsamp(recordname, sampfrom=0, sampto=[], channels=[], physical=1, stacksegments=1):
    fields=readheader(recordname) # Get the info from the header file
    if fields["nsig"]==0:
        sys.exit("This record has no signals. Use rdann to read annotations")
    if sampfrom<0:
        sys.exit("sampfrom must be non-negative")
    dirname, baserecordname=os.path.split(recordname)
    if dirname:
        dirname=dirname+"/"
    
    if fields["nseg"]==1: # single segment file
        if (len(set(fields["filename"]))==1): # single dat (or binary) file in the segment
            if not fields["nsamp"]: # Signal length was not specified in the header, calculate it from the file size.
                bytespersample={'8': 1, '16': 2, '24': 3, '32': 4, '61': 2, 
                    '80': 1, '160':2, '212': 1.5, '310': 4/3, '311': 4/3}
                filesize=os.path.getsize(dirname+fields["filename"][0])
                fields["nsamp"]=int((filesize-fields["byteoffset"][0])/fields["nsig"]/bytespersample[fields["fmt"][0]]) 
            if sampto:
                if sampto>fields["nsamp"]:
                    sys.exit("sampto exceeds length of record: ", fields["nsamp"])
            else:
                sampto=fields["nsamp"]
            sig=readdat(dirname+fields["filename"][0], fields["fmt"][0], fields["byteoffset"][0], sampfrom, sampto, fields["nsig"], fields["nsamp"])
            
            if channels: # User input channels
                if channels!=list(range(0, fields["nsig"])): # If input channels are not equal to the full set of ordered channels, remove the non-included ones and reorganize the remaining ones that require it. 
                    sig=sig[:, channels]
                 
                    # Rearrange the channel dependent fields
                    arrangefields=["filename", "fmt", "byteoffset", "gain", "units", "baseline", "initvalue", "signame"]
                    for fielditem in arrangefields:
                        fields[fielditem]=[fields[fielditem][ch] for ch in channels]
                    fields["nsig"]=len(channels) # Number of signals. 
                    
            if (physical==1):
                # Insert nans/invalid samples. 
                # Values that correspond to NAN (Format 8 has no NANs)
                wfdbInvalids={'16': -32768, '24': -8388608, '32': -2147483648, '61': -32768, 
                              '80': -128, '160':-32768, '212': -2048, '310': -512, '311': -512} 
                sig=sig.astype(float)
                sig[sig==wfdbInvalids[fields["fmt"][0]]]=np.nan
                sig=np.subtract(sig, np.array([float(i) for i in fields["baseline"]]))
                sig=np.divide(sig, np.array([fields["gain"]]))
                
            
        else: # Multiple dat files in the segment. Read different dats and merge them channel wise. 
            
            if not channels: # Default all channels 
                channels=list(range(0, fields["nsig"]))
                
            # Get info to arrange channels correctly 
            filenames=[] # The unique dat files to open,
            filechannels={} # All the channels that each dat file contains
            filechannelsout={} # The channels for each dat file to be returned and the corresponding target output channels
            for ch in list(range(0, fields["nsig"])):
                sigfile=fields["filename"][ch]
                # Correspond every channel to a dat file
                if sigfile not in filechannels:
                    filechannels[sigfile]=[ch] 
                else:
                    filechannels[sigfile].append(ch)
                # Get only relevant files/channels to load
                if ch in channels:
                    if sigfile not in filenames: # Newly encountered dat file
                        filenames.append(sigfile) 
                        filechannelsout[sigfile]=[[filechannels[sigfile].index(ch), channels.index(ch)]]
                    else:
                        filechannelsout[sigfile].append([filechannels[sigfile].index(ch), channels.index(ch)])
              
            # Allocate final output array
            if not fields["nsamp"]: # header doesn't have signal length. Figure it out from the first dat to read. 
                
                # Bytes required to hold each sample (including wasted space)
                bytespersample={'8': 1, '16': 2, '24': 3, '32': 4, '61': 2, 
                                '80': 1, '160':2, '212': 1.5, '310': 4/3, '311': 4/3}       
                filesize=os.path.getsize(dirname+filenames[0]) 
                fields["nsamp"]=int((filesize-fields["byteoffset"][channels[0]])/len(filechannels[filenames[0]])/bytespersample[fields["fmt"][channels[0]]])
                
                
            if not sampto: # if sampto field is empty 
                sampto=fields["nsamp"]
            if physical==0:
                sig=np.empty([sampto-sampfrom, len(channels)], dtype='int')
            else:
                sig=np.empty([sampto-sampfrom, len(channels)]) 
                        
            # Rearrange the fields 
            arrangefields=["filename", "fmt", "byteoffset", "gain", "units", "baseline", "initvalue", "signame"]
            for fielditem in arrangefields:
                fields[fielditem]=[fields[fielditem][ch] for ch in channels]
            fields["nsig"]=len(channels) # Number of signals. 
            
            # Read signal files one at a time and store the relevant channels  
            for sigfile in filenames:
                # def readdat(filename, fmt, byteoffset, sampfrom, sampto, nsig, siglen)
                sig[:, [outchan[1] for outchan in filechannelsout[sigfile]]]=readdat(dirname+sigfile, fields["fmt"][fields["filename"].index(sigfile)], fields["byteoffset"][fields["filename"].index(sigfile)], sampfrom, sampto, len(filechannels[sigfile]), fields["nsamp"])[:, [datchan[0] for datchan in filechannelsout[sigfile]]] # Fix the byte offset...
                
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
                
    else: # Multi-segment file. Preprocess and recursively call rdsamp on single segments. 
        
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
        ################ Done obtaining segments to read and samples to read in each segment. 

       
        # Preprocess/preallocate according to chosen output format
        if not channels:
            channels=list(range(0, fields["nsig"])) # All channels
        if stacksegments==1: # Output a single concatenated numpy array
            # Figure out the dimensions of the entire signal
            if sampto: # User inputs sampto
                nsamp=sampto
            elif fields["nsamp"]: # master header has number of samples
                nsamp=fields["nsamp"]
            else: # Have to figure out length by adding up all segment lengths
                nsamp=sum(fields["nsampseg"][startseg:])
            nsamp=nsamp-sampfrom
            if physical==0:
                sig=np.empty((nsamp, len(channels)), dtype='int')
            else:
                sig=np.empty((nsamp, len(channels)), dtype='float')
            
            indstart=0 # The overall signal index of the start of the current segment for filling in the large np array
        else: # Output a list of numpy arrays
            sig=[None]*len(readsegs)  # Empty list for storing segment signals. 
        segmentfields=[None]*len(readsegs) # List for storing segment fields. 
     
    
        # Read and store segments one at a time.
        for i in [r+startseg for r in readsegs]: # i is the segment number. It accounts for the layout record if exists and skips past it.    
            segrecordname=fields["filename"][i] 

            # Work out the relative channels to return from each segment 
            if startseg==0: # Fixed layout signal. Channels for record are always same. 
                segchannels=channels
            else: # Variable layout signal. Work out which channels from the segment to load if any. 
                if segrecordname!='~':
                    sfields=readheader(dirname+segrecordname)
                    wantsignals=[layoutfields["signame"][c] for c in channels] # Signal names of wanted channels
                    segchannels=[] # The channels wanted that are contained in the segment
                    returninds=[] # 1 and 0 marking channels of the numpy array to be filled by the returned segment channels 
                    for ws in wantsignals:
                        if ws in sfields["signame"]:
                            segchannels.append(sfields["signame"].index(ws))
                            returninds.append(1)
                        else:
                            returninds.append(0)
                    returninds=np.array(returninds)
                    emptyinds=np.where(returninds==0)[0] # The channels of the overall array that the segment doesn't contain
                    returninds=np.where(returninds==1)[0] # The channels of the overall array that the segment contains
                else:
                    segchannels=[]

            if stacksegments==0: # Return list of np arrays
                if (segrecordname=='~')|(not segchannels): # Empty segment or wrong channels. Store indicator and segment length. 
                    #sig[i-startseg-readsegs[0]]=fields["nsampseg"][i] # store the segment length
                    sig[i-startseg-readsegs[0]]=readsamps[i-startseg-readsegs[0]][1]-readsamps[i-startseg-readsegs[0]][0]
                    segmentfields[i-startseg-readsegs[0]]="Empty Segment" 
                    
                else: # Non-empty segment that contains wanted channels. Read its signal and header fields
                    sig[i-startseg-readsegs[0]], segmentfields[i-startseg-readsegs[0]] = rdsamp(recordname=dirname+segrecordname, physical=physical, sampfrom=readsamps[i-startseg-readsegs[0]][0], sampto=readsamps[i-startseg-readsegs[0]][1], channels=segchannels)

            else: # Return single stacked np array of all (selected) channels 
                indend=indstart+readsamps[i-startseg-readsegs[0]][1]-readsamps[i-startseg-readsegs[0]][0] # end index of the large array for this segment
                if (segrecordname=='~')|(not segchannels) : # Empty segment or no wanted channels: fill in invalids
                    if physical==0:
                        sig[indstart:indend, :] = sig[indstart:indend, emptyinds]=-2147483648
                    else:
                        sig[indstart:indend, :] = np.nan 
                    segmentfields[i-startseg-readsegs[0]]="Empty Segment"
                else: # Non-empty segment - Get samples
                    if startseg==1: # Variable layout format. Load data then rearrange channels. 
   
                        sig[indstart:indend, returninds], segmentfields[i-startseg-readsegs[0]] = rdsamp(recordname=dirname+segrecordname, physical=physical, sampfrom=readsamps[i-startseg-readsegs[0]][0], sampto=readsamps[i-startseg-readsegs[0]][1], channels=segchannels) # Load all the wanted channels that the segment contains
                        if physical==0: # Fill the rest with invalids
                            sig[indstart:indend, emptyinds]=-2147483648
                        else:
                            sig[indstart:indend, emptyinds]=np.nan 

                        # Expand the channel dependent fields to match the overall layout. Remove this following block if you wish to keep the returned channels' fields without any 'no channel' placeholders between. 
                        arrangefields=["filename", "fmt", "byteoffset", "gain", "units", "baseline", "initvalue", "signame"]
                        expandedfields=dict.copy(segmentfields[i-startseg-readsegs[0]]) 
                        for fielditem in arrangefields:
                            expandedfields[fielditem]=['No Channel']*len(channels)
                            for c in range(0, len(returninds)):
                                expandedfields[fielditem][returninds[c]]=segmentfields[i-startseg-readsegs[0]][fielditem][c] 
                            segmentfields[i-startseg-readsegs[0]][fielditem]=expandedfields[fielditem]
                        # Keep fields['nsig'] as the value of returned channels from the segments. 
 
                    else: # Fixed layout - channels already arranged                 
                        sig[indstart:indend, :] , segmentfields[i-startseg] = rdsamp(recordname=dirname+segrecordname, physical=physical, sampfrom=readsamps[i-startseg-readsegs[0]][0], sampto=readsamps[i-startseg-readsegs[0]][1], channels=segchannels) 
                indstart=indend # Update the start index for filling in the next part of the array

        # Done reading all segments
             
        # Return a list of fields. First element is the master, next is layout if any, last is a list of all the segment fields. 
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
    if nsamp:
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

# Read samples from a WFDB binary file 
def readdat(filename, fmt, byteoffset, sampfrom, sampto, nsig, siglen): 
    # nsig and siglen define whole file, not selected inputs. siglen is signal length in samples. 
   
    # Bytes required to hold each sample (including wasted space)
    bytespersample={'8': 1, '16': 2, '24': 3, '32': 4, '61': 2, 
                    '80': 1, '160':2, '212': 1.5, '310': 4/3, '311': 4/3}
    
    # Data type objects for each format to load. Doesn't directly correspond for final 3 formats. 
    datatypes={'8':'<i1', '16':'<i2', '24':'<i3', '32':'<i4', 
               '61':'>i2', '80':'<u1', '160':'<u2', 
               '212':'<u1', '310':'<u1', '311':'<u2'} 
    if not sampto:
        
        #if not siglen: # Signal length was not obtained, calculate it from the file size.   # Move this to rdsamp 
        #    filesize=os.path.getsize(filename) 
        #    siglen=(filesize-byteoffset)/nsig/bytespersample[fmt] 
        
        sampto=siglen
    
    fp=open(filename,'rb')
    
    fp.seek(int(sampfrom*nsig*bytespersample[fmt])+int(byteoffset)) # Point to the starting sample 
    
    # Reading the dat file into np array and reshaping. 
    if fmt=='212': # 212, 310, and 311 need special loading and processing. 
        nbytesload=int(math.ceil((sampto-sampfrom)*nsig*1.5)) # Loaded bytes
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

