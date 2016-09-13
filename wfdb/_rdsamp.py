import numpy as np
import re
import os
import sys
import requests

def dlrecordfiles(pbrecname, targetdir):
    """Check a specified local directory for all necessary files required to read a Physiobank
       record, and download any missing files into the same directory. Returns a list of files
       downloaded, or exits with error if an invalid Physiobank record is specified.

    Usage: dledfiles = dlrecordfiles(pbrecname, targetdir)

    Input arguments:
    - pbrecname (required): The name of the MIT format Physiobank record to be read, prepended
      with the Physiobank subdirectory the file is contained in (without any file extensions).
      eg. pbrecname=prcp/12726 to download files http://physionet.org/physiobank/database/prcp/12726.hea
      and 12727.dat
    - targetdir (required): The local directory to check for files required to read the record,
      in which missing files are also downloaded.

    Output arguments:
    - dledfiles:  The list of files downloaded from PhysioBank.

    """

    physioneturl = "http://physionet.org/physiobank/database/"
    pbdir, baserecname = os.path.split(pbrecname)
    displaydlmsg=1
    dledfiles = [] 
    
    if not os.path.isdir(targetdir):  # Make the target dir if it doesn't exist
        os.makedirs(targetdir)
        print("Created local directory: ", targetdir)
    
    # For any missing file, check if the input physiobank record name is
    # valid, ie whether the file exists on physionet. Download if valid, exit
    # if invalid.
    dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+baserecname+".hea", os.path.join(targetdir, 
        baserecname+".hea"), dledfiles, displaydlmsg, targetdir)
        
    fields = readheader(os.path.join(targetdir, baserecname))

    # Need to check validity of link if ANY file is missing.
    if fields["nseg"] == 1:  # Single segment. Check for all the required dat files
        for f in set(fields["filename"]):
            # Missing dat file
            dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+f, os.path.join(targetdir, f), 
                dledfiles, displaydlmsg, targetdir)
    else:  # Multi segment. Check for all segment headers and their dat files
        for segment in fields["filename"]:
            if segment != '~':
                # Check the segment header
                dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+segment+".hea", 
                    os.path.join(targetdir, segment+".hea"), dledfiles, displaydlmsg, targetdir)    
                segfields = readheader(os.path.join(targetdir, segment))
                for f in set(segfields["filename"]):
                    if f != '~':
                        # Check the segment's dat file
                        dledfiles, displaydlmsg = dlifmissing(physioneturl+pbdir+"/"+f, 
                            os.path.join(targetdir, f), dledfiles, displaydlmsg, targetdir)
                            
    if dledfiles:
        print('Downloaded all missing files for record.')
    return dledfiles  # downloaded files


# Download a file if it is missing. Also error check 0 byte files.  
def dlifmissing(url, filename, dledfiles, displaydlmsg, targetdir):
    fileexists = os.path.isfile(filename)  
    if fileexists:
        # Likely interrupted download
        if os.path.getsize(filename)==0:
            try:
                input = raw_input
            except NameError:
                pass
            userresponse=input("Warning: File "+filename+" is 0 bytes.\n"
                "Likely interrupted download. Remove file and redownload? [y/n]: ")
            # override input for python 2 compatibility
            while userresponse not in ['y','n']:
                userresponse=input("Remove file and redownload? [y/n]: ")
            if userresponse=='y':
                os.remove(filename)
                dledfiles.append(dlorexit(url, filename, displaydlmsg, targetdir))
                displaydlmsg=0
            else:
                print("Skipping download.")
        # File is already present.
        else:
            print("File "+filename+" is already present.")
    else:
        dledfiles.append(dlorexit(url, filename, displaydlmsg, targetdir))
        displaydlmsg=0
    
    # If a file gets downloaded, displaydlmsg is set to 0. No need to print the message more than once. 
    return dledfiles, displaydlmsg
                 
    
# Download the file from the specified 'url' as the 'filename', or exit with warning.
def dlorexit(url, filename, displaydlmsg=0, targetdir=[]):
    if displaydlmsg: # We want this message to be called once for all files downloaded.
        print('Downloading missing file(s) into directory: {}'.format(targetdir))
    try:
        r = requests.get(url)
        with open(filename, "wb") as writefile:
            writefile.write(r.content)
        return filename
    except requests.HTTPError:
        sys.exit("Attempted to download invalid target file: " + url)


# Download files required to read a wfdb annotation.
def dlannfiles():
    return dledfiles


# Download all the records in a physiobank database.
def dlPBdatabase(database, targetdir):
    return dledfiles

# Read header file to get comment and non-comment lines
def getheaderlines(recordname):
    with open(recordname + ".hea", 'r') as fp:
        headerlines = []  # Store record line followed by the signal lines if any
        commentlines = []  # Comments
        for line in fp:
            line = line.strip()
            if line.startswith('#'):  # comment line
                commentlines.append(line)
            elif line:  # Non-empty non-comment line = header line.
                ci = line.find('#')
                if ci > 0:
                    headerlines.append(line[:ci])  # header line
                    # comment on same line as header line
                    commentlines.append(line[ci:])
                else:
                    headerlines.append(line)
    return headerlines, commentlines

def readheader(recordname):  # For reading signal headers

    # To do: Allow exponential input format for some fields

    # Output dictionary
    fields = {
        'nseg': [],
        'nsig': [],
        'fs': [],
        'nsamp': [],
        'basetime': [],
        'basedate': [],
        'filename': [],
        'fmt': [],
        'sampsperframe': [],
        'skew': [],
        'byteoffset': [],
        'gain': [],
        'units': [],
        'baseline': [],
        'initvalue': [],
        'signame': [],
        'nsampseg': [],
        'comments': []}
    # filename stores file names for both multi and single segment headers.
    # nsampseg is only for multi-segment


    # RECORD LINE fields (o means optional, delimiter is space or tab unless specified):
    # record name, nsegments (o, delim=/), nsignals, fs (o), counter freq (o, delim=/, needs fs),
    # base counter (o, delim=(), needs counter freq), nsamples (o, needs fs), base time (o),
    # base date (o needs base time).

    # Regexp object for record line
    rxRECORD = re.compile(
        ''.join(
            [
                "(?P<name>[\w]+)/?(?P<nseg>\d*)[ \t]+",
                "(?P<nsig>\d+)[ \t]*",
                "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<basecounter>\d*\.?\d*)\)?[ \t]*",
                "(?P<nsamples>\d*)[ \t]*",
                "(?P<basetime>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
                "(?P<basedate>\d{,2}/?\d{,2}/?\d{,4})"]))
    # Watch out for potential floats: fs (and also exponent notation...),
    # counterfs, basecounter

    # SIGNAL LINE fields (o means optional, delimiter is space or tab unless specified):
    # file name, format, samplesperframe(o, delim=x), skew(o, delim=:), byteoffset(o,delim=+),
    # ADCgain(o), baseline(o, delim=(), requires ADCgain), units(o, delim=/, requires baseline),
    # ADCres(o, requires ADCgain), ADCzero(o, requires ADCres), initialvalue(o, requires ADCzero),
    # checksum(o, requires initialvalue), blocksize(o, requires checksum),
    # signame(o, requires block)

    # Regexp object for signal lines. Consider flexible filenames, and also ~
    rxSIGNAL = re.compile(
        ''.join(
            [
                "(?P<filename>[\w]*\.?[\w]*~?)[ \t]+(?P<format>\d+)x?"
                "(?P<sampsperframe>\d*):?(?P<skew>\d*)\+?(?P<byteoffset>\d*)[ \t]*",
                "(?P<ADCgain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^/-]*)[ \t]*",
                "(?P<ADCres>\d*)[ \t]*(?P<ADCzero>-?\d*)[ \t]*(?P<initialvalue>-?\d*)[ \t]*",
                "(?P<checksum>-?\d*)[ \t]*(?P<blocksize>\d*)[ \t]*(?P<signame>[\S]*)"]))

    # Units characters: letters, numbers, /, ^, -,
    # Watch out for potentially negative fields: baseline, ADCzero, initialvalue, checksum,
    # Watch out for potential float: ADCgain.

    # Read the header file and get the comment and non-comment lines
    headerlines, commentlines = getheaderlines(recordname)

    # Get record line parameters
    (_, nseg, nsig, fs, counterfs, basecounter, nsamp,
    basetime, basedate) = rxRECORD.findall(headerlines[0])[0]

    # These fields are either mandatory or set to defaults.
    if not nseg:
        nseg = '1'
    if not fs:
        fs = '250'

    fields['nseg'] = int(nseg)
    fields['fs'] = float(fs)
    fields['nsig'] = int(nsig)

    # These fields might by empty
    if nsamp:
        fields['nsamp'] = int(nsamp)
    fields['basetime'] = basetime
    fields['basedate'] = basedate


    # Signal or Segment line paramters
    # Multi segment header - Process segment spec lines in current master
    # header.
    if int(nseg) > 1:
        for i in range(0, int(nseg)):
            (filename, nsampseg) = re.findall(
                '(?P<filename>\w*~?)[ \t]+(?P<nsampseg>\d+)', headerlines[i + 1])[0]
            fields["filename"].append(filename)
            fields["nsampseg"].append(int(nsampseg))
    # Single segment header - Process signal spec lines in regular header.
    else:
        for i in range(0, int(nsig)):  # will not run if nsignals=0
            # get signal line parameters
            (filename,
            fmt,
            sampsperframe,
            skew,
            byteoffset,
            adcgain,
            baseline,
            units,
            adcres,
            adczero,
            initvalue,
            checksum,
            blocksize,
            signame) = rxSIGNAL.findall(headerlines[i + 1])[0]
            
            # Setting defaults
            if not sampsperframe:
                # Setting strings here so we can always convert strings case
                # below.
                sampsperframe = '1'
            if not skew:
                skew = '0'
            if not byteoffset:
                byteoffset = '0'
            if not adcgain:
                adcgain = '200'
            if not baseline:
                if not adczero:
                    baseline = '0'
                else:
                    baseline = adczero  # missing baseline actually takes adczero value if present
            if not units:
                units = 'mV'
            if not initvalue:
                initvalue = '0'
            if not signame:
                signame = "ch" + str(i + 1)
            if not initvalue:
                initvalue = '0'

            fields["filename"].append(filename)
            fields["fmt"].append(fmt)
            fields["sampsperframe"].append(int(sampsperframe))
            fields["skew"].append(int(skew))
            fields['byteoffset'].append(int(byteoffset))
            fields["gain"].append(float(adcgain))
            fields["baseline"].append(int(baseline))
            fields["units"].append(units)
            fields["initvalue"].append(int(initvalue))
            fields["signame"].append(signame)

    for comment in commentlines:
        fields["comments"].append(comment.strip('\s#'))

    return fields



def skewsignal(sig, skew, fp, nsig, fmt, siglen, sampfrom, sampto, startbyte, 
    nbytesread, byteoffset, sampsperframe, tsampsperframe):
    if max(skew) > 0:
        # Array of samples to fill in the final samples of the skewed channels.
        extrasig = np.empty([max(skew), nsig])
        extrasig.fill(wfdbInvalids[fmt])

        # Load the extra samples if the end of the file hasn't been reached.
        if siglen - (sampto - sampfrom):
            startbyte = startbyte + nbytesread
            # Point the the file pointer to the start of a block of 3 or 4 and
            # keep track of how many samples to discard after reading. For
            # regular formats the file pointer is already at the correct
            # location.
            if fmt == '212':
                # Extra samples to read
                floorsamp = (startbyte - byteoffset) % 3
                # Now the byte pointed to is the first of a byte triplet
                # storing 2 samples. It happens that the extra samples match
                # the extra bytes for fmt 212
                startbyte = startbyte - floorsamp
            elif (fmt == '310') | (fmt == '311'):
                floorsamp = (startbyte - byteoffset) % 4
                # Now the byte pointed to is the first of a byte quartet
                # storing 3 samples.
                startbyte = startbyte - floorsamp
            startbyte = startbyte
            fp.seek(startbyte)
            # The length of extra signals to be loaded
            extraloadlen = min(siglen - (sampto - sampfrom), max(skew))
            nsampextra = extraloadlen * tsampsperframe
            extraloadedsig = processwfdbbytes(
                fp,
                fmt,
                extraloadlen,
                nsig,
                sampsperframe,
                floorsamp)[0]  # Array of extra loaded samples
            # Fill in the extra loaded samples
            extrasig[:extraloadedsig.shape[0], :] = extraloadedsig

        # Fill in the skewed channels with the appropriate values.
        for ch in range(0, nsig):
            if skew[ch] > 0:
                sig[:-skew[ch], ch] = sig[skew[ch]:, ch]
                sig[-skew[ch]:, ch] = extrasig[:skew[ch], ch]
    return sig

# Get samples from a WFDB binary file
def readdat(
        filename,
        fmt,
        byteoffset,
        sampfrom,
        sampto,
        nsig,
        siglen,
        sampsperframe,
        skew):
    # nsig defines whole file, not selected channels. siglen refers to signal
    # length of whole file, not selected duration.
    # Recall that channel selection is not performed in this function but in
    # rdsamp.

    tsampsperframe = sum(sampsperframe)  # Total number of samples per frame

    # Figure out the starting byte to read the dat file from. Special formats
    # store samples in specific byte blocks.
    startbyte = int(sampfrom * tsampsperframe *
                    bytespersample[fmt]) + int(byteoffset)
    floorsamp = 0
    # Point the the file pointer to the start of a block of 3 or 4 and keep
    # track of how many samples to discard after reading.
    if fmt == '212':
        floorsamp = (startbyte - byteoffset) % 3  # Extra samples to read
        # Now the byte pointed to is the first of a byte triplet storing 2
        # samples. It happens that the extra samples match the extra bytes for
        # fmt 212
        startbyte = startbyte - floorsamp
    elif (fmt == '310') | (fmt == '311'):
        floorsamp = (startbyte - byteoffset) % 4
        # Now the byte pointed to is the first of a byte quartet storing 3
        # samples.
        startbyte = startbyte - floorsamp

    fp = open(filename, 'rb')

    fp.seek(startbyte)  # Point to the starting sample
    # Read the dat file into np array and reshape.
    sig, nbytesread = processwfdbbytes(
        fp, fmt, sampto - sampfrom, nsig, sampsperframe, floorsamp)

    # Shift the samples in the channels with skew if any
    sig=skewsignal(sig, skew, fp, nsig, fmt, siglen, sampfrom, sampto, startbyte, 
        nbytesread, byteoffset, sampsperframe, tsampsperframe)

    fp.close()

    return sig


# Read data from a wfdb dat file and process it to obtain digital samples.
# Returns the signal and the number of bytes read.
def processwfdbbytes(fp, fmt, siglen, nsig, sampsperframe, floorsamp=0):
    # siglen refers to the length of the signal to be read. Different from siglen input argument for readdat.
    # floorsamp is the extra sample index used to read special formats.

    tsampsperframe = sum(sampsperframe)  # Total number of samples per frame.
    # Total number of signal samples to be collected (including discarded ones)
    nsamp = siglen * tsampsperframe + floorsamp

    # Reading the dat file into np array and reshaping. Formats 212, 310, and 311 need special processing.
    # Note that for these formats with multi samples/frame, have to convert
    # bytes to samples before returning average frame values.
    if fmt == '212':
        # The number of bytes needed to be loaded given the number of samples
        # needed
        nbytesload = int(np.ceil((nsamp) * 1.5))
        sigbytes = np.fromfile(
            fp,
            dtype=np.dtype(
                datatypes[fmt]),
            count=nbytesload).astype('uint')  # Loaded as unsigned 1 byte blocks

        if tsampsperframe == nsig:  # No extra samples/frame
            # Turn the bytes into actual samples.
            sig = np.zeros(nsamp)  # 1d array of actual samples
            # One sample pair is stored in one byte triplet.
            sig[0::2] = sigbytes[0::3] + 256 * \
                np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples
            if len(sig > 1):
                # Odd numbered samples
                sig[1::2] = sigbytes[2::3] + 256 * \
                    np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]
            # Reshape into final array of samples
            sig = sig.reshape(siglen, nsig)
            sig = sig.astype(int)
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^11-1 are negative.
            sig[sig > 2047] -= 4096
        else:  # At least one channel has multiple samples per frame. All extra samples are discarded.
            # Turn the bytes into actual samples.
            sigall = np.zeros(nsamp)  # 1d array of actual samples
            sigall[0::2] = sigbytes[0::3] + 256 * \
                np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples

            if len(sigall) > 1:
                # Odd numbered samples
                sigall[1::2] = sigbytes[2::3] + 256 * \
                    np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]
            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^11-1 are negative.
            sigall[sigall > 2047] -= 4096
            # Give the average sample in each frame for each channel
            sig = np.zeros([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

    elif fmt == '310':  # Three 10 bit samples packed into 4 bytes with 2 bits discarded

        # The number of bytes needed to be loaded given the number of samples
        # needed
        nbytesload = int(((nsamp) + 2) / 3.) * 4
        if (nsamp - 1) % 3 == 0:
            nbytesload -= 2
        sigbytes = np.fromfile(
            fp,
            dtype=np.dtype(
                datatypes[fmt]),
            count=nbytesload).astype('uint')  # Loaded as unsigned 1 byte blocks
        if tsampsperframe == nsig:  # No extra samples/frame
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sig = np.zeros(nsamp)

            sig[0::3] = (sigbytes[0::4] >> 1)[0:len(sig[0::3])] + 128 * \
                np.bitwise_and(sigbytes[1::4], 0x07)[0:len(sig[0::3])]
            if len(sig > 1):
                sig[1::3] = (sigbytes[2::4] >> 1)[0:len(sig[1::3])] + 128 * \
                    np.bitwise_and(sigbytes[3::4], 0x07)[0:len(sig[1::3])]
            if len(sig > 2):
                sig[2::3] = np.bitwise_and((sigbytes[1::4] >> 3), 0x1f)[0:len(
                    sig[2::3])] + 32 * np.bitwise_and(sigbytes[3::4] >> 3, 0x1f)[0:len(sig[2::3])]
            # First signal is 7 msb of first byte and 3 lsb of second byte.
            # Second signal is 7 msb of third byte and 3 lsb of forth byte
            # Third signal is 5 msb of second byte and 5 msb of forth byte

            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]
            # Reshape into final array of samples
            sig = sig.reshape(siglen, nsig)
            # Convert to int64 to be able to hold -ve values
            sig = sig.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sig[sig > 511] -= 1024

        else:  # At least one channel has multiple samples per frame. All extra samples are averaged.
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sigall = np.zeros(nsamp)
            sigall[0::3] = (sigbytes[0::4] >> 1)[0:len(
                sigall[0::3])] + 128 * np.bitwise_and(sigbytes[1::4], 0x07)[0:len(sigall[0::3])]
            if len(sigall > 1):
                sigall[1::3] = (sigbytes[2::4] >> 1)[0:len(
                    sigall[1::3])] + 128 * np.bitwise_and(sigbytes[3::4], 0x07)[0:len(sigall[1::3])]
            if len(sigall > 2):
                sigall[2::3] = np.bitwise_and((sigbytes[1::4] >> 3), 0x1f)[0:len(
                    sigall[2::3])] + 32 * np.bitwise_and(sigbytes[3::4] >> 3, 0x1f)[0:len(sigall[2::3])]
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]
            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sigall[sigall > 511] -= 1024

            # Give the average sample in each frame for each channel
            sig = np.zeros([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

    elif fmt == '311':  # Three 10 bit samples packed into 4 bytes with 2 bits discarded
        nbytesload = int((nsamp - 1) / 3.) + nsamp + 1
        sigbytes = np.fromfile(
            fp,
            dtype=np.dtype(
                datatypes[fmt]),
            count=nbytesload).astype('uint')  # Loaded as unsigned 1 byte blocks

        if tsampsperframe == nsig:  # No extra samples/frame
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sig = np.zeros(nsamp)

            sig[0::3] = sigbytes[0::4][
                0:len(sig[0::3])] + 256 * np.bitwise_and(sigbytes[1::4], 0x03)[0:len(sig[0::3])]
            if len(sig > 1):
                sig[1::3] = (sigbytes[1::4] >> 2)[0:len(sig[1::3])] + 64 * \
                    np.bitwise_and(sigbytes[2::4], 0x0f)[0:len(sig[1::3])]
            if len(sig > 2):
                sig[2::3] = (sigbytes[2::4] >> 4)[0:len(sig[2::3])] + 16 * \
                    np.bitwise_and(sigbytes[3::4], 0x7f)[0:len(sig[2::3])]
            # First signal is first byte and 2 lsb of second byte.
            # Second signal is 6 msb of second byte and 4 lsb of third byte
            # Third signal is 4 msb of third byte and 6 msb of forth byte
            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]
            # Reshape into final array of samples
            sig = sig.reshape(siglen, nsig)
            # Convert to int64 to be able to hold -ve values
            sig = sig.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sig[sig > 511] -= 1024

        else:  # At least one channel has multiple samples per frame. All extra samples are averaged.
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sigall = np.zeros(nsamp)
            sigall[
                0::3] = sigbytes[
                0::4][
                0:len(
                    sigall[
                        0::3])] + 256 * np.bitwise_and(
                sigbytes[
                    1::4], 0x03)[
                0:len(
                    sigall[
                        0::3])]
            if len(sigall > 1):
                sigall[1::3] = (sigbytes[1::4] >> 2)[0:len(
                    sigall[1::3])] + 64 * np.bitwise_and(sigbytes[2::4], 0x0f)[0:len(sigall[1::3])]
            if len(sigall > 2):
                sigall[2::3] = (sigbytes[2::4] >> 4)[0:len(
                    sigall[2::3])] + 16 * np.bitwise_and(sigbytes[3::4], 0x7f)[0:len(sigall[2::3])]
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]
            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sigall[sigall > 511] -= 1024
            # Give the average sample in each frame for each channel
            sig = np.zeros([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

    else:  # Simple format signals that can be loaded as they are stored.

        if tsampsperframe == nsig:  # No extra samples/frame
            sig = np.fromfile(fp, dtype=np.dtype(datatypes[fmt]), count=nsamp)
            sig = sig.reshape(siglen, nsig).astype('int')
        else:  # At least one channel has multiple samples per frame. Extra samples are averaged.
            sigall = np.fromfile(fp,
                                 dtype=np.dtype(datatypes[fmt]),
                                 count=nsamp)  # All samples loaded
            # Keep the first sample in each frame for each channel
            sig = np.empty([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')
        # Correct byte offset format data
        if fmt == '80':
            sig = sig - 128
        elif fmt == '160':
            sig = sig - 32768
        nbytesload = nsamp * bytespersample[fmt]

    return sig, nbytesload

# Bytes required to hold each sample (including wasted space) for
# different wfdb formats
bytespersample = {'8': 1, '16': 2, '24': 3, '32': 4, '61': 2,
                  '80': 1, '160': 2, '212': 1.5, '310': 4 / 3., '311': 4 / 3.}

# Values that correspond to NAN (Format 8 has no NANs)
wfdbInvalids = {
    '16': -32768,
    '24': -8388608,
    '32': -2147483648,
    '61': -32768,
    '80': -128,
    '160': -32768,
    '212': -2048,
    '310': -512,
    '311': -512}

# Data type objects for each format to load. Doesn't directly correspond
# for final 3 formats.
datatypes = {'8': '<i1', '16': '<i2', '24': '<i3', '32': '<i4',
             '61': '>i2', '80': '<u1', '160': '<u2',
             '212': '<u1', '310': '<u1', '311': '<u1'}

# Channel dependent field items that need to be rearranged if input
# channels is not default.
arrangefields = [
    "filename",
    "fmt",
    "sampsperframe",
    "skew",
    "byteoffset",
    "gain",
    "units",
    "baseline",
    "initvalue",
    "signame"]


def processsegment(fields, dirname, baserecordname, sampfrom, sampto, channels, physical):
    if (len(set(fields["filename"])) ==
                1):  # single dat (or binary) file in the segment
            # Signal length was not specified in the header, calculate it from
            # the file size.
            if not fields["nsamp"]:
                filesize = os.path.getsize(
                    os.path.join(dirname, fields["filename"][0]))
                fields["nsamp"] = int(
                    (filesize -
                     fields["byteoffset"][0]) /
                    fields["nsig"] /
                    bytespersample[
                        fields["fmt"][0]])
            if sampto:
                if sampto > fields["nsamp"]:
                    sys.exit(
                        "sampto exceeds length of record: ",
                        fields["nsamp"])
            else:
                sampto = fields["nsamp"]

            sig = readdat(
                os.path.join(
                    dirname,
                    fields["filename"][0]),
                fields["fmt"][0],
                fields["byteoffset"][0],
                sampfrom,
                sampto,
                fields["nsig"],
                fields["nsamp"],
                fields["sampsperframe"],
                fields["skew"])

            if channels:  # User input channels
                # If input channels are not equal to the full set of ordered
                # channels, remove the non-included ones and reorganize the
                # remaining ones that require it.
                if channels != list(range(0, fields["nsig"])):
                    sig = sig[:, channels]

                    # Rearrange the channel dependent fields
                    for fielditem in arrangefields:
                        fields[fielditem] = [fields[fielditem][ch]
                                             for ch in channels]
                    fields["nsig"] = len(channels)  # Number of signals.

            if (physical == 1) & (fields["fmt"] != '8'):
                # Insert nans/invalid samples.
                sig = sig.astype(float)
                sig[sig == wfdbInvalids[fields["fmt"][0]]] = np.nan
                sig = np.subtract(sig, np.array(
                    [float(i) for i in fields["baseline"]]))
                sig = np.divide(sig, np.array([fields["gain"]]))

    else:  # Multiple dat files in the segment. Read different dats and merge them channel wise.

            if not channels:  # Default all channels
                channels = list(range(0, fields["nsig"]))

            # Get info to arrange channels correctly
            filenames = []  # The unique dat files to open,
            filechannels = {}  # All the channels that each dat file contains
            # The channels for each dat file to be returned and the
            # corresponding target output channels
            filechannelsout = {}
            for ch in list(range(0, fields["nsig"])):
                sigfile = fields["filename"][ch]
                # Correspond every channel to a dat file
                if sigfile not in filechannels:
                    filechannels[sigfile] = [ch]
                else:
                    filechannels[sigfile].append(ch)
                # Get only relevant files/channels to load
                if ch in channels:
                    if sigfile not in filenames:  # Newly encountered dat file
                        filenames.append(sigfile)
                        filechannelsout[sigfile] = [
                            [filechannels[sigfile].index(ch), channels.index(ch)]]
                    else:
                        filechannelsout[sigfile].append(
                            [filechannels[sigfile].index(ch), channels.index(ch)])

            # Allocate final output array
            # header doesn't have signal length. Figure it out from the first
            # dat to read.
            if not fields["nsamp"]:
                filesize = os.path.getsize(
                    os.path.join(dirname, fields["filename"][0]))
                fields["nsamp"] = int((filesize -
                                       fields["byteoffset"][channels[0]]) /
                                      len(filechannels[filenames[0]]) /
                                      bytespersample[fields["fmt"][channels[0]]])

            if not sampto:  # if sampto field is empty
                sampto = fields["nsamp"]
            if physical == 0:
                sig = np.empty([sampto - sampfrom, len(channels)], dtype='int')
            else:
                sig = np.empty([sampto - sampfrom, len(channels)])

            # So we do need to rearrange the fields below for the final output
            # 'fields' item and even for the physical conversion below. The
            # rearranged fields will only contain the fields of the selected
            # channels in their specified order. But before we do this, we need
            # to extract sampsperframe and skew for the individual dat files
            filesampsperframe = {}  # The sampsperframe for each channel in each file
            fileskew = {}  # The skew for each channel in each file
            for sigfile in filenames:
                filesampsperframe[sigfile] = [fields["sampsperframe"][
                    datchan] for datchan in filechannels[sigfile]]
                fileskew[sigfile] = [fields["skew"][datchan]
                                     for datchan in filechannels[sigfile]]

            # Rearrange the fields given the input channel dependent fields
            for fielditem in arrangefields:
                fields[fielditem] = [fields[fielditem][ch] for ch in channels]
            fields["nsig"] = len(channels)  # Number of signals.

            # Read signal files one at a time and store the relevant channels
            for sigfile in filenames:
                sig[:,
                    [outchan[1] for outchan in filechannelsout[sigfile]]] = readdat(os.path.join(dirname,
                                                                                                 sigfile),
                                                                                    fields["fmt"][fields["filename"].index(sigfile)],
                                                                                    fields["byteoffset"][fields["filename"].index(sigfile)],
                                                                                    sampfrom,
                                                                                    sampto,
                                                                                    len(filechannels[sigfile]),
                                                                                    fields["nsamp"],
                                                                                    filesampsperframe[sigfile],
                                                                                    fileskew[sigfile])[:,
                                                                                                       [datchan[0] for datchan in filechannelsout[sigfile]]]

            if (physical == 1) & (fields["fmt"] != '8'):
                # Insert nans/invalid samples.
                sig = sig.astype(float)
                for ch in range(0, fields["nsig"]):
                    sig[sig[:, ch] == wfdbInvalids[
                        fields["fmt"][ch]], ch] = np.nan
                sig = np.subtract(sig, np.array(
                    [float(b) for b in fields["baseline"]]))
                sig = np.divide(sig, np.array([fields["gain"]]))

    return sig, fields

def fixedorvariable(fields, dirname):
    if fields["nsampseg"][
            0] == 0:  # variable layout - first segment is layout specification file
        startseg = 1
        # Store the layout header info.
        layoutfields = readheader(
            os.path.join(
                dirname,
                fields["filename"][0]))
    else:  # fixed layout - no layout specification file.
        startseg = 0
        layoutfields=[]
    return startseg, layoutfields

# Determine the segments and samples that have to be read in a multi-segment record
def requiredsections(fields, sampfrom, sampto, startseg):
    # Cumulative sum of segment lengths
    cumsumlengths = list(np.cumsum(fields["nsampseg"][startseg:]))

    if not sampto:
        sampto = cumsumlengths[len(cumsumlengths) - 1]

    if sampto > cumsumlengths[len(cumsumlengths) - 1]:
        sys.exit(
            "sampto exceeds length of record: ",
            cumsumlengths[
                len(cumsumlengths) - 1])

    # First segment
    readsegs = [[sampfrom < cs for cs in cumsumlengths].index(True)]
    if sampto == cumsumlengths[len(cumsumlengths) - 1]:
        readsegs.append(len(cumsumlengths) - 1)  # Final segment
    else:
        readsegs.append([sampto < cs for cs in cumsumlengths].index(True))

    if readsegs[1] == readsegs[0]:  # Only one segment to read
        readsegs = [readsegs[0]]
        # The sampfrom and sampto for each segment
        readsamps = [[sampfrom, sampto]]
    else:
        readsegs = list(
            range(
                readsegs[0],
                readsegs[1] +
                1))  # Expand into list
        # The sampfrom and sampto for each segment. Most are [0, nsampseg]
        readsamps = [[0, fields["nsampseg"][s + startseg]]
                     for s in readsegs]
        # Starting sample for first segment
        readsamps[0][0] = sampfrom - ([0] + cumsumlengths)[readsegs[0]]
        readsamps[len(readsamps) - 1][1] = sampto - ([0] + cumsumlengths)[
            readsegs[len(readsamps) - 1]]  # End sample for last segment
    # Done obtaining segments to read and samples to read in each segment.
    return readsegs, readsamps, sampto


# Allocate empty structures for storing information from multiple segments
def allocateoutput(fields, channels, stacksegments, sampfrom, sampto, physical, startseg, readsegs):

    if not channels:
        channels = list(range(0, fields["nsig"]))  # All channels

    # Figure out the dimensions of the entire signal
    if sampto:  # User inputs sampto
        nsamp = sampto-sampfrom
    elif fields["nsamp"]:  # master header has number of samples
        nsamp = fields["nsamp"]-sampfrom
    else:  # Have to figure out length by adding up all segment lengths
        nsamp = sum(fields["nsampseg"][startseg:])-sampfrom

    if stacksegments == 1:  # Output a single concatenated numpy array
        indstart = 0  # The start index of the stacked numpy array to begin filling in the current segment
        if physical == 0:
            sig = np.empty((nsamp, len(channels)), dtype='int')
        else:
            sig = np.empty((nsamp, len(channels)), dtype='float')
    else:  # Output a list of numpy arrays
        # Empty list for storing segment signals.
        sig = [None] * len(readsegs)
        indstart=[]
    # List for storing segment fields.
    segmentfields = [None] * len(readsegs)

    return sig, channels, nsamp, segmentfields, indstart

# Determine the channels to be returned for each segment
def getsegmentchannels(startseg, segrecordname, dirname, layoutfields, channels):

    if startseg == 0:  # Fixed layout signal. Channels for record are always same.
        segchannels = channels
    else:  # Variable layout signal. Work out which channels from the segment to load if any.
        if segrecordname != '~':
            sfields = readheader(os.path.join(dirname, segrecordname))
            wantsignals = [layoutfields["signame"][c] for c in channels]  # Signal names of wanted channels
            segchannels = []  # The channel numbers wanted that are contained in the segment
            returninds = []  # 1 and 0 marking channels of the numpy array to be filled by
                                     # the returned segment channels
            for ws in wantsignals:
                if ws in sfields["signame"]:
                    segchannels.append(sfields["signame"].index(ws))
                    returninds.append(1)
                else:
                    returninds.append(0)
            returninds = np.array(returninds)
            # emptyinds: the channels of the overall array that the segment doesn't contain
            emptyinds = np.where(returninds == 0)[0]
            # returninds: the channels of the overall array that the segment does contain
            returninds = np.where(returninds == 1)[0]
        else:
            segchannels = []
            returninds=[]
            emptyinds=[]

    return segchannels, returninds, emptyinds

# Expand the channel dependent fields of a segment to match the overall layout specification header.
def expandfields(segmentfields, segnum, startseg, readsegs, channels, returninds):

    expandedfields = dict.copy(segmentfields[segnum - startseg - readsegs[0]])
    for fielditem in arrangefields:
        expandedfields[fielditem] = ['No Channel'] * len(channels)
        for c in range(0, len(returninds)):
            expandedfields[fielditem][returninds[c]] = segmentfields[segnum - startseg - readsegs[0]][fielditem][c]
        segmentfields[segnum - startseg - readsegs[0]][fielditem] = expandedfields[fielditem]
        # Keep fields['nsig'] as the number of returned channels from the segments.
    return segmentfields

            
def checkrecordfiles(recordname, pbdl, dldir):
    """Figure out the directory in which to process record files and download missing 
    files if specified. *If you wish to directly download files for a record, call
    'dlrecordfiles'. This is a helper function for rdsamp. 

    Input arguments:
    - recordname: name of the record
    - pbdl: flag specifying whether a physiobank record should be downloaded
    - dldir: directory in which to download physiobank files

    Output arguments:
    - dirname: the directory name from where the data files will be read
    - baserecordname: the base name of the WFDB record without any file paths
    - filestoremove: a list of downloaded files that are to be removed
    """
    
    filestoremove=[]
    
    # Download physiobank files if specified
    if pbdl == 1:  
        dledfiles = dlrecordfiles(recordname, dldir)

        # The directory to read the files from is the downloaded directory
        dirname = dldir
        (_, baserecordname)= os.path.split(recordname)
    else:
        dirname, baserecordname = os.path.split(recordname)
        
    return dirname, baserecordname
    
            
            
def rdsamp(
        recordname,
        sampfrom=0,
        sampto=[],
        channels=[],
        physical=1,
        stacksegments=1,
        pbdl=0,
        dldir=os.getcwd()):
    """Read a WFDB record and return the signal as a numpy array and the metadata as a dictionary.

    Usage:
    sig, fields = rdsamp(recordname, sampfrom=0, sampto=[], channels=[], physical=1, stacksegments=1, 
        pbdl=0, dldir=os.cwd())

    Input arguments:
    - recordname (required): The name of the WFDB record to be read (without any file extensions). 
      If the argument contains any path delimiter characters, the argument will be interpreted as 
      PATH/baserecord and the data files will be searched for in the local path. If the pbdownload 
      flag is set to 1, recordname will be interpreted as a physiobank record name including the 
      database subdirectory. 
    - sampfrom (default=0): The starting sample number to read for each channel.
    - sampto (default=length of entire signal): The final sample number to read for each channel.
    - channels (default=all channels): Indices specifying the channel to be returned.
    - physical (default=1): Flag that specifies whether to return signals in physical (1) or 
      digital (0) units.
    - stacksegments (default=1): Flag used only for multi-segment files. Specifies whether to 
      return the signal as a single stacked/concatenated numpy array (1) or as a list of one 
      numpy array for each segment (0).
    - pbdl (default=0): If this argument is set, the function will assume that the user is trying 
      to download a physiobank file. Therefore the 'recordname' argument will be interpreted as 
      a physiobank record name including the database subdirectory, rather than a local directory. 
    - dldir (default=os.getcwd()): The directory to download physiobank files to. 

    Output variables:
    - sig: An nxm numpy array where n is the signal length and m is the number of channels.
      If the input record is a multi-segment record, depending on the input stacksegments flag,
      sig will either be a single stacked/concatenated numpy array (1) or a list of one numpy
      array for each segment (0). For empty segments, stacked format will contain Nan values,
      and non-stacked format will contain a single integer specifying the length of the empty segment.
    - fields: A dictionary of metadata about the record extracted or deduced from the header/signal file.
      If the input record is a multi-segment record, the output argument will be a list of dictionaries:
              : The first list element will be a dictionary of metadata about the master header.
              : If the record is in variable layout format, the next list element will be a dictionary
                of metadata about the layout specification header.
              : The last list element will be a list of dictionaries of metadata for each segment.
                For empty segments, the dictionary will be replaced by a single string: 'Empty Segment'
                
    Example: sig, fields = wfdb.rdsamp('macecgdb/test01_00s', sampfrom=800, pbdl=1, 
        dldir='/home/username/Downloads/wfdb')
    """

    if sampfrom < 0:
        sys.exit("sampfrom must be non-negative")
    if channels and min(channels) < 0:
        sys.exit("input channels must be non-negative")
    
    dirname, baserecordname = checkrecordfiles(recordname, pbdl, dldir)
    
    fields = readheader(os.path.join(dirname, baserecordname))  

    if fields["nsig"] == 0:
        sys.exit("This record has no signals. Use rdann to read annotations")

    # Begin processing the data files.
    
    # Single segment file
    if fields["nseg"] == 1:  
        sig, fields = processsegment(fields, dirname, baserecordname, sampfrom, sampto, 
            channels, physical)

    # Multi-segment file. Preprocess and recursively call rdsamp on segments
    else:
        # Determine if the record is fixed or variable layout.
        # startseg is the first signal segment, 1 or 0.
        startseg, layoutfields = fixedorvariable(fields, dirname)

        # Determine the segments and samples that have to be read
        readsegs, readsamps, sampto = requiredsections(fields, sampfrom, sampto, startseg)

        # Preprocess/preallocate according to the chosen output format
        sig, channels, nsamp, segmentfields, indstart= allocateoutput(fields, channels, 
            stacksegments, sampfrom, sampto, physical, startseg, readsegs)

        # Read and store segments one at a time.
        # segnum (the segment number) accounts for the layout record if exists
        # and skips past it.
        for segnum in [r + startseg for r in readsegs]:

            segrecordname = fields["filename"][segnum]

            # Work out the relative channels to return from this segment
            segchannels, returninds, emptyinds = getsegmentchannels(startseg, segrecordname, 
                dirname, layoutfields, channels)

            if stacksegments == 0:  # Return list of np arrays
                # Empty segment or no desired channels in segment. Store indicator and segment
                # length.
                if (segrecordname == '~') | (not segchannels):
                    # sig[segnum-startseg-readsegs[0]]=fields["nsampseg"][segnum] # store
                    # the entire segment length? Or just selected length? Preference...

                    sig[segnum - startseg - readsegs[0]] = readsamps[segnum - startseg - \
                        readsegs[0]][1] - readsamps[segnum - startseg - readsegs[0]][0]
                    segmentfields[segnum - startseg - readsegs[0]] = "Empty Segment"

                else:  # Non-empty segment that contains wanted channels. Read its signal and header fields
                    sig[segnum -
                        startseg -
                        readsegs[0]], segmentfields[segnum -
                                                    startseg -
                                                    readsegs[0]] = rdsamp(recordname=os.path.join(dirname,
                                                        segrecordname), physical=physical, sampfrom=readsamps[segnum - startseg -readsegs[0]][0], sampto=readsamps[segnum - startseg - readsegs[0]][1], channels=segchannels)

            else:  # Return single stacked np array of all (selected) channels

                indend = indstart + readsamps[segnum - startseg - readsegs[0]][1] - readsamps[
                    segnum - startseg - readsegs[0]][0]  # end index of the large array for this segment
                if (segrecordname == '~') | (not segchannels):  # Empty segment or no wanted channels: fill in invalids
                    if physical == 0:
                        sig[indstart:indend, :] = -2147483648
                    else:
                        sig[indstart:indend, :] = np.nan
                    segmentfields[segnum - startseg - readsegs[0]] = "Empty Segment"
                else:  # Non-empty segment - Get samples
                    if startseg == 1:  # Variable layout format. Load data then rearrange channels.
                        sig[indstart:indend, returninds], segmentfields[segnum -
                                                                        startseg -
                                                                        readsegs[0]] = rdsamp(recordname=os.path.join(dirname, segrecordname), physical=physical, sampfrom=readsamps[segnum - startseg - readsegs[0]][0], sampto=readsamps[segnum -startseg - readsegs[0]][1], channels=segchannels)  # Load all the wanted channels that the segment contains
                        if physical == 0:  # Fill the rest with invalids
                            sig[indstart:indend, emptyinds] = -2147483648
                        else:
                            sig[indstart:indend, emptyinds] = np.nan
                        # Expand the channel dependent fields to match the overall layout.
                        segmentfields=expandfields(segmentfields, segnum, startseg, readsegs, channels, returninds)

                    else:  # Fixed layout - channels are already arranged
                        sig[
                            indstart:indend, :], segmentfields[
                            segnum - startseg] = rdsamp(
                            recordname=os.path.join(
                                dirname, segrecordname), physical=physical, sampfrom=readsamps[
                                segnum - startseg - readsegs[0]][0], sampto=readsamps[
                                segnum - startseg - readsegs[0]][1], channels=segchannels)
                indstart = indend  # Update the start index for filling in the next part of the array

        # Done reading all segments

        # Return a list of fields. First element is the master, next is layout
        # if any, last is a list of all the segment fields.
        if startseg == 1:  # Variable layout format
            fields = [fields, layoutfields, segmentfields]
        else:  # Fixed layout format.
            fields = [fields, segmentfields]

    return (sig, fields)


if __name__ == '__main__':
    rdsamp(sys.argv)
