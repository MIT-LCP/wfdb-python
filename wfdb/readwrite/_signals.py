import numpy as np
import os
import math
from . import downloads

# All defined WFDB dat formats
datformats = ["80","212","16","24","32"]

specialfmts = ['212','310','311']

# Class with signal methods
# To be inherited by Record from records.py.
class SignalsMixin(object):


    def wrdats(self, expanded):
        # Write all dat files associated with a record
        # expanded=True to use e_d_signals instead of d_signals

        if not self.nsig:
            return
        
        # Get all the fields used to write the header
        # Assuming this method was called through wrsamp,
        # these will have already been checked in wrheader()
        writefields = self.getwritefields()

        if expanded:
            # Using list of arrays e_d_signals
            self.checkfield('e_d_signals', channels = 'all')
        else:
            # Check the validity of the d_signals field
            self.checkfield('d_signals')

        # Check the cohesion of the d_signals field against the other fields used to write the header
        self.checksignalcohesion(writefields, expanded)
        
        # Write each of the specified dat files
        self.wrdatfiles(expanded)



    # Check the cohesion of the d_signals/e_d_signals field with the other fields used to write the record
    def checksignalcohesion(self, writefields, expanded):

        # Using list of arrays e_d_signals
        if expanded:
            # Set default sampsperframe
            spf = self.sampsperframe
            for ch in range(len(spf)):
                if spf[ch] is None:
                    spf[ch] = 1

            # Match the actual signal shape against stated length and number of channels
            if self.nsig != len(self.e_d_signals):
                raise ValueError('nsig does not match the length of e_d_signals')
            for ch in range(self.nsig):
                if len(self.e_d_signals[ch]) != spf[ch]*self.siglen:
                    raise ValueError('Length of channel '+str(ch)+'does not match sampsperframe['+str(ch+']*siglen'))

            # For each channel (if any), make sure the digital format has no values out of bounds
            for ch in range(0, self.nsig):
                fmt = self.fmt[ch]
                dmin, dmax = digi_bounds(self.fmt[ch])
                
                chmin = min(self.e_d_signals[ch])
                chmax = max(self.e_d_signals[ch])
                if (chmin < dmin) or (chmax > dmax):
                    raise IndexError("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))
            
            # Ensure that the checksums and initial value fields match the digital signal (if the fields are present)
            if self.nsig>0:
                if 'checksum' in writefields:
                    realchecksum = self.calc_checksum(expanded)
                    if self.checksum != realchecksum:
                        print("The actual checksum of e_d_signals is: ", realchecksum)
                        raise ValueError("checksum field does not match actual checksum of e_d_signals")
                if 'initvalue' in writefields:
                    realinitvalue = [self.e_d_signals[ch][0] for ch in range(self.nsig)]
                    if self.initvalue != realinitvalue:
                        print("The actual initvalue of e_d_signals is: ", realinitvalue)
                        raise ValueError("initvalue field does not match actual initvalue of e_d_signals")

        # Using uniform d_signals
        else:
            # Match the actual signal shape against stated length and number of channels
            if (self.siglen, self.nsig) != self.d_signals.shape:
                print('siglen: ', self.siglen)
                print('nsig: ', self.nsig)
                print('d_signals.shape: ', self.d_signals.shape)
                raise ValueError('siglen and nsig do not match shape of d_signals')

            # For each channel (if any), make sure the digital format has no values out of bounds
            for ch in range(0, self.nsig):
                fmt = self.fmt[ch]
                dmin, dmax = digi_bounds(self.fmt[ch])
                
                chmin = min(self.d_signals[:,ch])
                chmax = max(self.d_signals[:,ch])
                if (chmin < dmin) or (chmax > dmax):
                    raise IndexError("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))
                        
            # Ensure that the checksums and initial value fields match the digital signal (if the fields are present)
            if self.nsig>0:
                if 'checksum' in writefields:
                    realchecksum = self.calc_checksum()
                    if self.checksum != realchecksum:
                        print("The actual checksum of d_signals is: ", realchecksum)
                        raise ValueError("checksum field does not match actual checksum of d_signals")
                if 'initvalue' in writefields:
                    realinitvalue = list(self.d_signals[0,:])
                    if self.initvalue != realinitvalue:
                        print("The actual initvalue of d_signals is: ", realinitvalue)
                        raise ValueError("initvalue field does not match actual initvalue of d_signals")

    
    def set_p_features(self, do_dac = False, expanded=False):
        """
        Use properties of the p_signals (expanded=False) or e_p_signals field to set other fields: 
          - nsig
          - siglen
        If expanded=True, sampsperframe is also required.

        If do_dac == True, the (e_)_d_signals field will be used to perform digital to analogue conversion
        to set the (e_)p_signals field, before (e_)p_signals is used. 
        Regarding dac conversion:
          - fmt, gain, and baseline must all be set in order to perform dac.
          - Unlike with adc, there is no way to infer these fields.
          - Using the fmt, gain and baseline fields, dac is performed, and (e_)p_signals is set.

        *Developer note: Seems this function will be very infrequently used.
         The set_d_features function seems far more useful.
        """

        if expanded:
            if do_dac == 1:
                self.checkfield('e_d_signals', channels = 'all')
                self.checkfield('fmt', 'all')
                self.checkfield('adcgain', 'all')
                self.checkfield('baseline', 'all')
                self.checkfield('sampsperframe', 'all')

                # All required fields are present and valid. Perform DAC
                self.e_p_signals = self.dac(expanded)

            # Use e_p_signals to set fields
            self.checkfield('e_p_signals', channels = 'all')
            self.siglen = int(len(self.e_p_signals[0])/self.sampsperframe[0])
            self.nsig = len(self.e_p_signals)
        else:
            if do_dac == 1:
                self.checkfield('d_signals')
                self.checkfield('fmt', 'all')
                self.checkfield('adcgain', 'all')
                self.checkfield('baseline', 'all')

                # All required fields are present and valid. Perform DAC
                self.p_signals = self.dac()

            # Use p_signals to set fields
            self.checkfield('p_signals')
            self.siglen = self.p_signals.shape[0]
            self.nsig = self.p_signals.shape[1]


    def set_d_features(self, do_adc = False, singlefmt = 1, expanded=False):
        """
        Use properties of the (e_)d_signals field to set other fields: nsig, siglen, initvalue, checksum, *(fmt, adcgain, baseline) 
        If do_adc == True, the (e_)p_signals field will first be used to perform analogue to digital conversion to set the (e_)d_signals 
        field, before (e_)d_signals is used. 

        Regarding adc conversion:
          - If fmt is unset:
            - Neither adcgain nor baseline may be set. If the digital values used to store the signal are known, then the file
              format should also be known. 
            - The most appropriate fmt for the signals will be calculated and the 'fmt' attribute will be set. Given that neither
              gain nor baseline are allowed to be set, optimal values for those fields will then be calculated and set as well.

          - If fmt is set:
            - If both adcgain and baseline are unset, optimal values for those fields will be calculated the fields will be set. 
            - If both adcgain and baseline are set, the function will continue.
            - If only one of adcgain and baseline are set, this function will throw an error. It makes no sense to know only
              one of those fields.

          ADC will occur after valid values for fmt, adcgain, and baseline are present, using all three fields.  
        """
        if expanded:
            # adc is performed.
            if do_adc == True:
                self.checkfield('e_p_signals', channels = 'all')

                # If there is no fmt set
                if self.fmt is None:
                    # Make sure that neither adcgain nor baseline are set
                    if self.adcgain is not None or self.baseline is not None:
                        raise Exception('If fmt is not set, gain and baseline may not be set either.')
                    # Choose appropriate fmts based on estimated signal resolutions. 
                    res = estres(self.e_p_signals)
                    self.fmt = wfdbfmt(res, singlefmt)
                # If there is a fmt set
                else:
                    self.checkfield('fmt', 'all')
                    # Neither field set
                    if self.adcgain is None and self.baseline is None:
                        # Calculate and set optimal gain and baseline values to convert physical signals
                        self.adcgain, self.baseline = self.calculate_adcparams()
                    # Exactly one field set
                    elif (self.adcgain is None) ^ (self.baseline is None):
                        raise Exception('If fmt is set, gain and baseline should both be set or not set.')
                
                self.checkfield('adcgain', 'all')
                self.checkfield('baseline', 'all')

                # All required fields are present and valid. Perform ADC
                self.d_signals = self.adc(expanded)

            # Use e_d_signals to set fields
            self.checkfield('e_d_signals', channels = 'all')
            self.siglen = int(len(self.e_d_signals[0])/self.sampsperframe[0])
            self.nsig = len(self.e_d_signals)
            self.initvalue = [sig[0] for sig in self.e_d_signals]
            self.checksum = self.calc_checksum(expanded)
        else:
            # adc is performed.
            if do_adc == True:
                self.checkfield('p_signals')

                # If there is no fmt set
                if self.fmt is None:
                    # Make sure that neither adcgain nor baseline are set
                    if self.adcgain is not None or self.baseline is not None:
                        raise Exception('If fmt is not set, gain and baseline may not be set either.')
                    # Choose appropriate fmts based on estimated signal resolutions. 
                    res = estres(self.p_signals)
                    self.fmt = wfdbfmt(res, singlefmt)
                # If there is a fmt set
                else:
                    
                    self.checkfield('fmt', 'all')
                    # Neither field set
                    if self.adcgain is None and self.baseline is None:
                        # Calculate and set optimal gain and baseline values to convert physical signals
                        self.adcgain, self.baseline = self.calculate_adcparams()
                    # Exactly one field set
                    elif (self.adcgain is None) ^ (self.baseline is None):
                        raise Exception('If fmt is set, gain and baseline should both be set or not set.')
                
                self.checkfield('adcgain', 'all')
                self.checkfield('baseline', 'all')

                # All required fields are present and valid. Perform ADC
                self.d_signals = self.adc()

            # Use d_signals to set fields
            self.checkfield('d_signals')
            self.siglen = self.d_signals.shape[0]
            self.nsig = self.d_signals.shape[1]
            self.initvalue = list(self.d_signals[0,:])
            self.checksum = self.calc_checksum()


    # Returns the analogue to digital conversion for the physical signal stored in p_signals. 
    # The p_signals, fmt, gain, and baseline fields must all be valid.
    def adc(self, expanded=False):
        
        # The digital nan values for each channel
        dnans = digi_nan(self.fmt)
        
        if expanded:
            d_signals = []
            for ch in range(0, self.nsig):
                chnanloc = np.isnan(self.e_p_signals)
                d_signals.append(self.e_p_signals[ch] * self.adcgain[ch] + self.baseline[ch])
                d_signals[ch][chnanlocs] = dnans[ch]
        else:
            d_signals = self.p_signals * self.adcgain + self.baseline
            
            for ch in range(0, np.shape(self.p_signals)[1]):
                # Nan locations
                nanlocs = np.isnan(self.p_signals[:,ch])
                if nanlocs.any():
                    d_signals[nanlocs,ch] = dnans[ch]
            
            d_signals = d_signals.astype('int64')

        return d_signals

    
    def dac(self, expanded=False):
        """
        Returns the digital to analogue conversion for a Record object's signal stored
        in d_signals if expanded is False, or e_d_signals if expanded is True.
        The d_signals/e_d_signals, fmt, gain, and baseline fields must all be valid.
        """
        # The digital nan values for each channel
        dnans = digi_nan(self.fmt)

        if expanded:
            p_signal = []
            for ch in range(0, self.nsig):
                # nan locations for the channel
                chnanlocs = self.e_d_signals[ch] == dnans[ch]
                p_signal.append((self.e_d_signals[ch] - self.baseline[ch])/float(self.adcgain[ch]))
                p_signal[ch][chnanlocs] = np.nan
        else:
            # nan locations
            nanlocs = self.d_signals == dnans
            p_signal = (self.d_signals - self.baseline)/[float(g) for g in self.adcgain]
            p_signal[nanlocs] = np.nan
                
        return p_signal


    # Compute appropriate gain and baseline parameters given the physical signal and the fmts 
    # self.fmt must be a list with length equal to the number of signal channels in self.p_signals 
    def calculate_adcparams(self):
             
        # digital - baseline / gain = physical     
        # physical * gain + baseline = digital

        gains = []
        baselines = []
        
        # min and max ignoring nans, unless whole channel is nan. Should suppress warning message. 
        minvals = np.nanmin(self.p_signals, axis=0) 
        maxvals = np.nanmax(self.p_signals, axis=0)
        
        dnans = digi_nan(self.fmt)
        
        for ch in range(0, np.shape(self.p_signals)[1]):
            dmin, dmax = digi_bounds(self.fmt[ch]) # Get the minimum and maximum (valid) storage values
            dmin = dmin + 1 # add 1 because the lowest value is used to store nans
            dnan = dnans[ch]
            
            pmin = minvals[ch]
            pmax = maxvals[ch]
            
            # map values using full digital range.
            
            # If the entire signal is nan, just put any. 
            if pmin == np.nan:
                gain = 1 
                baseline = 1
            # If the signal is just one value, store all values as digital 1. 
            elif pmin == pmax:
                if minval ==0:
                    gain = 1
                    baseline = 1
                else:
                    gain = 1/minval # wait.. what if minval is 0... 
                    baseline = 0 
            else:
                
                gain = (dmax-dmin) / (pmax - pmin)
                baseline = dmin - gain * pmin

            # What about roundoff error? Make sure values don't map to beyond range. 
            baseline = int(baseline) 
            
            # WFDB library limits...     
            if abs(gain)>214748364 or abs(baseline)>2147483648:
                raise Exception('adcgain and baseline must have magnitudes < 214748364')
                    
            gains.append(gain)
            baselines.append(baseline)     
        
        return (gains, baselines)

    def calc_checksum(self, expanded=False):
        """
        Calculate the checksum(s) of the d_signals (expanded=False)
        or e_d_signals field (expanded=True)
        """
        if expanded:
            cs = [int(np.sum(self.e_d_signals[ch]) % 65536) for ch in range(self.nsig)]
        else:
            cs = np.sum(self.d_signals, 0) % 65536
            cs = [int(c) for c in cs]
        return cs

    # Write each of the specified dat files
    def wrdatfiles(self, expanded=False):

        # Get the set of dat files to be written, and
        # the channels to be written to each file. 
        filenames, datchannels = orderedsetlist(self.filename)

        # Get the fmt and byte offset corresponding to each dat file
        datfmts={}
        datoffsets={}
        for fn in filenames:
            datfmts[fn] = self.fmt[datchannels[fn][0]]

            # byteoffset may not be present
            if self.byteoffset is None:
                datoffsets[fn] = 0
            else:
                datoffsets[fn] = self.byteoffset[datchannels[fn][0]]

        # Write the dat files
        if expanded:
            for fn in filenames:
                wrdatfile(fn, datfmts[fn], None , datoffsets[fn], True, [self.e_d_signals[ch] for ch in datchannels[fn]], self.sampsperframe)
        else:
            # Create a copy to prevent overwrite
            dsig = self.d_signals.copy()
            for fn in filenames:
                wrdatfile(fn, datfmts[fn], dsig[:, datchannels[fn][0]:datchannels[fn][-1]+1], datoffsets[fn])


    def smoothframes(self, sigtype='physical'):
        """
        Convert expanded signals with different samples/frame into
        a uniform numpy array. 
        
        Input parameters
        - sigtype (default='physical'): Specifies whether to mooth
          the e_p_signals field ('physical'), or the e_d_signals
          field ('digital').
        """
        spf = self.sampsperframe[:]
        for ch in range(len(spf)):
            if spf[ch] is None:
                spf[ch] = 1

        # Total samples per frame
        tspf = sum(spf)

        if sigtype == 'physical':
            nsig = len(self.e_p_signals)
            siglen = int(len(self.e_p_signals[0])/spf[0])
            signal = np.zeros((siglen, nsig), dtype='float64')

            for ch in range(nsig):
                if spf[ch] == 1:
                    signal[:, ch] = self.e_p_signals[ch]
                else:
                    for frame in range(spf[ch]):
                        signal[:, ch] += self.e_p_signals[ch][frame::spf[ch]]
                    signal[:, ch] = signal[:, ch] / spf[ch]

        elif sigtype == 'digital':
            nsig = len(self.e_d_signals)
            siglen = int(len(self.e_d_signals[0])/spf[0])
            signal = np.zeros((siglen, nsig), dtype='int64')

            for ch in range(nsig):
                if spf[ch] == 1:
                    signal[:, ch] = self.e_d_signals[ch]
                else:
                    for frame in range(spf[ch]):
                        signal[:, ch] += self.e_d_signals[ch][frame::spf[ch]]
                    signal[:, ch] = signal[:, ch] / spf[ch]
        else:
            raise ValueError("sigtype must be 'physical' or 'digital'")

        return signal

#------------------- Reading Signals -------------------#

def rdsegment(filename, dirname, pbdir, nsig, fmt, siglen, byteoffset,
              sampsperframe, skew, sampfrom, sampto, channels,
              smoothframes, ignoreskew):
    """
    Read the samples from a single segment record's associated dat file(s)
    'channels', 'sampfrom', 'sampto', 'smoothframes', and 'ignoreskew' are
    user desired input fields.
    All other input arguments are specifications of the segment
    """

    # Avoid changing outer variables
    byteoffset = byteoffset[:]
    sampsperframe = sampsperframe[:]
    skew = skew[:]

    # Set defaults for empty fields
    for i in range(0, nsig):
        if byteoffset[i] == None:
            byteoffset[i] = 0
        if sampsperframe[i] == None:
            sampsperframe[i] = 1
        if skew[i] == None:
            skew[i] = 0

    # If skew is to be ignored, set all to 0
    if ignoreskew:
        skew = [0]*nsig

    # Get the set of dat files, and the
    # channels that belong to each file.
    filename, datchannel = orderedsetlist(filename)

    # Some files will not be read depending on input channels.
    # Get the the wanted fields only.
    w_filename = [] # one scalar per dat file
    w_fmt = {} # one scalar per dat file
    w_byteoffset = {} # one scalar per dat file
    w_sampsperframe = {} # one list per dat file
    w_skew = {} # one list per dat file
    w_channel = {} # one list per dat file

    for fn in filename:
        # intersecting dat channels between the input channels and the channels of the file 
        idc = [c for c in datchannel[fn] if c in channels]
        
        # There is at least one wanted channel in the dat file
        if idc != []:
            w_filename.append(fn)
            w_fmt[fn] = fmt[datchannel[fn][0]]
            w_byteoffset[fn] = byteoffset[datchannel[fn][0]]
            w_sampsperframe[fn] = [sampsperframe[c] for c in datchannel[fn]]
            w_skew[fn] = [skew[c] for c in datchannel[fn]]
            w_channel[fn] = idc
        
    # Wanted dat channels, relative to the dat file itself
    r_w_channel =  {}
    # The channels in the final output array that correspond to the read channels in each dat file
    out_datchannel = {}
    for fn in w_channel:
        r_w_channel[fn] = [c - min(datchannel[fn]) for c in w_channel[fn]]
        out_datchannel[fn] = [channels.index(c) for c in w_channel[fn]]
    
    # Signals with multiple samples/frame are smoothed, or all signals have 1 sample/frame.
    # Return uniform numpy array
    if smoothframes or sum(sampsperframe)==nsig:

        # Allocate signal array
        signals = np.zeros([sampto-sampfrom, len(channels)], dtype = 'int64')

        # Read each wanted dat file and store signals
        for fn in w_filename:
            signals[:, out_datchannel[fn]] = rddat(fn, dirname, pbdir, w_fmt[fn], len(datchannel[fn]), 
                siglen, w_byteoffset[fn], w_sampsperframe[fn], w_skew[fn], sampfrom, sampto, smoothframes)[:, r_w_channel[fn]]
    
    # Return each sample in signals with multiple samples/frame, without smoothing.
    # Return a list of numpy arrays for each signal.
    else:
        signals=[None]*len(channels)

        for fn in w_filename:
            # Get the list of all signals contained in the dat file 
            datsignals = rddat(fn, dirname, pbdir, w_fmt[fn], len(datchannel[fn]), 
                siglen, w_byteoffset[fn], w_sampsperframe[fn], w_skew[fn], sampfrom, sampto, smoothframes)

            # Copy over the wanted signals
            for cn in range(len(out_datchannel[fn])):
                signals[out_datchannel[fn][cn]] = datsignals[r_w_channel[fn][cn]]

    return signals 


def rddat(filename, dirname, pbdir, fmt, nsig,
        siglen, byteoffset, sampsperframe,
        skew, sampfrom, sampto, smoothframes):
    """
    Get samples from a WFDB dat file.
    'sampfrom', 'sampto', and smoothframes are user desired
    input fields. All other fields specify the file parameters. 

    Returns all channels

    Input arguments:
    - filename: The name of the dat file.
    - dirname: The full directory where the dat file is located, if the dat file is local.
    - pbdir: The physiobank directory where the dat file is located, if the dat file is remote.
    - fmt: The format of the dat file
    - nsig: The number of signals contained in the dat file  
    - siglen : The signal length (per channel) of the dat file
    - byteoffset: The byte offsets of the dat file
    - sampsperframe: The samples/frame for the signals of the dat file
    - skew: The skew for the signals of the dat file
    - sampfrom: The starting sample number to be read from the signals
    - sampto: The final sample number to be read from the signals
    - smoothframes: Whether to smooth channels with multiple samples/frame 
    """

    # Total number of samples per frame
    tsampsperframe = sum(sampsperframe)
    # The signal length to read (per channel)
    readlen = sampto - sampfrom

    # Calculate parameters used to read and process the dat file
    startbyte, nreadsamples, blockfloorsamples, extraflatsamples, nanreplace = calc_read_params(fmt, siglen, byteoffset,
                                                                                                skew, tsampsperframe,
                                                                                                sampfrom, sampto)

    # Number of bytes to be read from the dat file
    totalreadbytes = requiredbytenum('read', fmt, nreadsamples)
    
    # Total samples to be processed in intermediate step. Includes extra padded samples beyond dat file
    totalprocesssamples = nreadsamples + extraflatsamples

    # Total number of bytes to be processed in intermediate step.
    totalprocessbytes = requiredbytenum('read', fmt, totalprocesssamples)
    
    # Get the intermediate bytes or samples to process. Bit of a discrepancy. Recall special formats
    # load uint8 bytes, other formats already load samples.

    # Read values from dat file, and append bytes/samples if needed.
    if extraflatsamples:
        if fmt in specialfmts:
            # Extra number of bytes to append onto the bytes read from the dat file.
            extrabytenum = totalprocessbytes - totalreadbytes

            sigbytes = np.concatenate((getdatbytes(filename, dirname, pbdir, fmt, startbyte, nreadsamples),
                                      np.zeros(extrabytenum, dtype = 'uint8')))
        else:
            sigbytes = np.concatenate((getdatbytes(filename, dirname, pbdir, fmt, startbyte, nreadsamples),
                                      np.zeros(extraflatsamples, dtype='int64')))
    else:
        sigbytes = getdatbytes(filename, dirname, pbdir, fmt, startbyte, nreadsamples)

    # Continue to process the read values into proper samples

    # For special fmts, Turn the bytes into actual samples
    if fmt in specialfmts:
        sigbytes = bytes2samples(sigbytes, totalprocesssamples, fmt)
        # Remove extra leading sample read within the byte block if any
        if blockfloorsamples:
            sigbytes = sigbytes[blockfloorsamples:]
    # Adjust for byte offset formats
    elif fmt == '80':
        sigbytes = sigbytes - 128
    elif fmt == '160':
        sigbytes = sigbytes - 32768

    # No extra samples/frame. Obtain original uniform numpy array
    if tsampsperframe==nsig:
        # Reshape into multiple channels
        sig = sigbytes.reshape(-1, nsig)
        # Skew the signal
        sig = skewsig(sig, skew, nsig, readlen, fmt, nanreplace)

    # Extra frames present to be smoothed. Obtain averaged uniform numpy array
    elif smoothframes:

        # Allocate memory for smoothed signal
        sig = np.zeros((int(len(sigbytes)/tsampsperframe) , nsig), dtype='int64')

        # Transfer and average samples
        for ch in range(nsig):
            if sampsperframe[ch] == 1:
                sig[:, ch] = sigbytes[sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
            else:
                for frame in range(sampsperframe[ch]):
                    sig[:, ch] += sigbytes[sum(([0] + sampsperframe)[:ch + 1]) + frame::tsampsperframe]
        # Have to change the dtype for averaging frames
        sig = (sig.astype('float64') / sampsperframe)
        # Skew the signal
        sig = skewsig(sig, skew, nsig, readlen, fmt, nanreplace)

    # Extra frames present without wanting smoothing. Return all expanded samples.
    else:
        # List of 1d numpy arrays
        sig=[]

        # Transfer over samples
        for ch in range(nsig):
            # Indices of the flat signal that belong to the channel
            ch_indices = np.concatenate([np.array(range(sampsperframe[ch])) + sum([0]+sampsperframe[:ch]) + tsampsperframe*framenum for framenum in range(int(len(sigbytes)/tsampsperframe))])
            sig.append(sigbytes[ch_indices])
        # Skew the signal
        sig = skewsig(sig, skew, nsig, readlen, fmt, nanreplace, sampsperframe)

    # Integrity check of signal shape after reading
    checksigdims(sig, readlen, nsig, sampsperframe)

    return sig

def calc_read_params(fmt, siglen, byteoffset, skew, tsampsperframe, sampfrom, sampto):
    """
    Calculate parameters used to read and process the dat file
    
    Output arguments:
    - startbyte - The starting byte to read the dat file from. Always points to the start of a 
      byte block for special formats.
    - nreadsamples - The number of flat samples to read from the dat file.
    - blockfloorsamples - The extra samples read prior to the first desired sample, for special
      formats in order to ensure entire byte blocks are read.
    - extraflatsamples - The extra samples desired beyond what is contained in the file.
    - nanreplace - The number of samples to replace with nan at the end of each signal
      due to skew wanting samples beyond the file


    Example Parameters:
    siglen=100, t = 4 (total samples/frame), skew = [0, 2, 4, 5]
    sampfrom=0, sampto=100 --> readlen = 100, nsampread = 100*t, extralen = 5, nanreplace = [0, 2, 4, 5]
    sampfrom=50, sampto=100 --> readlen = 50, nsampread = 50*t, extralen = 5, nanreplace = [0, 2, 4, 5]
    sampfrom=0, sampto=50 --> readlen = 50, nsampread = 55*t, extralen = 0, nanreplace = [0, 0, 0, 0]
    sampfrom=95, sampto=99 --> readlen = 4, nsampread = 5*t, extralen = 4, nanreplace = [0, 1, 3, 4]
    """

    # First flat sample number to read (if all channels were flattened)
    startflatsample = sampfrom * tsampsperframe
    
    #endflatsample = min((sampto + max(skew)-sampfrom), siglen) * tsampsperframe

    # Calculate the last flat sample number to read.
    # Cannot exceed siglen * tsampsperframe, the number of samples stored in the file.
    # If extra 'samples' are desired by the skew, keep track.
    # Where was the -sampfrom derived from? Why was it in the formula?
    if (sampto + max(skew))>siglen:
        endflatsample = siglen*tsampsperframe
        extraflatsamples = (sampto + max(skew) - siglen) * tsampsperframe
    else:
        endflatsample = (sampto + max(skew)) * tsampsperframe
        extraflatsamples = 0

    # Adjust the starting sample number to read from start of blocks for special fmts.
    # Keep track of how many preceeding samples are read, to be discarded later.
    if fmt == '212':
        # Samples come in groups of 2, in 3 byte blocks
        blockfloorsamples = startflatsample % 2
        startflatsample = startflatsample - blockfloorsamples
    elif fmt in ['310', '311']:
        # Samples come in groups of 3, in 4 byte blocks
        blockfloorsamples = startflatsample % 3
        startflatsample = startflatsample - blockfloorsamples
    else:
        blockfloorsamples = 0

    # The starting byte to read from
    startbyte = byteoffset + int(startflatsample * bytespersample[fmt])

    # The number of samples to read
    nreadsamples = endflatsample - startflatsample

    # The number of samples to replace with nan at the end of each signal
    # due to skew wanting samples beyond the file

    # Calculate this using the above statement case: if (sampto + max(skew))>siglen:
    nanreplace = [max(0, sampto + s - siglen) for s in skew]

    return (startbyte, nreadsamples, blockfloorsamples, extraflatsamples, nanreplace)

def requiredbytenum(mode, fmt, nsamp):
    """
    Determine how many signal bytes are needed to read a file, or now many
    should be written to a file, for special formats.

    Input arguments:
    - mode: 'read' or 'write'
    - fmt: format
    - nsamp: number of samples

    It would be nice if read and write were the same, but fmt 311 for
    n_extra == 2 ruins it.
    """

    if fmt == '212':
        nbytes = math.ceil(nsamp*1.5)
    elif fmt in ['310', '311']:
        n_extra = nsamp % 3

        if n_extra == 2:
            if fmt == '310':
                nbytes = upround(nsamp * 4/3, 4)
            # 311
            else:
                if mode == 'read':
                    nbytes = math.ceil(nsamp * 4/3)
                # Have to write more bytes for wfdb c to work
                else:
                    nbytes = upround(nsamp * 4/3, 4)
        # 0 or 1
        else:
            nbytes = math.ceil(nsamp * 4/3 )
    else:
        nbytes = nsamp * bytespersample[fmt]

    return int(nbytes)


def getdatbytes(filename, dirname, pbdir, fmt, startbyte, nsamp):
    """
    Read bytes from a dat file, either local or remote, into a numpy array.
    Slightly misleading function name. Does not return bytes object. 
    Output argument dtype varies depending on fmt. Non-special fmts are
    read in their final required format. Special format are read as uint8.
    
    Input arguments:
    - nsamp: The total number of samples to read. Does NOT need to create whole blocks
      for special format. Any number of samples should be readable. But see below*.
    - startbyte: The starting byte to read from. * See below.

    * nsamp and startbyte should make it so that the bytes are read from the start
      of a byte block, even if sampfrom points into the middle of one. This will not
      be checked here. calc_read_params should ensure it.
    """

    # elementcount is the number of elements to read using np.fromfile (for local files)
    # bytecount is the number of bytes to read (for streaming files)

    if fmt == '212':
        bytecount = requiredbytenum('read', '212', nsamp)
        elementcount = bytecount
    elif fmt in ['310', '311']:
        bytecount = requiredbytenum('read', fmt, nsamp)
        elementcount = bytecount
    else:
        elementcount = nsamp
        bytecount = nsamp*bytespersample[fmt]

    # Local dat file
    if pbdir is None:
        fp = open(os.path.join(dirname, filename), 'rb')
        fp.seek(startbyte)

        # Read file using corresponding dtype
        # Cast to int64 for further processing
        sigbytes = np.fromfile(fp, dtype=np.dtype(dataloadtypes[fmt]), count=elementcount).astype('int')

        fp.close()

    # Stream dat file from physiobank
    # Same output as above np.fromfile.
    else:
        sigbytes = downloads.streamdat(filename, pbdir, fmt, bytecount, startbyte, dataloadtypes)

    return sigbytes


def bytes2samples(sigbytes, nsamp, fmt):
    """
    Converts loaded uint8 blocks into samples for special formats
    """
    if fmt == '212':
        # Easier to process when dealing with whole blocks
        if nsamp % 2:
            nsamp = nsamp + 1
            addedsamps = 1
            sigbytes = np.append(sigbytes, np.zeros(1, dtype='uint8'))
        else:
            addedsamps = 0

        sig = np.zeros(nsamp, dtype='int64')

        # One sample pair is stored in one byte triplet.
        
        # Even numbered samples
        sig[0::2] = sigbytes[0::3] + 256 * np.bitwise_and(sigbytes[1::3], 0x0f)
        # Odd numbered samples (len(sig) always >1 due to processing of whole blocks)
        sig[1::2] = sigbytes[2::3] + 256*np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)

        # Remove trailing sample read within the byte block if originally odd sampled
        if addedsamps:
            sig = sig[:-addedsamps]

        # Loaded values as unsigned. Convert to 2's complement form:
        # values > 2^11-1 are negative.
        sig[sig > 2047] -= 4096

    elif fmt == '310':
        # Easier to process when dealing with whole blocks
        if nsamp % 3:
            nsamp = upround(nsamp,3)
            addedsamps = nsamp % 3
            sigbytes = np.append(sigbytes, np.zeros(addedsamps, dtype='uint8'))
        else:
            addedsamps = 0

        # 1d array of actual samples. Fill the individual triplets.
        sig = np.zeros(nsamp, dtype='int64')

        # One sample triplet is stored in one byte quartet
        # First sample is 7 msb of first byte and 3 lsb of second byte.
        sig[0::3] = (sigbytes[0::4] >> 1)[0:len(sig[0::3])] + 128 * np.bitwise_and(sigbytes[1::4], 0x07)[0:len(sig[0::3])]
        # Second signal is 7 msb of third byte and 3 lsb of forth byte
        sig[1::3] = (sigbytes[2::4] >> 1)[0:len(sig[1::3])] + 128 * np.bitwise_and(sigbytes[3::4], 0x07)[0:len(sig[1::3])]
        # Third signal is 5 msb of second byte and 5 msb of forth byte
        sig[2::3] = np.bitwise_and((sigbytes[1::4] >> 3), 0x1f)[0:len(sig[2::3])] + 32 * np.bitwise_and(sigbytes[3::4] >> 3, 0x1f)[0:len(sig[2::3])]

        # Remove trailing samples read within the byte block if originally not 3n sampled
        if addedsamps:
            sig = sig[:-addedsamps]

        # Loaded values as unsigned. Convert to 2's complement form:
        # values > 2^9-1 are negative.
        sig[sig > 511] -= 1024
    
    elif fmt == '311':
        # Easier to process when dealing with whole blocks
        if nsamp % 3:
            nsamp = upround(nsamp,3)
            addedsamps = nsamp % 3
            sigbytes = np.append(sigbytes, np.zeros(addedsamps, dtype='uint8'))
        else:
            addedsamps = 0

        # 1d array of actual samples. Fill the individual triplets.
        sig = np.zeros(nsamp, dtype='int64')

        # One sample triplet is stored in one byte quartet
        # First sample is first byte and 2 lsb of second byte.
        sig[0::3] = sigbytes[0::4][0:len(sig[0::3])] + 256 * np.bitwise_and(sigbytes[1::4], 0x03)[0:len(sig[0::3])]
        # Second sample is 6 msb of second byte and 4 lsb of third byte
        sig[1::3] = (sigbytes[1::4] >> 2)[0:len(sig[1::3])] + 64 * np.bitwise_and(sigbytes[2::4], 0x0f)[0:len(sig[1::3])]
        # Third sample is 4 msb of third byte and 6 msb of forth byte
        sig[2::3] = (sigbytes[2::4] >> 4)[0:len(sig[2::3])] + 16 * np.bitwise_and(sigbytes[3::4], 0x7f)[0:len(sig[2::3])]
        
        # Remove trailing samples read within the byte block if originally not 3n sampled
        if addedsamps:
            sig = sig[:-addedsamps]

        # Loaded values as unsigned. Convert to 2's complement form:
        # values > 2^9-1 are negative.
        sig[sig > 511] -= 1024
    return sig


def skewsig(sig, skew, nsig, readlen, fmt, nanreplace, sampsperframe=None):
    """
    Skew the signal, insert nans and shave off end of array if needed.

    fmt is just for the correct nan value.
    sampsperframe is only used for skewing expanded signals.
    """
    if max(skew)>0:

        # Expanded frame samples. List of arrays. 
        if type(sig) == list:
            # Shift the channel samples
            for ch in range(nsig):
                if skew[ch]>0:
                    sig[ch][:readlen*sampsperframe[ch]] = sig[ch][skew[ch]*sampsperframe[ch]:]

            # Shave off the extra signal length at the end
            for ch in range(nsig):
                sig[ch] = sig[ch][:readlen*sampsperframe[ch]]

            # Insert nans where skewed signal overran dat file
            for ch in range(nsig):
                if nanreplace[ch]>0:
                    sig[ch][-nanreplace[ch]:] = digi_nan(fmt)
        # Uniform array
        else:
            # Shift the channel samples
            for ch in range(nsig):
                if skew[ch]>0:
                    sig[:readlen, ch] = sig[skew[ch]:, ch]
            # Shave off the extra signal length at the end
            sig = sig[:readlen, :]

            # Insert nans where skewed signal overran dat file
            for ch in range(nsig):
                if nanreplace[ch]>0:
                    sig[-nanreplace[ch]:, ch] = digi_nan(fmt)

    return sig

            
# Integrity check of signal shape after reading
def checksigdims(sig, readlen, nsig, sampsperframe):
    if type(sig) == np.ndarray:
        if sig.shape != (readlen, nsig):
            raise ValueError('Samples were not loaded correctly')
    else:
        if len(sig) != nsig:
            raise ValueError('Samples were not loaded correctly')
        for ch in range(nsig):
            if len(sig[ch]) != sampsperframe[ch] * readlen:
                raise ValueError('Samples were not loaded correctly')


# Bytes required to hold each sample (including wasted space) for
# different wfdb formats
bytespersample = {'8': 1, '16': 2, '24': 3, '32': 4, '61': 2,
                  '80': 1, '160': 2, '212': 1.5, '310': 4 / 3., '311': 4 / 3.}

# Data type objects for each format to load. Doesn't directly correspond
# for final 3 formats.
dataloadtypes = {'8': '<i1', '16': '<i2', '24': '<i3', '32': '<i4',
             '61': '>i2', '80': '<u1', '160': '<u2',
             '212': '<u1', '310': '<u1', '311': '<u1'}

#------------------- /Reading Signals -------------------#


# Return min and max digital values for each format type. Accepts lists.
def digi_bounds(fmt):
    if type(fmt) == list:
        digibounds = []
        for f in fmt:
            digibounds.append(digi_bounds(f))
        return digibounds

    if fmt == '80':
        return (-128, 127)
    elif fmt == '212':
        return (-2048, 2047)
    elif fmt == '16':
        return (-32768, 32767)
    elif fmt == '24':
        return (-8388608, 8388607)
    elif fmt == '32':
        return (-2147483648, 2147483647)
    
# Return nan value for the format type(s). 
def digi_nan(fmt):
    if type(fmt) == list:
        diginans = []
        for f in fmt:
            diginans.append(digi_nan(f))
        return diginans
        
    if fmt == '80':
        return -128
    if fmt == '310':
        return -512
    if fmt == '311':
        return -512
    elif fmt == '212':
        return -2048
    elif fmt == '16':
        return -32768
    elif fmt == '61':
        return -32768
    elif fmt == '160':
        return -32768
    elif fmt == '24':
        return -8388608
    elif fmt == '32':
        return -2147483648



reslevels = np.power(2, np.arange(0,33))
def estres(signals):
    """
    def estres(signals):

    Estimate the resolution of each signal in a multi-channel signal in bits. Maximum of 32 bits.
    Input arguments:
    - signals: A 2d numpy array representing a uniform multichannel signal, or a list of 1d numpy arrays
      representing multiple channels of signals with different numbers of samples per frame.
    """
    
    # Expanded sample signals. List of numpy arrays                
    if type(signals) == list:
        nsig = len(signals)
    # Uniform numpy array
    else:
        if signals.ndim ==1:
            nsig = 1
        else:
            nsig = signals.shape[1]
    res = []
        
    for ch in range(nsig):
        # Estimate the number of steps as the range divided by the minimum increment. 
        if type(signals) == list:
            sortedsig = np.sort(signals[ch])
        else:
            sortedsig = np.sort(signals[:,ch])
        
        min_inc = min(np.diff(sortedsig))
        
        if min_inc == 0:
            # Case where signal is flat. Resolution is 0.  
            res.append(0)
        else:
            nlevels = 1 + (sortedsig[-1]-sortedsig[0])/min_inc
            if nlevels>=reslevels[-1]:
                res.append(32)
            else:
                res.append(np.where(reslevels>=nlevels)[0][0])
            
    return res


# Return the most suitable wfdb format(s) to use given signal resolutions.
# If singlefmt == 1, the format for the maximum resolution will be returned.
def wfdbfmt(res, singlefmt = 1):

    if type(res) == list:
        # Return a single format
        if singlefmt == 1:
            res = [max(res)]*len(res)

        fmts = []
        for r in res:
            fmts.append(wfdbfmt(r))
        return fmts
    
    if res<=8:
        return '80'
    elif res<=12:
        return '212'
    elif res<=16:
        return '16'
    elif res<=24:
        return '24'
    else:
        return '32'

# Return the resolution of the WFDB format(s).
def wfdbfmtres(fmt):

    if type(fmt)==list:
        res = []
        for f in fmt:
            res.append(wfdbfmtres(f))
        return res
    
    if fmt in ['8', '80']:
        return 8
    elif fmt in ['310', '311']:
        return 10
    elif fmt == '212':
        return 12
    elif fmt in ['16', '61']:
        return 16
    elif fmt == '24':
        return 24
    elif fmt == '32':
        return 32
    else:
        raise ValueError('Invalid WFDB format.')

# Write a dat file.
# All bytes are written one at a time
# to avoid endianness issues.
def wrdatfile(filename, fmt, d_signals, byteoffset, expanded=False, e_d_signals=None, sampsperframe=None):
    f=open(filename,'wb')

    # Combine list of arrays into single array
    if expanded:
        nsig = len(e_d_signals)
        siglen = int(len(e_d_signals[0])/sampsperframe[0])
        # Effectively create MxN signal, with extra frame samples acting like extra channels
        d_signals = np.zeros((siglen, sum(sampsperframe)), dtype = 'int64')
        # Counter for channel number
        expand_ch = 0
        for ch in range(nsig):
            spf = sampsperframe[ch]
            for framenum in range(spf):
                d_signals[:, expand_ch] = e_d_signals[ch][framenum::spf]
                expand_ch = expand_ch + 1
    
    # This nsig is used for making list items.
    # Does not necessarily represent number of signals (ie. for expanded=True)
    nsig = d_signals.shape[1]

    if fmt == '80':
        # convert to 8 bit offset binary form
        d_signals = d_signals + 128
        # Concatenate into 1D
        d_signals = d_signals.reshape(-1)
        # Convert to unsigned 8 bit dtype to write
        bwrite = d_signals.astype('uint8')

    elif fmt == '212':

        # Each sample is represented by a 12 bit two's complement amplitude. 
        # The first sample is obtained from the 12 least significant bits of the first byte pair (stored least significant byte first). 
        # The second sample is formed from the 4 remaining bits of the first byte pair (which are the 4 high bits of the 12-bit sample) 
        # and the next byte (which contains the remaining 8 bits of the second sample). 
        # The process is repeated for each successive pair of samples. 

        # convert to 12 bit two's complement 
        d_signals[d_signals<0] = d_signals[d_signals<0] + 4096
        
        # Concatenate into 1D
        d_signals = d_signals.reshape(-1)

        nsamp = len(d_signals)
        # use this for byte processing
        processnsamp = nsamp

        # Odd numbered number of samples. Fill in extra blank for following byte calculation. 
        if processnsamp % 2:
            d_signals = np.concatenate([d_signals, np.array([0])])
            processnsamp +=1

        # The individual bytes to write
        bwrite = np.zeros([int(1.5*processnsamp)], dtype = 'uint8')

        # Fill in the byte triplets

        # Triplet 1 from lowest 8 bits of sample 1
        bwrite[0::3] = d_signals[0::2] & 255 
        # Triplet 2 from highest 4 bits of samples 1 (lower) and 2 (upper)
        bwrite[1::3] = ((d_signals[0::2] & 3840) >> 8) + ((d_signals[1::2] & 3840) >> 4)
        # Triplet 3 from lowest 8 bits of sample 2
        bwrite[2::3] = d_signals[1::2] & 255

        # If we added an extra sample for byte calculation, remove the last byte (don't write)
        if nsamp % 2:
            bwrite = bwrite[:-1]

    elif fmt == '16':
        # convert to 16 bit two's complement 
        d_signals[d_signals<0] = d_signals[d_signals<0] + 65536
        # Split samples into separate bytes using binary masks
        b1 = d_signals & [255]*nsig
        b2 = ( d_signals & [65280]*nsig ) >> 8
        # Interweave the bytes so that the same samples' bytes are consecutive 
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        bwrite = np.concatenate((b1, b2), axis=1)
        bwrite = bwrite.reshape((1,-1))[0]
        # Convert to unsigned 8 bit dtype to write
        bwrite = bwrite.astype('uint8')
    elif fmt == '24':
        # convert to 24 bit two's complement 
        d_signals[d_signals<0] = d_signals[d_signals<0] + 16777216
        # Split samples into separate bytes using binary masks
        b1 = d_signals & [255]*nsig
        b2 = ( d_signals & [65280]*nsig ) >> 8
        b3 = ( d_signals & [16711680]*nsig ) >> 16
        # Interweave the bytes so that the same samples' bytes are consecutive 
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        b3 = b3.reshape((-1, 1))
        bwrite = np.concatenate((b1, b2, b3), axis=1)
        bwrite = bwrite.reshape((1,-1))[0]
        # Convert to unsigned 8 bit dtype to write
        bwrite = bwrite.astype('uint8')
    
    elif fmt == '32':
        # convert to 32 bit two's complement 
        d_signals[d_signals<0] = d_signals[d_signals<0] + 4294967296
        # Split samples into separate bytes using binary masks
        b1 = d_signals & [255]*nsig
        b2 = ( d_signals & [65280]*nsig ) >> 8
        b3 = ( d_signals & [16711680]*nsig ) >> 16
        b4 = ( d_signals & [4278190080]*nsig ) >> 24
        # Interweave the bytes so that the same samples' bytes are consecutive 
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        b3 = b3.reshape((-1, 1))
        b4 = b4.reshape((-1, 1))
        bwrite = np.concatenate((b1, b2, b3, b4), axis=1)
        bwrite = bwrite.reshape((1,-1))[0]
        # Convert to unsigned 8 bit dtype to write
        bwrite = bwrite.astype('uint8')
    else:
        raise ValueError('This library currently only supports writing the following formats: 80, 16, 24, 32')
        
    # Byte offset in the file
    if byteoffset is not None and byteoffset>0:
        print('Writing file '+filename+' with '+str(byteoffset)+' empty leading bytes')
        bwrite = np.append(np.zeros(byteoffset, dtype = 'uint8'), bwrite)

    # Write the file
    bwrite.tofile(f)

    f.close()

# Returns the unique elements in a list in the order that they appear. 
# Also returns the indices of the original list that correspond to each output element. 
def orderedsetlist(fulllist):
    uniquelist = []
    original_inds = {}

    for i in range(0, len(fulllist)):
        item = fulllist[i]
        # new item
        if item not in uniquelist:
            uniquelist.append(item)
            original_inds[item] = [i]
        # previously seen item
        else:
            original_inds[item].append(i)
    return uniquelist, original_inds


# Round down to nearest <base>
def downround(x, base):
    return base * math.floor(float(x)/base)

# Round up to nearest <base>
def upround(x, base):
    return base * math.ceil(float(x)/base)
