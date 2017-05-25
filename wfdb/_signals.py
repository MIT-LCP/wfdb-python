import numpy as np
import os
import math
from . import downloads
import pdb
# All defined WFDB dat formats
datformats = ["80","212","16","24","32"]

# Class with signal methods
# To be inherited by Record from records.py.
class SignalsMixin(object):

    def wrdats(self):
    
        if not self.nsig:
            return
        # Get all the fields used to write the header
        # Assuming this method was called through wrsamp,
        # these will have already been checked in wrheader()
        writefields = self.getwritefields()

        # Check the validity of the d_signals field
        self.checkfield('d_signals')
        # Check the cohesion of the d_signals field against the other fields used to write the header
        self.checksignalcohesion(writefields)
        # Write each of the specified dat files
        self.wrdatfiles()


    # Check the cohesion of the d_signals field with the other fields used to write the record
    def checksignalcohesion(self, writefields):

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

    
    def set_p_features(self, do_dac = False):
        """
        Use properties of the p_signals field to set other fields: nsig, siglen
        If do_dac == True, the d_signals field will be used to perform digital to analogue conversion
        to set the p_signals field, before p_signals is used. 
        Regarding dac conversion:
          - fmt, gain, and baseline must all be set in order to perform dac.
            Unlike with adc, there is no way to infer these fields.
          - Using the fmt, gain and baseline fields, dac is performed, and p_signals is set.  
        """
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


    def set_d_features(self, do_adc = False, singlefmt = 1):
        """
        Use properties of the d_signals field to set other fields: nsig, siglen, initvalue, checksum, *(fmt, adcgain, baseline) 
        If do_adc == True, the p_signals field will first be used to perform analogue to digital conversion to set the d_signals 
        field, before d_signals is used. 

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
    def adc(self, expanded):
        
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

    
    def dac(self, expanded):
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
                chnanloc = self.e_d_signals == dnans
                p_signal.append((self.e_d_signals[ch] - self.baseline[ch])/float(self.adcgain[ch]))
                p_signal[ch][chnanlocs] = np.nan
        else:
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
                raise Exception('cx1111, please fix this!')
                    
            gains.append(gain)
            baselines.append(baseline)     
        
        return (gains, baselines)

    def calc_checksum(self):
        """
        Calculate the checksum(s) of the d_signals field
        """
        cs = list(np.sum(self.d_signals, 0) % 65536)
        cs = [int(c) for c in cs]
        return cs

    # Write each of the specified dat files
    def wrdatfiles(self):

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
        # Create a copy to prevent overwrite
        dsig = self.d_signals.copy()
        for fn in filenames:
            wrdatfile(fn, datfmts[fn], dsig[:, datchannels[fn][0]:datchannels[fn][-1]+1], datoffsets[fn])



#------------------- Reading Signals -------------------#

# Read the samples from a single segment record's associated dat file(s)
# 'channels', 'sampfrom', 'sampto', and 'smoothframes' are user desired input fields.
# All other input arguments are specifications of the segment
def rdsegment(filename, dirname, pbdir, nsig, fmt, siglen, byteoffset,
              sampsperframe, skew, sampfrom, sampto, channels, smoothframes):

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
        signals = np.empty([sampto-sampfrom, len(channels)], dtype = 'int64')

        # Read each wanted dat file and store signals
        for fn in w_filename:
            signals[:, out_datchannel[fn]] = rddat(fn, dirname, pbdir, w_fmt[fn], len(datchannel[fn]), 
                siglen, w_byteoffset[fn], w_sampsperframe[fn], w_skew[fn], sampfrom, sampto, smoothframes)[:, r_w_channel[fn]]
    
    # Return each sample in signals with multiple samples/frame, without smoothing.
    # Return a list of numpy arrays for each signal.
    else:
        signals=[]

        for fn in w_filename:
            signals.append(rddat())


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
    startbyte, nsampread, extralen, nanreplace, floorsamp = calc_read_params(fmt, siglen, byteoffset,
                                                                             skew, tsampsperframe,
                                                                             sampfrom, sampto)

    # Read the required bytes from the dat file.
    # Pad the end if necessary for skewed signals beyond the entire file.
    if extralen >0:
        # Non-special formats already load samples.
        # Special formats load uint8 which are not samples. Make sure their extra padded samples come in complete blocks.
       
        if fmt == '212':
            sigbytes = np.concatenate((getdatbytes(filename, dirname, pbdir, fmt, startbyte, nsampread), 
                                      np.empty(upround(extralen*tsampsperframe*bytespersample[fmt], 3),
                                               dtype=np.dtype(dataloadtypes[fmt]))))
        elif fmt in ['310', '311']:
            sigbytes = np.concatenate((getdatbytes(filename, dirname, pbdir, fmt, startbyte, nsampread),
                                      np.empty(upround(extralen*tsampsperframe*bytespersample[fmt], 4),
                                               dtype=np.dtype(dataloadtypes[fmt]))))
        else:
            sigbytes = np.concatenate((getdatbytes(filename, dirname, pbdir, fmt, startbyte, nsampread), 
                                      np.empty(extralen*tsampsperframe,
                                               dtype=np.dtype(dataloadtypes[fmt]))))
    else:
        sigbytes = getdatbytes(filename, dirname, pbdir, fmt, startbyte, nsampread)


    # Continue to process the read values into proper samples
    if fmt == '212':

        # No extra samples/frame. Obtain original uniform numpy array
        if tsampsperframe==nsig:

            # Intermediate number of samples to process. (like readlen)
            #processnsamp = readlen * tsampsperframe + floorsamp
            # Now has to take skew into account, which sigbytes already has.
            processnsamp = int(sigbytes.shape[0]*2/3)

            # For odd sampled records, imagine an extra sample and add an extra byte 
            # to simplify the processing step and remove the extra sample at the end. 

            # Now how will this work with the new processnsamp?
            # This cannot go here, if we have to expand samples, we have to do it to sigbytes too. Maybe.
            if processnsamp % 2:
                sigbytes = np.append(sigbytes, np.zeros(1, dtype='uint8'))
                processnsamp+=1



            # No extra samples/frame
            if tsampsperframe == nsig:
                # Turn the bytes into actual samples.
                # 1d array of actual samples

                # This has to be longer. Match the extra bytes read because of skew too.
                sig = np.zeros(processnsamp, dtype='int64')


                #pdb.set_trace()


                # One sample pair is stored in one byte triplet.
                # Even numbered samples
                sig[0::2] = sigbytes[0::3] + 256 * np.bitwise_and(sigbytes[1::3], 0x0f)
                if len(sig > 1):
                    # Odd numbered samples
                    sig[1::2] = sigbytes[2::3] + 256*np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
                # Remove extra leading sample read within the byte block
                if floorsamp:
                    sig = sig[floorsamp:]

                # Remove extra trailing sample read within the byte block if originally odd sampled
                if (sigbytes.shape[0]/3) % 2:
                    sig = sig[:-1]

                # Reshape into final array of samples
                sig = sig.reshape(-1, nsig)

                # Loaded values as unsigned. Convert to 2's complement form:
                # values > 2^11-1 are negative.
                sig[sig > 2047] -= 4096
                sig[sig > 2047] -= 4096


                #pdb.set_trace()

                # Adjust for skew
                if max(skew)>0:
                    # Skew the channels required
                    for ch in range(nsig):
                        if skew[ch]>0:
                            sig[:readlen, ch] = sig[skew[ch]:, ch]

                    # Shave off the extra signal length at the end
                    sig = sig[:readlen, :]

        # Extra frames present to be smoothed. Obtain averaged uniform numpy array
        elif smoothframes:
            pass
        # Extra frames present without wanting smoothing. Return all expanded samples.
        else:
            pass



        # ------------------------- original ------------------------- #

        # # use this for byte processing
        # processnsamp = nsamp

        # # For odd sampled records, imagine an extra sample and add an extra byte 
        # # to simplify the processing step and remove the extra sample at the end. 
        # if processnsamp % 2:
        #     sigbytes = np.append(sigbytes, 0).astype('uint64')
        #     processnsamp+=1


        # # No extra samples/frame
        # if tsampsperframe == nsig:  
        #     # Turn the bytes into actual samples.
        #     sig = np.zeros(processnsamp, dtype='int64')  # 1d array of actual samples
        #     # One sample pair is stored in one byte triplet.
        #     sig[0::2] = sigbytes[0::3] + 256 * \
        #         np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples
        #     if len(sig > 1):
        #         # Odd numbered samples
        #         sig[1::2] = sigbytes[2::3] + 256*np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
        #     # Remove extra leading sample read within the byte block
        #     if floorsamp:
        #         sig = sig[floorsamp:]

        #     # Remove extra trailing sample read within the byte block if originally odd sampled
        #     if nsamp % 2:
        #         sig = sig[:-1]

        #     # Reshape into final array of samples
        #     sig = sig.reshape(readlen, nsig)
        #     sig = sig.astype(int)
        #     # Loaded values as unsigned. Convert to 2's complement form: values
        #     # > 2^11-1 are negative.
        #     sig[sig > 2047] -= 4096


        # # At least one channel has multiple samples per frame. All extra samples are averaged
        # else:
        #     # Turn the bytes into actual samples.
        #     sigall = np.zeros(processnsamp)  # 1d array of actual samples
        #     sigall[0::2] = sigbytes[0::3] + 256 * \
        #         np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples

        #     if len(sigall) > 1:
        #         # Odd numbered samples
        #         sigall[1::2] = sigbytes[2::3] + 256 * \
        #             np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
        #     # Remove extra leading sample read within the byte block
        #     if floorsamp:
        #         sigall = sigall[floorsamp:]

        #     # Remove extra trailing sample read within the byte block if originally odd sampled
        #     if nsamp % 2:
        #         sigall = sigall[:-1]

        #     # Convert to int64 to be able to hold -ve values
        #     sigall = sigall.astype('int')
        #     # Loaded values as unsigned. Convert to 2's complement form: values
        #     # > 2^11-1 are negative.
        #     sigall[sigall > 2047] -= 4096
        #     # Give the average sample in each frame for each channel
        #     sig = np.zeros([readlen, nsig])
        #     for ch in range(0, nsig):
        #         if sampsperframe[ch] == 1:
        #             sig[:, ch] = sigall[
        #                 sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
        #         else:
        #             for frame in range(0, sampsperframe[ch]):
        #                 sig[:, ch] += sigall[sum(([0] + sampsperframe)
        #                                          [:ch + 1]) + frame::tsampsperframe]
        #     sig = (sig / sampsperframe).astype('int')


    elif fmt == '310':
        pass
    elif fmt == '311':
        pass
    # Simple format signals that are loaded as they are stored.
    else:
        # Adjust for skew, reshape, and consider sampsperframe.

        # No extra samples/frame. Obtain original uniform numpy array
        if tsampsperframe==nsig:

            # Reshape the array into the correct number of channels
            # At this point, cannot reshape(readlen, nsig) because there might
            # be extra skew samples
            sig = sigbytes.reshape(-1, nsig).astype('int')

            # Adjust for skew
            if max(skew)>0:
                # Skew the channels required
                
                for ch in range(nsig):
                    sig[:readlen, ch] = sig[skew[ch]:, ch]
                
                # Shave off the extra signal length at the end
                sig = sig[:readlen, :]

        # Extra frames present to be smoothed. Obtain averaged uniform numpy array
        elif smoothframes:
            
            # Allocate memory for signal
            sig = np.empty([readlen, nsig], dtype='int')

            # Samples are averaged within frames
            # Obtain the correct samples, factoring in skew
            for ch in range(nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigbytes[skew[ch]*tsampsperframe+sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(sampsperframe[ch]):
                        sig[:, ch] += sigbytes[skew[ch]*tsampsperframe+sum(([0] + sampsperframe)[:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe)

        # Extra frames present without wanting smoothing. Return all expanded samples.
        else:
            # List of 1d numpy arrays
            sig=[]
            for ch in range(nsig):
                chansig = np.zeros(readlen*sampsperframe[ch], dtype='int')
                for frame in range(sampsperframe[ch]):
                    chansig[frame::sampsperframe[ch]] = sigbytes[skew[ch]*tsampsperframe+sum(([0] + sampsperframe)[:ch + 1]) + frame::tsampsperframe]

                sig.append(chansig)

        # Adjust for byte offset formats
        if fmt == '80':
            sig = sig - 128
        elif fmt == '160':
            sig = sig - 32768

    # Insert nans where skewed signal overran dat file
    # Expanded samples per frame
    if type(sig) == list:
        for ch in range(nsig):
            if nanreplace[ch]>0:
                sig[ch][-nanreplace[ch]:] = digi_nan(fmt)
    # Uniform array
    else:
        for ch in range(nsig):
            if nanreplace[ch]>0:
                sig[-nanreplace[ch]:, ch] = digi_nan(fmt)
    

    return sig



def calc_read_params(fmt, siglen, byteoffset, skew, tsampsperframe, sampfrom, sampto):
    """
    Calculate parameters used to read and process the dat file
    
    Output arguments:
    - startbyte
    - nsampread
    - extralen
    - nanreplace
    - floorsamp

    Example Parameters:
    siglen=100, t = 4 (total samples/frame), skew = [0, 2, 4, 5]
    sampfrom=0, sampto=100 --> readlen = 100, nsampread = 100*t, extralen = 5, nanreplace = [0, 2, 4, 5]
    sampfrom=50, sampto=100 --> readlen = 50, nsampread = 50*t, extralen = 5, nanreplace = [0, 2, 4, 5]
    sampfrom=0, sampto=50 --> readlen = 50, nsampread = 55*t, extralen = 0, nanreplace = [0, 0, 0, 0]
    sampfrom=95, sampto=99 --> readlen = 4, nsampread = 5*t, extralen = 4, nanreplace = [0, 1, 3, 4]
    """

    # 1. Calculate the starting byte to read the dat file from. 
    startbyte = int(sampfrom*tsampsperframe*bytespersample[fmt]) + int(byteoffset)

    # The above formula needs to be adjusted for special fmts.
    # Special formats store samples in specific byte blocks.
    # The starting byte should be at the start of a block of 3 or 4.
    if fmt == '212':
        # Extra samples to read
        floorsamp = (startbyte - byteoffset) % 3  
        startbyte = startbyte - floorsamp
    elif fmt in ['310', '311']:
        floorsamp = (startbyte - byteoffset) % 4
        startbyte = startbyte - floorsamp
    else:
        floorsamp=0
    # Question: Why do we need floorsamp to collect 'extra bytes'?
    # Because nsampread may bring us partial way into a byte triplet or quartet for special formats.
    # Sometimes you need to stretch bytes.

    # Question: Why did startbyte go back with floorsamp?
    # Because we have to read from the start of a block.

    # Question: Why are these values the same?



    # 2. Total number of samples to be read from the dat file (including discarded ones)
    # Have to read extra samples if there is a skew, but can't read beyond the limits of the
    # dat file.
    nsampread = (min(sampto+max(skew), siglen) - sampfrom )*tsampsperframe

    # If the skew requires samples beyond the dat file, pad the bytes with
    # zeros, and keep track of channels insert nans into.
    
    # 3. The extra signal length desired beyond the dat file
    extralen = max(0, sampto + max(skew) - siglen)
    
    # 4. The number of samples at the end of each signal to replace with nans
    nanreplace = [max(0, sampto + s - siglen) for s in skew]

    return (startbyte, nsampread, extralen, nanreplace, floorsamp)


def getdatbytes(filename, dirname, pbdir, fmt, startbyte, nsamp):
    """
    Read bytes from a dat file, either local or remote
    
    Input arguments:
    - nsamp: The total number of samples to read
    - startbyte: The starting byte to read
    """

    # count is the number of elements to read using np.fromfile
    # bytecount is the number of bytes to read
    if fmt == '212':
        bytecount = int(np.ceil((nsamp) * 1.5))
        count = bytecount
    elif fmt == '310':
        bytecount = int(((nsamp) + 2) / 3.) * 4
        if (nsamp - 1) % 3 == 0:
            bytecount -= 2
        count = bytecount
    elif fmt == '311':
        bytecount = int((nsamp - 1) / 3.) + nsamp + 1
        count = bytecount
    else:
        count = nsamp
        bytecount = nsamp*bytespersample[fmt] 

    # Local dat file
    if pbdir is None:
        fp = open(os.path.join(dirname, filename), 'rb')
        fp.seek(startbyte)

        # Read file using corresponding dtype
        sigbytes = np.fromfile(fp, dtype=np.dtype(dataloadtypes[fmt]), count=count)

        # For special formats that were read as unsigned 1 byte blocks to be further processed,
        # convert dtype from uint8 to uint64. Why? We are not reshaping these. We are sampling from them.
        #if fmt in ['212', '310', '311']:
        #    sigbytes = sigbytes.astype('uint')

        fp.close()

    # Stream dat file from physiobank
    else:
        sigbytes = downloads.streamdat(filename, pbdir, fmt, bytecount, startbyte, dataloadtypes)

    return sigbytes


def bytes2sig(sigbytes):
    pass


# Read digital samples from a wfdb signal file
# Returns the signal and the number of bytes read.
def processwfdbbytes(filename, dirname, pbdir, fmt, startbyte, readlen, nsig, sampsperframe, floorsamp=0):
    """
    Read digital samples from a wfdb dat file

    Input variables:
    - filename: the name of the dat file.
    - dirname: the full directory where the dat file is located, if the dat file is local.
    - pbdir: the physiobank directory where the dat file is located, if the dat file is remote.
    - fmt: the format of the dat file
    - startbyte: the starting byte to skip to
    - readlen: the length of the signal to be read, in samples.
    - nsig: 
    - floorsamp: the extra sample index used to read special formats (212, 310, 311).
    
    """

    # Total number of samples per frame
    tsampsperframe = sum(sampsperframe)

    # Total number of signal samples to be collected (including discarded ones)
    nsamp = readlen * tsampsperframe + floorsamp

    # Reading the dat file into np array and reshaping. Formats 212, 310, and 311 need special processing.
    # Note that for these formats with multi samples/frame, have to convert
    # bytes to samples before returning average frame values.
    if fmt == '212':
        # Read the bytes from the file as unsigned integer blocks
        sigbytes, bytesloaded = getdatbytes(filename, dirname, pbdir, fmt, nsamp, startbyte)

        # use this for byte processing
        processnsamp = nsamp

        # For odd sampled records, imagine an extra sample and add an extra byte 
        # to simplify the processing step and remove the extra sample at the end. 
        if processnsamp % 2:
            sigbytes = np.append(sigbytes, 0).astype('uint64')
            processnsamp+=1

        # No extra samples/frame
        if tsampsperframe == nsig:  
            # Turn the bytes into actual samples.
            sig = np.zeros(processnsamp)  # 1d array of actual samples
            # One sample pair is stored in one byte triplet.
            sig[0::2] = sigbytes[0::3] + 256 * \
                np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples
            if len(sig > 1):
                # Odd numbered samples
                sig[1::2] = sigbytes[2::3] + 256*np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]

            # Remove extra sample read if originally odd sampled
            if nsamp % 2:
                sig = sig[:-1]

            # Reshape into final array of samples
            sig = sig.reshape(readlen, nsig)
            sig = sig.astype(int)
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^11-1 are negative.
            sig[sig > 2047] -= 4096
        # At least one channel has multiple samples per frame. All extra samples are averaged
        else:
            # Turn the bytes into actual samples.
            sigall = np.zeros(processnsamp)  # 1d array of actual samples
            sigall[0::2] = sigbytes[0::3] + 256 * \
                np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples

            if len(sigall) > 1:
                # Odd numbered samples
                sigall[1::2] = sigbytes[2::3] + 256 * \
                    np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]

            # Remove extra sample read if originally odd sampled
            if nsamp % 2:
                sigall = sigall[:-1]

            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^11-1 are negative.
            sigall[sigall > 2047] -= 4096
            # Give the average sample in each frame for each channel
            sig = np.zeros([readlen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')
    # Three 10 bit samples packed into 4 bytes with 2 bits discarded
    elif fmt == '310':  

        # Read the bytes from the file as unsigned integer blocks
        sigbytes, bytesloaded = getdatbytes(filename, dirname, pbdir, fmt, nsamp, startbyte)

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
            sig = sig.reshape(readlen, nsig)
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
            sig = np.zeros([readlen, nsig])
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
        
        # Read the bytes from the file as unsigned integer blocks
        sigbytes, bytesloaded = getdatbytes(filename, dirname, pbdir, fmt, nsamp, startbyte)
        
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
            sig = sig.reshape(readlen, nsig)
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
            sig = np.zeros([readlen, nsig])
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

        # Read the dat file in the specified format
        sigbytes, bytesloaded = getdatbytes(filename, dirname, pbdir, fmt, nsamp, startbyte)

        # No extra samples/frame. Just reshape results.
        if tsampsperframe == nsig:  
            sig = sigbytes.reshape(readlen, nsig).astype('int')
        # At least one channel has multiple samples per frame. Extra samples are averaged.
        else:  
            # Keep the first sample in each frame for each channel
            sig = np.empty([readlen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigbytes[sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigbytes[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

        # Adjust for byte offset formats
        if fmt == '80':
            sig = sig - 128
        elif fmt == '160':
            sig = sig - 32768

    return sig, bytesloaded



def skewsignal(sig, skew, filename, dirname, pbdir, nsig, fmt, siglen, sampfrom, sampto, startbyte, 
    bytesloaded, byteoffset, sampsperframe):

    # Total number of samples per frame.
    tsampsperframe = sum(sampsperframe)

    if max(skew) > 0:
        # Array of samples to fill in the final samples of the skewed channels.
        extrasig = np.empty([max(skew), nsig])
        extrasig.fill(digi_nan(fmt))

        # Load the extra samples if the end of the file hasn't been reached.
        if siglen - (sampto - sampfrom):
            startbyte = startbyte + bytesloaded
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

            # The length of extra signals to be loaded
            extraloadlen = min(siglen - (sampto - sampfrom), max(skew))

            # Array of extra loaded samples
            extraloadedsig = processwfdbbytes(filename, dirname, pbdir,
                fmt, startbyte, extraloadlen, nsig, sampsperframe, floorsamp)[0]  

            # Fill in the extra loaded samples
            extrasig[:extraloadedsig.shape[0], :] = extraloadedsig

        # Fill in the skewed channels with the appropriate values.
        for ch in range(0, nsig):
            if skew[ch] > 0:
                sig[:-skew[ch], ch] = sig[skew[ch]:, ch]
                sig[-skew[ch]:, ch] = extrasig[:skew[ch], ch]
    return sig



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


# Estimate the resolution of each signal in a multi-channel signal in bits. Maximum of 32 bits. 
reslevels = np.power(2, np.arange(0,33))
def estres(signals):
    
    if signals.ndim ==1:
        nsig = 1
    else:
        nsig = signals.shape[1]
    res = nsig*[]
    
    for ch in range(0, nsig):
        # Estimate the number of steps as the range divided by the minimum increment. 
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
                res.append(np.where(reslevels>nlevels)[0][0])
            
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
def wrdatfile(filename, fmt, d_signals, byteoffset):
    f=open(filename,'wb')  
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
