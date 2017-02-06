import numpy as np
import re
import os
import sys

# All defined WFDB dat formats
datformats = ["80","212","16","24","32"]


# Class with signal methods
# To be inherited by WFDBrecord from records.py.
class SignalsMixin():

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
            print('siglen and nsig do not match shape of d_signals')
            print('siglen: ', self.siglen)
            print('nsig: ', self.nsig)
            print('d_signals.shape: ', self.d_signals.shape)
            sys.exit()

        # For each channel (if any), make sure the digital format has no values out of bounds
        for ch in range(0, self.nsig):
            fmt = self.fmt[ch]
            dmin, dmax = digi_bounds(self.fmt[ch])
            
            chmin = min(self.d_signals[:,ch])
            chmax = max(self.d_signals[:,ch])
            if (chmin < dmin) or (chmax > dmax):
                sys.exit("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))
                    
        # Ensure that the checksums and initial value fields match the digital signal (if the fields are present)
        if self.nsig>0:
            if 'checksum' in writefields:
                realchecksum = self.calc_checksum()
                if self.checksum != realchecksum:
                    print("checksum field does not match actual checksum of d_signals: ", realchecksum)
                    sys.exit()
            if 'initvalue' in writefields:
                realinitvalue = list(self.d_signals[0,:])
                if self.initvalue != realinitvalue:
                    print("initvalue field does not match actual initvalue of d_signals: ", realinitvalue)
                    sys.exit()



    # Use properties of the p_signals field to set other fields: nsig, siglen
    # If do_dac == 1, the d_signals field will be used to perform digital to analogue conversion
    # to set the p_signals field, before p_signals is used. 
    # Regarding dac conversion:
    #     1. fmt, gain, and baseline must all be set in order to perform dac.
    #        Unlike with adc, there is no way to infer these fields.
    #     2. Using the fmt, gain and baseline fields, dac is performed, and p_signals is set.  
    def set_p_features(self, do_dac = 0):
        if do_dac == 1:
            self.checkfield('d_signals')
            self.checkfield('fmt')
            self.checkfield('adcgain')
            self.checkfield('baseline')

            # All required fields are present and valid. Perform DAC
            self.p_signals = self.dac()

        # Use p_signals to set fields
        self.checkfield('p_signals')
        self.siglen = self.p_signals.shape[0]
        self.nsig = self.p_signals.shape[1]


    # Use properties of the d_signals field to set other fields: nsig, siglen, fmt*, initvalue*, checksum* 
    # If do_adc == 1, the p_signals field will first be used to perform analogue to digital conversion
    # to set the d_signals field, before d_signals is used.
    # Regarding adc conversion:
    #     1. If fmt is unset, the most appropriate fmt for the signals will 
    #        be calculated and the field will be set. If singlefmt ==1, only one 
    #        fmt will be returned for all channels. If fmt is already set, it will be kept.
    #     2. If either gain or baseline are missing, optimal gains and baselines 
    #        will be calculated and the fields will be set. If they are already set, they will be kept.
    #     3. Using the fmt, gain and baseline fields, adc is performed, and d_signals is set.   
    def set_d_features(self, do_adc = 0, singlefmt = 1):

        # adc is performed.
        if do_adc == 1:
            self.checkfield('p_signals')

            # If there is no fmt, choose appropriate fmts. 
            if self.fmt is None:
                res = estres(self.p_signals)
                self.fmt = wfdbfmt(res, singlefmt)
            self.checkfield('fmt')

            # If either gain or baseline are missing, compute and set optimal values
            if self.adcgain is None or self.baseline is None:
                #print('Calculating optimal gain and baseline values to convert physical signal')
                self.adcgain, self.baseline = self.calculate_adcparams()
            self.checkfield('adcgain')
            self.checkfield('baseline')

            # All required fields are present and valid. Perform ADC
            #print('Performing ADC')
            self.d_signals = self.adc()

        # Use d_signals to set fields
        self.checkfield('d_signals')
        self.siglen = self.d_signals.shape[0]
        self.nsig = self.d_signals.shape[1]
        self.initvalue = list(self.d_signals[0,:])
        self.checksum = self.calc_checksum() 





    # Returns the analogue to digital conversion for the physical signal stored in p_signals. 
    # The p_signals, fmt, gain, and baseline fields must all be valid.
    def adc(self):
        
        # The digital nan values for each channel
        dnans = digi_nan(self.fmt)
        
        d_signals = self.p_signals * self.adcgain + self.baseline
        
        for ch in range(0, np.shape(self.p_signals)[1]):
            # Nan values 
            nanlocs = np.isnan(self.p_signals[:,ch])
            if nanlocs.any():
                d_signals[nanlocs,ch] = dnans[ch]
        
        d_signals = d_signals.astype('int64')

        return d_signals

    # Returns the digital to analogue conversion for a WFDBrecord signal stored in d_signals
    # The d_signals, fmt, gain, and baseline fields must all be valid.
    def dac(self):
        
        # The digital nan values for each channel
        dnans = digi_nan(self.fmt) 
        
        # Get nan indices, indicated by minimum value. 
        nanlocs = self.d_signals == dnans
        
        p_signal = (self.signals - self.baseline)/self.adcgain
            
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
                sys.exit('Chen, please fix this')
                    
            gains.append(gain)
            baselines.append(baseline)     
        
        return (gains, baselines)


    # Calculate the checksum(s) of the d_signals field
    def calc_checksum(self):
        return list(np.sum(self.d_signals, 0) % 65536)

    # Write each of the specified dat files
    def wrdatfiles(self):

        # Get the set of dat files to be written, and
        # the channels to be written to each file. 
        filenames, datchannels = orderedsetlist(self.filename)

        for i in range(0, len(filenames)):
            #print(filenames[i]) 
            #print(fmt[min(datchannels[filenames[i]])])
            #print(sig[:, min(datchannels[filenames[i]]):max(datchannels[filenames[i]])+1])

            wrdatfile(filenames[i], self.fmt[min(datchannels[filenames[i]])], 
                self.d_signals[:, min(datchannels[filenames[i]]):max(datchannels[filenames[i]])+1])



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
    elif fmt == '212':
        return -2048
    elif fmt == '16':
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
        sys.exit('Invalid WFDB format.')

# Write a dat file.
def wrdatfile(filename, fmt, d_signals):
    f=open(filename,'wb')
    
    # All bytes are written one at a time
    # to avoid endianness issues.

    nsig = d_signals.shape[1]

    if fmt == '80':
        # convert to 8 bit offset binary form
        d_signals = d_signals + 128
        # Convert to unsigned 8 bit dtype to write
        bwrite = d_signals.astype('uint8')

    elif fmt == '212':
        # convert to 12 bit two's complement 
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
        sys.exit('This library currently only supports the following formats: 80, 16, 24, 32')
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