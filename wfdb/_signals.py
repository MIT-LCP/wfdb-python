import numpy as np
import re
import os
import sys
import _headers

# All defined WFDB dat formats
datformats = ["80","212","16","24","32"]


# Class with signal methods
# To be inherited by WFDBrecord from records.py.
class Signals_Mixin():

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
            dmin, dmax = digi_bounds(fmt)
            
            chmin = min(self.d_signals[:,ch])
            chmax = max(self.d_signals[:,ch])
            if (chmin < dmin) or (chmax > dmax):
                sys.exit("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))
                    
        # Ensure that the checksums and initial value fields match the digital signal
        if self.nsig>0:
            realchecksum = calc_checksum(self.d_signals)
            if self.checksum != realchecksum:
                print("checksum field does not match actual checksum(s) of d_signals: ", realchecksum)
                sys.exit()
            realinitvalues = list(self.d_signals[0,:])
            if self.initvalues != realinitvalues:
                print("initvalue field does not match actual initvalue(s) of d_signals: ", realinitvalue)
                sys.exit()


    # Using p_signals and potentially fmt, compute optimal gain and baseline, and use them to perform ADC. 
    # Using the calculated results, set: siglen, nsig, gain, baseline, and d_signals. 
    # If fmt has been set, it will be 
    def set_adc_fields(self):

        request_approval('This method will set the following fields: siglen, nsig, gain, baseline, and d_signals. Continue?')

        # First, check the physical signal
        self.checkfield('p_signals') 
            
        # Fields potentially set via p_signals: nsig, siglen, d_signals, fmt, gain, baseline, initvalue, checksum
        
        self.nsig = p_signals.shape[1]
        self.siglen = p_signals.shape[0]
        
        # fmt is necessary to obtain gain and baseline.
        if self.fmt == 'None':
            res = estres(self.signals)
            self.fmt = self.nsig*[wfdbfmt(max(res))]
        else:
            self.checkfield('fmt') 

        
        # Check fmt before performing adc. No need to check gain and baseline which are automatically calculated.
        self.checkfield('fmt')
        
        # Do ADC and store value in d_signals. 
        print('Calculating optimal gain and baseline values to convert physical signal...')
        self.gain, self.baseline = adcparams(self.p_signals, self.fmt)
        print('Performing ADC and storing result in d_signals field...')
        self.d_signals = self.adc()
        
        if 'initvalue' in writefields:
            self.initvalue = list(self.d_signals[0,:])
        if 'checksum' in writefields:
            self.checksum = calc_checksum(self.d_signals)


    # Using gain, baseline, and d_signals, perform DAC.
    # Using the calculated result, set: nsig, siglen, p_signals.
    def set_dac_fields(self):

        request_approval('This method will set the following fields: siglen, nsig, and p_signals. Continue?')

        # First, check the digital signal
        self.checkfield('d_signals') 

        # Fields potentially set via d_signals: nsig, siglen, initvalue, checksum

        self.nsig = d_signals.shape[1]
        self.siglen = d_signals.shape[0]
        if 'initvalue' in writefields:
            self.initvalue = list(self.d_signals[0,:])
        if 'checksum' in writefields:
            self.checksum = calc_checksum(self.d_signals) 
        


# Return min and max digital values for each format type.
def digi_bounds(fmt):
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
    
# Return nan value for the format type (accepts lists) 
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

# Calculate the checksums of a multi-channel digital signal
def calc_checksum(signals):
    return list(np.sum(signals, 0) % 65536)