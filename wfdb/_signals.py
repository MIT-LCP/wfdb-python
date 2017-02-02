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

        # For each channel, make sure there are no values out of bounds for the digital format
        for ch in range(0, self.nsig):
            fmt = self.fmt[ch]
            dmin, dmax = digi_bounds(fmt)
            
            chmin = min(self.d_signals[:,ch])
            chmax = max(self.d_signals[:,ch])
            if (chmin < dmin) or (chmax > dmax):
                sys.exit("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))
                    

        























    # Use the chosen signals fields to potentially set missing required fields
    # This method WILL overwrite fields
    # Separate out the adc? 
    def setsignalfeatures(self, writefields):
        
        # Physical signal
        if usesignals == 1:
            
            # First, check the signal
            errormsg = self.checkfield(self, p_signals) 
            if errormsg:
                sys.exit(errormsg)
                
            # Fields potentially set via p_signals: nsig, siglen, d_signals, fmt, gain, baseline, initvalue, checksum
            
            self.nsig = p_signals.shape[1]
            self.siglen = p_signals.shape[0]
            
            # fmt is necessary to obtain gain and baseline.
            if self.fmt == 'None':
                res = estres(self.signals)
                self.fmt = self.nsig*[wfdbfmt(max(res))]
            else:
                errormsg = self.checkfield('fmt', 0) 
                if errormsg:
                    sys.exit(errormsg)
            
            # Check fmt before performing adc. No need to check gain and baseline which are automatically calculated.
            self.checkfield('fmt', 0)
            
            # Do ADC and store value in d_signals. 
            print('Calculating optimal gain and baseline values to convert physical signal...')
            self.gain, self.baseline = adcparams(self.p_signals, self.fmt)
            print('Performing ADC and storing result in d_signals field...')
            self.d_signals = self.adc()
            
            if 'initvalue' in writefields:
                self.initvalue = list(self.d_signals[0,:])
            if 'checksum' in writefields:
                self.checksum = calc_checksum(self.d_signals)

        # Digital signal
        elif usesignals == 2:

            # First, check the signal
            errormsg = self.checkfield(self, d_signals) 
            if errormsg:
                sys.exit(errormsg)
                
            # Fields potentially set via d_signals: nsig, siglen, initvalue, checksum

            self.nsig = d_signals.shape[1]
            self.siglen = d_signals.shape[0]
            if 'initvalue' in writefields:
                self.initvalue = list(self.d_signals[0,:])
            if 'checksum' in writefields:
                self.checksum = calc_checksum(self.d_signals) 
        
        # No need to call checkfieldcohesion here. It will be called at the end of wrheader. 
        
        return


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
    