import math
import numpy as np
import os

from . import download
import pdb
# WFDB dat formats - https://www.physionet.org/physiotools/wag/signal-5.htm
SIMPLE_FMTS = ['80', '16', '24', '32']
SPECIAL_FMTS = ['212', '310', '311']
DAT_FMTS = SIMPLE_FMTS + SPECIAL_FMTS


class SignalMixin(object):
    """
    Mixin class with signal methods. Inherited by Record class.
    """

    def wr_dats(self, expanded, write_dir):
        # Write all dat files associated with a record
        # expanded=True to use e_d_signal instead of d_signal

        if not self.n_sig:
            return

        # Get all the fields used to write the header
        # Assuming this method was called through wrsamp,
        # these will have already been checked in wrheader()
        write_fields = self.get_write_fields()

        if expanded:
            # Using list of arrays e_d_signal
            self.check_field('e_d_signal')
        else:
            # Check the validity of the d_signal field
            self.check_field('d_signal')

        # Check the cohesion of the d_signal field against the other
        # fields used to write the header
        self.check_sig_cohesion(write_fields, expanded)

        # Write each of the specified dat files
        self.wr_dat_files(expanded=expanded, write_dir=write_dir)




    def check_sig_cohesion(self, write_fields, expanded):
        """
        Check the cohesion of the d_signal/e_d_signal field with the other
        fields used to write the record
        """
        # Using list of arrays e_d_signal
        if expanded:
            # Set default samps_per_frame
            spf = self.samps_per_frame
            for ch in range(len(spf)):
                if spf[ch] is None:
                    spf[ch] = 1

            # Match the actual signal shape against stated length and number of channels
            if self.n_sig != len(self.e_d_signal):
                raise ValueError('n_sig does not match the length of e_d_signal')
            for ch in range(self.n_sig):
                if len(self.e_d_signal[ch]) != spf[ch]*self.sig_len:
                    raise ValueError('Length of channel '+str(ch)+'does not match samps_per_frame['+str(ch+']*sig_len'))

            # For each channel (if any), make sure the digital format has no values out of bounds
            for ch in range(self.n_sig):
                fmt = self.fmt[ch]
                dmin, dmax = digi_bounds(self.fmt[ch])

                chmin = min(self.e_d_signal[ch])
                chmax = max(self.e_d_signal[ch])
                if (chmin < dmin) or (chmax > dmax):
                    raise IndexError("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))

            # Ensure that the checksums and initial value fields match the digital signal (if the fields are present)
            if self.n_sig > 0:
                if 'checksum' in write_fields:
                    realchecksum = self.calc_checksum(expanded)
                    if self.checksum != realchecksum:
                        print("The actual checksum of e_d_signal is: ", realchecksum)
                        raise ValueError("checksum field does not match actual checksum of e_d_signal")
                if 'init_value' in write_fields:
                    realinit_value = [self.e_d_signal[ch][0] for ch in range(self.n_sig)]
                    if self.init_value != realinit_value:
                        print("The actual init_value of e_d_signal is: ", realinit_value)
                        raise ValueError("init_value field does not match actual init_value of e_d_signal")

        # Using uniform d_signal
        else:
            # Match the actual signal shape against stated length and number of channels
            if (self.sig_len, self.n_sig) != self.d_signal.shape:
                print('sig_len: ', self.sig_len)
                print('n_sig: ', self.n_sig)
                print('d_signal.shape: ', self.d_signal.shape)
                raise ValueError('sig_len and n_sig do not match shape of d_signal')

            # For each channel (if any), make sure the digital format has no values out of bounds
            for ch in range(self.n_sig):
                fmt = self.fmt[ch]
                dmin, dmax = digi_bounds(self.fmt[ch])

                chmin = min(self.d_signal[:,ch])
                chmax = max(self.d_signal[:,ch])
                if (chmin < dmin) or (chmax > dmax):
                    raise IndexError("Channel "+str(ch)+" contain values outside allowed range ["+str(dmin)+", "+str(dmax)+"] for fmt "+str(fmt))

            # Ensure that the checksums and initial value fields match the digital signal (if the fields are present)
            if self.n_sig>0:
                if 'checksum' in write_fields:
                    realchecksum = self.calc_checksum()
                    if self.checksum != realchecksum:
                        print("The actual checksum of d_signal is: ", realchecksum)
                        raise ValueError("checksum field does not match actual checksum of d_signal")
                if 'init_value' in write_fields:
                    realinit_value = list(self.d_signal[0,:])
                    if self.init_value != realinit_value:
                        print("The actual init_value of d_signal is: ", realinit_value)
                        raise ValueError("init_value field does not match actual init_value of d_signal")


    def set_p_features(self, do_dac=False, expanded=False):
        """
        Use properties of the physical signal field to set the following
        features: n_sig, sig_len.

        Parameters
        ----------
        do_dac : bool
            Whether to use the digital signal field to perform dac
            conversion to get the physical signal field beforehand.
        expanded : bool
            Whether to use the `e_p_signal` or `p_signal` field. If
            True, the `samps_per_frame` attribute is also required.

        Notes
        -----
        Regarding dac conversion:
          - fmt, gain, and baseline must all be set in order to perform
            dac.
          - Unlike with adc, there is no way to infer these fields.
          - Using the fmt, gain and baseline fields, dac is performed,
            and (e_)p_signal is set.

        *Developer note: Seems this function will be very infrequently used.
         The set_d_features function seems far more useful.

        """

        if expanded:
            if do_dac:
                self.check_field('e_d_signal')
                self.check_field('fmt', 'all')
                self.check_field('adc_gain', 'all')
                self.check_field('baseline', 'all')
                self.check_field('samps_per_frame', 'all')

                # All required fields are present and valid. Perform DAC
                self.e_p_signal = self.dac(expanded)

            # Use e_p_signal to set fields
            self.check_field('e_p_signal', channels = 'all')
            self.sig_len = int(len(self.e_p_signal[0])/self.samps_per_frame[0])
            self.n_sig = len(self.e_p_signal)
        else:
            if do_dac:
                self.check_field('d_signal')
                self.check_field('fmt', 'all')
                self.check_field('adc_gain', 'all')
                self.check_field('baseline', 'all')

                # All required fields are present and valid. Perform DAC
                self.p_signal = self.dac()

            # Use p_signal to set fields
            self.check_field('p_signal')
            self.sig_len = self.p_signal.shape[0]
            self.n_sig = self.p_signal.shape[1]


    def set_d_features(self, do_adc=False, single_fmt=True, expanded=False):
        """
        Use properties of the digital signal field to set the following
        features: n_sig, sig_len, init_value, checksum, and possibly
        *(fmt, adc_gain, baseline).

        Parameters
        ----------
        do_adc : bools
            Whether to use the physical signal field to perform adc
            conversion to get the digital signal field beforehand.
        single_fmt : bool
            Whether to use a single digital format during adc, if it is
            performed.
        expanded : bool
            Whether to use the `e_d_signal` or `d_signal` field.

        Notes
        -----
        Regarding adc conversion:
        - If fmt is unset:
          - Neither adc_gain nor baseline may be set. If the digital values
            used to store the signal are known, then the file format should
            also be known.
          - The most appropriate fmt for the signals will be calculated and the
            `fmt` attribute will be set. Given that neither `adc_gain` nor
            `baseline` is allowed to be set, optimal values for those fields will
            then be calculated and set as well.
        - If fmt is set:
          - If both adc_gain and baseline are unset, optimal values for those
            fields will be calculated the fields will be set.
          - If both adc_gain and baseline are set, the function will continue.
          - If only one of adc_gain and baseline are set, this function will
            raise an error. It makes no sense to know only one of those fields.
        - ADC will occur after valid values for fmt, adc_gain, and baseline are
          present, using all three fields.

        """
        if expanded:
            # adc is performed.
            if do_adc:
                self.check_field('e_p_signal', channels='all')

                # If there is no fmt set it, adc_gain, and baseline
                if self.fmt is None:
                    # Make sure that neither adc_gain nor baseline are set
                    if self.adc_gain is not None or self.baseline is not None:
                        raise Exception('If fmt is not set, gain and baseline may not be set either.')
                    # Choose appropriate fmts based on estimated signal resolutions.
                    res = est_res(self.e_p_signal)
                    self.fmt = wfdbfmt(res, single_fmt)
                # If there is a fmt set
                else:
                    self.check_field('fmt', 'all')
                    # Neither field set
                    if self.adc_gain is None and self.baseline is None:
                        # Calculate and set optimal gain and baseline values to convert physical signals
                        self.adc_gain, self.baseline = self.calc_adc_params()
                    # Exactly one field set
                    elif (self.adc_gain is None) ^ (self.baseline is None):
                        raise Exception('If fmt is set, gain and baseline should both be set or not set.')

                self.check_field('adc_gain', 'all')
                self.check_field('baseline', 'all')

                # All required fields are present and valid. Perform ADC
                self.d_signal = self.adc(expanded)

            # Use e_d_signal to set fields
            self.check_field('e_d_signal', channels='all')
            self.sig_len = int(len(self.e_d_signal[0])/self.samps_per_frame[0])
            self.n_sig = len(self.e_d_signal)
            self.init_value = [sig[0] for sig in self.e_d_signal]
            self.checksum = self.calc_checksum(expanded)
        else:
            # adc is performed.
            if do_adc:
                self.check_field('p_signal')

                # If there is no fmt set
                if self.fmt is None:
                    # Make sure that neither adc_gain nor baseline are set
                    if self.adc_gain is not None or self.baseline is not None:
                        raise Exception('If fmt is not set, gain and baseline may not be set either.')
                    # Choose appropriate fmts based on estimated signal resolutions.
                    res = est_res(self.p_signal)
                    self.fmt = wfdbfmt(res, single_fmt)
                    # Calculate and set optimal gain and baseline values to convert physical signals
                    self.adc_gain, self.baseline = self.calc_adc_params()

                # If there is a fmt set
                else:
                    self.check_field('fmt', 'all')
                    # Neither field set
                    if self.adc_gain is None and self.baseline is None:
                        # Calculate and set optimal gain and baseline values to convert physical signals
                        self.adc_gain, self.baseline = self.calc_adc_params()
                    # Exactly one field set
                    elif (self.adc_gain is None) ^ (self.baseline is None):
                        raise Exception('If fmt is set, gain and baseline should both be set or not set.')

                self.check_field('adc_gain', 'all')
                self.check_field('baseline', 'all')

                # All required fields are present and valid. Perform ADC
                self.d_signal = self.adc()

            # Use d_signal to set fields
            self.check_field('d_signal')
            self.sig_len = self.d_signal.shape[0]
            self.n_sig = self.d_signal.shape[1]
            self.init_value = list(self.d_signal[0,:])
            self.checksum = self.calc_checksum()


    def adc(self, expanded=False, inplace=False):
        """
        Performs analogue to digital conversion of the physical signal stored
        in p_signal if expanded is False, or e_p_signal if expanded is True.

        The p_signal/e_p_signal, fmt, gain, and baseline fields must all be
        valid.

        If inplace is True, the adc will be performed inplace on the variable,
        the d_signal/e_d_signal attribute will be set, and the
        p_signal/e_p_signal field will be set to None.

        Parameters
        ----------
        expanded : bool, optional
            Whether to transform the `e_p_signal` attribute (True) or
            the `p_signal` attribute (False).
        inplace : bool, optional
            Whether to automatically set the object's corresponding
            digital signal attribute and set the physical
            signal attribute to None (True), or to return the converted
            signal as a separate variable without changing the original
            physical signal attribute (False).

        Returns
        -------
        d_signal : numpy array, optional
            The digital conversion of the signal. Either a 2d numpy
            array or a list of 1d numpy arrays.

        Examples:
        ---------
        >>> import wfdb
        >>> record = wfdb.rdsamp('sample-data/100')
        >>> d_signal = record.adc()
        >>> record.adc(inplace=True)
        >>> record.dac(inplace=True)

        """

        # The digital nan values for each channel
        dnans = digi_nan(self.fmt)

        # To do: choose the minimum return res needed
        intdtype = 'int64'

        # Do inplace conversion and set relevant variables.
        if inplace:
            if expanded:
                for ch in range(self.n_sig):
                    # nan locations for the channel
                    ch_nanlocs = np.isnan(self.e_p_signal[ch])
                    np.multiply(self.e_p_signal[ch], self.adc_gain[ch], self.e_p_signal[ch])
                    np.add(e_p_signal[ch], self.baseline[ch], self.e_p_signal[ch])
                    self.e_p_signal[ch] = self.e_p_signal[ch].astype(intdtype, copy=False)
                    self.e_p_signal[ch][ch_nanlocs] = dnans[ch]
                self.e_d_signal = self.e_p_signal
                self.e_p_signal = None
            else:
                nanlocs = np.isnan(self.p_signal)
                np.multiply(self.p_signal, self.adc_gain, self.p_signal)
                np.add(self.p_signal, self.baseline, self.p_signal)
                self.p_signal = self.p_signal.astype(intdtype, copy=False)
                self.d_signal = self.p_signal
                self.p_signal = None

        # Return the variable
        else:
            if expanded:
                d_signal = []
                for ch in range(self.n_sig):
                    # nan locations for the channel
                    ch_nanlocs = np.isnan(self.e_p_signal[ch])
                    ch_d_signal = self.e_p_signal.copy()
                    np.multiply(ch_d_signal, self.adc_gain[ch], ch_d_signal)
                    np.add(ch_d_signal, self.baseline[ch], ch_d_signal)
                    ch_d_signal = ch_d_signal.astype(intdtype, copy=False)
                    ch_d_signal[ch_nanlocs] = dnans[ch]
                    d_signal.append(ch_d_signal)

            else:
                nanlocs = np.isnan(self.p_signal)
                # Cannot cast dtype to int now because gain is float.
                d_signal = self.p_signal.copy()
                np.multiply(d_signal, self.adc_gain, d_signal)
                np.add(d_signal, self.baseline, d_signal)
                d_signal = d_signal.astype(intdtype, copy=False)

                if nanlocs.any():
                    for ch in range(d_signal.shape[1]):
                        if nanlocs[:,ch].any():
                            d_signal[nanlocs[:,ch],ch] = dnans[ch]

            return d_signal


    def dac(self, expanded=False, return_res=64, inplace=False):
        """
        Performs the digital to analogue conversion of the signal stored
        in d_signal if expanded is False, or e_d_signal if expanded is True.

        The d_signal/e_d_signal, fmt, gain, and baseline fields must all be
        valid.

        If inplace is True, the dac will be performed inplace on the variable,
        the p_signal/e_p_signal attribute will be set, and the
        d_signal/e_d_signal field will be set to None.

        Parameters
        ----------
        expanded : bool, optional
            Whether to transform the `e_d_signal attribute` (True) or
            the `d_signal` attribute (False).
        inplace : bool, optional
            Whether to automatically set the object's corresponding
            physical signal attribute and set the digital signal
            attribute to None (True), or to return the converted
            signal as a separate variable without changing the original
            digital signal attribute (False).

        Returns
        -------
        p_signal : numpy array, optional
            The physical conversion of the signal. Either a 2d numpy
            array or a list of 1d numpy arrays.

        Examples
        --------
        >>> import wfdb
        >>> record = wfdb.rdsamp('sample-data/100', physical=False)
        >>> p_signal = record.dac()
        >>> record.dac(inplace=True)
        >>> record.adc(inplace=True)

        """

        # The digital nan values for each channel
        dnans = digi_nan(self.fmt)

        # Get the appropriate float dtype
        if return_res == 64:
            floatdtype = 'float64'
        elif return_res == 32:
            floatdtype = 'float32'
        else:
            floatdtype = 'float16'

        # Do inplace conversion and set relevant variables.
        if inplace:
            if expanded:
                for ch in range(self.n_sig):
                    # nan locations for the channel
                    ch_nanlocs = self.e_d_signal[ch] == dnans[ch]
                    self.e_d_signal[ch] = self.e_d_signal[ch].astype(floatdtype, copy=False)
                    np.subtract(self.e_d_signal[ch], self.baseline[ch], self.e_d_signal[ch])
                    np.divide(self.e_d_signal[ch], self.adc_gain[ch], self.e_d_signal[ch])
                    self.e_d_signal[ch][ch_nanlocs] = np.nan
                self.e_p_signal = self.e_d_signal
                self.e_d_signal = None
            else:
                nanlocs = self.d_signal == dnans
                # Do float conversion immediately to avoid potential under/overflow
                # of efficient int dtype
                self.d_signal = self.d_signal.astype(floatdtype, copy=False)
                np.subtract(self.d_signal, self.baseline, self.d_signal)
                np.divide(self.d_signal, self.adc_gain, self.d_signal)
                self.d_signal[nanlocs] = np.nan
                self.p_signal = self.d_signal
                self.d_signal = None

        # Return the variable
        else:
            if expanded:
                p_signal = []
                for ch in range(self.n_sig):
                    # nan locations for the channel
                    ch_nanlocs = self.e_d_signal[ch] == dnans[ch]
                    ch_p_signal = self.e_d_signal[ch].astype(floatdtype, copy=False)
                    np.subtract(ch_p_signal, self.baseline[ch], ch_p_signal)
                    np.divide(ch_p_signal, self.adc_gain[ch], ch_p_signal)
                    ch_p_signal[ch_nanlocs] = np.nan
                    p_signal.append(ch_p_signal)
            else:
                nanlocs = self.d_signal == dnans
                p_signal = self.d_signal.astype(floatdtype, copy=False)
                np.subtract(p_signal, self.baseline, p_signal)
                np.divide(p_signal, self.adc_gain, p_signal)
                p_signal[nanlocs] = np.nan

            return p_signal


    def calc_adc_params(self):
        """
        Compute appropriate gain and baseline parameters given the physical
        signal and the fmts.

        digital - baseline / gain = physical
        physical * gain + baseline = digital
        """
        gains = []
        baselines = []

        # min and max ignoring nans, unless whole channel is nan.
        # Should suppress warning message.
        minvals = np.nanmin(self.p_signal, axis=0)
        maxvals = np.nanmax(self.p_signal, axis=0)

        dnans = digi_nan(self.fmt)

        for ch in range(np.shape(self.p_signal)[1]):
            # Get the minimum and maximum (valid) storage values
            dmin, dmax = digi_bounds(self.fmt[ch])
            # add 1 because the lowest value is used to store nans
            dmin = dmin + 1
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
                if pmin == 0:
                    gain = 1
                    baseline = 1
                else:
                    gain = 1 / pmin
                    baseline = 0
            # Regular mixed signal case
            # Todo:
            else:
                gain = (dmax-dmin) / (pmax-pmin)
                baseline = dmin - gain*pmin

            # What about roundoff error? Make sure values don't map to beyond
            # range.
            baseline = int(baseline)

            # WFDB library limits...
            if abs(gain)>214748364 or abs(baseline)>2147483648:
                raise Exception('adc_gain and baseline must have magnitudes < 214748364')

            gains.append(gain)
            baselines.append(baseline)

        return (gains, baselines)


    def convert_dtype(self, physical, return_res, smooth_frames):
        if physical is True:
            returndtype = 'float'+str(return_res)
            if smooth_frames is True:
                currentdtype = self.p_signal.dtype
                if currentdtype != returndtype:
                    self.p_signal = self.p_signal.astype(returndtype, copy=False)
            else:
                for ch in range(self.n_sig):
                    if self.e_p_signal[ch].dtype != returndtype:
                        self.e_p_signal[ch] = self.e_p_signal[ch].astype(returndtype, copy=False)
        else:
            returndtype = 'int'+str(return_res)
            if smooth_frames is True:
                currentdtype = self.d_signal.dtype
                if currentdtype != returndtype:
                    # Do not allow changing integer dtype to lower value due to over/underflow
                    if int(str(currentdtype)[3:])>int(str(returndtype)[3:]):
                        raise Exception('Cannot convert digital samples to lower dtype. Risk of overflow/underflow.')
                    self.d_signal = self.d_signal.astype(returndtype, copy=False)
            else:
                for ch in range(self.n_sig):
                    currentdtype = self.e_d_signal[ch].dtype
                    if currentdtype != returndtype:
                        # Do not allow changing integer dtype to lower value due to over/underflow
                        if int(str(currentdtype)[3:])>int(str(returndtype)[3:]):
                            raise Exception('Cannot convert digital samples to lower dtype. Risk of overflow/underflow.')
                        self.e_d_signal[ch] = self.e_d_signal[ch].astype(returndtype, copy=False)
        return

    def calc_checksum(self, expanded=False):
        """
        Calculate the checksum(s) of the d_signal (expanded=False)
        or e_d_signal field (expanded=True)
        """
        if expanded:
            cs = [int(np.sum(self.e_d_signal[ch]) % 65536) for ch in range(self.n_sig)]
        else:
            cs = np.sum(self.d_signal, 0) % 65536
            cs = [int(c) for c in cs]
        return cs

    def wr_dat_files(self, expanded=False, write_dir=''):
        """
        Write each of the specified dat files

        """
        # Get the set of dat files to be written, and
        # the channels to be written to each file.
        file_names, dat_channels = describe_list_indices(self.file_name)

        # Get the fmt and byte offset corresponding to each dat file
        DAT_FMTS = {}
        dat_offsets = {}
        for fn in file_names:
            DAT_FMTS[fn] = self.fmt[dat_channels[fn][0]]

            # byte_offset may not be present
            if self.byte_offset is None:
                dat_offsets[fn] = 0
            else:
                dat_offsets[fn] = self.byte_offset[dat_channels[fn][0]]

        # Write the dat files
        if expanded:
            for fn in file_names:
                wr_dat_file(fn, DAT_FMTS[fn], None , dat_offsets[fn], True,
                            [self.e_d_signal[ch] for ch in dat_channels[fn]],
                            self.samps_per_frame, write_dir=write_dir)
        else:
            # Create a copy to prevent overwrite
            dsig = self.d_signal.copy()
            for fn in file_names:
                wr_dat_file(fn, DAT_FMTS[fn],
                            dsig[:, dat_channels[fn][0]:dat_channels[fn][-1]+1],
                            dat_offsets[fn], write_dir=write_dir)


    def smooth_frames(self, sigtype='physical'):
        """
        Convert expanded signals with different samples/frame into
        a uniform numpy array.

        Input parameters
        - sigtype (default='physical'): Specifies whether to mooth
          the e_p_signal field ('physical'), or the e_d_signal
          field ('digital').
        """
        spf = self.samps_per_frame[:]
        for ch in range(len(spf)):
            if spf[ch] is None:
                spf[ch] = 1

        # Total samples per frame
        tspf = sum(spf)

        if sigtype == 'physical':
            n_sig = len(self.e_p_signal)
            sig_len = int(len(self.e_p_signal[0])/spf[0])
            signal = np.zeros((sig_len, n_sig), dtype='float64')

            for ch in range(n_sig):
                if spf[ch] == 1:
                    signal[:, ch] = self.e_p_signal[ch]
                else:
                    for frame in range(spf[ch]):
                        signal[:, ch] += self.e_p_signal[ch][frame::spf[ch]]
                    signal[:, ch] = signal[:, ch] / spf[ch]

        elif sigtype == 'digital':
            n_sig = len(self.e_d_signal)
            sig_len = int(len(self.e_d_signal[0])/spf[0])
            signal = np.zeros((sig_len, n_sig), dtype='int64')

            for ch in range(n_sig):
                if spf[ch] == 1:
                    signal[:, ch] = self.e_d_signal[ch]
                else:
                    for frame in range(spf[ch]):
                        signal[:, ch] += self.e_d_signal[ch][frame::spf[ch]]
                    signal[:, ch] = signal[:, ch] / spf[ch]
        else:
            raise ValueError("sigtype must be 'physical' or 'digital'")

        return signal


#------------------- Reading Signals -------------------#

def rd_segment(file_name, dirname, pb_dir, n_sig, fmt, sig_len, byte_offset,
              samps_per_frame, skew, sampfrom, sampto, channels,
              smooth_frames, ignore_skew):
    """
    Read the samples from a single segment record's associated dat file(s)
    'channels', 'sampfrom', 'sampto', 'smooth_frames', and 'ignore_skew' are
    user desired input fields.
    All other input arguments are specifications of the segment
    """

    # Avoid changing outer variables
    byte_offset = byte_offset[:]
    samps_per_frame = samps_per_frame[:]
    skew = skew[:]

    # Set defaults for empty fields
    for i in range(n_sig):
        if byte_offset[i] == None:
            byte_offset[i] = 0
        if samps_per_frame[i] == None:
            samps_per_frame[i] = 1
        if skew[i] == None:
            skew[i] = 0

    # If skew is to be ignored, set all to 0
    if ignore_skew:
        skew = [0]*n_sig

    # Get the set of dat files, and the
    # channels that belong to each file.
    file_name, datchannel = describe_list_indices(file_name)

    # Some files will not be read depending on input channels.
    # Get the the wanted fields only.
    w_file_name = [] # one scalar per dat file
    w_fmt = {} # one scalar per dat file
    w_byte_offset = {} # one scalar per dat file
    w_samps_per_frame = {} # one list per dat file
    w_skew = {} # one list per dat file
    w_channel = {} # one list per dat file

    for fn in file_name:
        # intersecting dat channels between the input channels and the channels of the file
        idc = [c for c in datchannel[fn] if c in channels]

        # There is at least one wanted channel in the dat file
        if idc != []:
            w_file_name.append(fn)
            w_fmt[fn] = fmt[datchannel[fn][0]]
            w_byte_offset[fn] = byte_offset[datchannel[fn][0]]
            w_samps_per_frame[fn] = [samps_per_frame[c] for c in datchannel[fn]]
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
    if smooth_frames or sum(samps_per_frame) == n_sig:
        # Figure out the largest required dtype for the segment to minimize memory usage
        max_dtype = np_dtype(fmt_res(fmt, max_res=True), discrete=True)
        # Allocate signal array. Minimize dtype
        signals = np.zeros([sampto-sampfrom, len(channels)], dtype=max_dtype)

        # Read each wanted dat file and store signals
        for fn in w_file_name:
            signals[:, out_datchannel[fn]] = rddat(fn, dirname, pb_dir, w_fmt[fn], len(datchannel[fn]),
                sig_len, w_byte_offset[fn], w_samps_per_frame[fn], w_skew[fn], sampfrom, sampto, smooth_frames)[:, r_w_channel[fn]]

    # Return each sample in signals with multiple samples/frame, without smoothing.
    # Return a list of numpy arrays for each signal.
    else:
        signals=[None]*len(channels)

        for fn in w_file_name:
            # Get the list of all signals contained in the dat file
            datsignals = rddat(fn, dirname, pb_dir, w_fmt[fn], len(datchannel[fn]),
                sig_len, w_byte_offset[fn], w_samps_per_frame[fn], w_skew[fn], sampfrom, sampto, smooth_frames)

            # Copy over the wanted signals
            for cn in range(len(out_datchannel[fn])):
                signals[out_datchannel[fn][cn]] = datsignals[r_w_channel[fn][cn]]

    return signals


def rddat(file_name, dirname, pb_dir, fmt, n_sig,
          sig_len, byte_offset, samps_per_frame,
          skew, sampfrom, sampto, smooth_frames):
    """
    Get samples from a WFDB dat file.
    'sampfrom', 'sampto', and smooth_frames are user desired
    input fields. All other fields specify the file parameters.

    Returns all channels

    Input arguments:
    - file_name: The name of the dat file.
    - dirname: The full directory where the dat file is located, if the dat file is local.
    - pb_dir: The physiobank directory where the dat file is located, if the dat file is remote.
    - fmt: The format of the dat file
    - n_sig: The number of signals contained in the dat file
    - sig_len : The signal length (per channel) of the dat file
    - byte_offset: The byte offsets of the dat file
    - samps_per_frame: The samples/frame for the signals of the dat file
    - skew: The skew for the signals of the dat file
    - sampfrom: The starting sample number to be read from the signals
    - sampto: The final sample number to be read from the signals
    - smooth_frames: Whether to smooth channels with multiple samples/frame
    """

    # Total number of samples per frame
    tsamps_per_frame = sum(samps_per_frame)
    # The signal length to read (per channel)
    readlen = sampto - sampfrom

    # Calculate parameters used to read and process the dat file
    startbyte, nreadsamples, blockfloorsamples, extraflatsamples, nanreplace = calc_read_params(fmt, sig_len, byte_offset,
                                                                                                skew, tsamps_per_frame,
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
        if fmt in SPECIAL_FMTS:
            # Extra number of bytes to append onto the bytes read from the dat file.
            extrabytenum = totalprocessbytes - totalreadbytes

            sigbytes = np.concatenate((getdatbytes(file_name, dirname, pb_dir, fmt, startbyte, nreadsamples),
                                      np.zeros(extrabytenum, dtype = np.dtype(dataloadtypes[fmt]))))
        else:
            sigbytes = np.concatenate((getdatbytes(file_name, dirname, pb_dir, fmt, startbyte, nreadsamples),
                                      np.zeros(extraflatsamples, dtype = np.dtype(dataloadtypes[fmt]))))
    else:
        sigbytes = getdatbytes(file_name, dirname, pb_dir, fmt, startbyte, nreadsamples)

    # Continue to process the read values into proper samples

    # For special fmts, Turn the bytes into actual samples
    if fmt in SPECIAL_FMTS:
        sigbytes = bytes2samples(sigbytes, totalprocesssamples, fmt)
        # Remove extra leading sample read within the byte block if any
        if blockfloorsamples:
            sigbytes = sigbytes[blockfloorsamples:]
    # Adjust for byte offset formats
    elif fmt == '80':
        sigbytes = (sigbytes.astype('int16') - 128).astype('int8')
    elif fmt == '160':
        sigbytes = (sigbytes.astype('int32') - 32768).astype('int16')

    # At this point, dtype of sigbytes is the minimum integer format required for storing
    # final samples.

    # No extra samples/frame. Obtain original uniform numpy array
    if tsamps_per_frame==n_sig:
        # Reshape into multiple channels
        sig = sigbytes.reshape(-1, n_sig)
        # Skew the signal
        sig = skewsig(sig, skew, n_sig, readlen, fmt, nanreplace)

    # Extra frames present to be smoothed. Obtain averaged uniform numpy array
    elif smooth_frames:

        # Allocate memory for smoothed signal.
        sig = np.zeros((int(len(sigbytes)/tsamps_per_frame) , n_sig), dtype=sigbytes.dtype)

        # Transfer and average samples
        for ch in range(n_sig):
            if samps_per_frame[ch] == 1:
                sig[:, ch] = sigbytes[sum(([0] + samps_per_frame)[:ch + 1])::tsamps_per_frame]
            else:
                if ch == 0:
                    startind = 0
                else:
                    startind = np.sum(samps_per_frame[:ch])
                sig[:,ch] = [np.average(sigbytes[ind:ind+samps_per_frame[ch]]) for ind in range(startind,len(sigbytes),tsamps_per_frame)]
        # Skew the signal
        sig = skewsig(sig, skew, n_sig, readlen, fmt, nanreplace)

    # Extra frames present without wanting smoothing. Return all expanded samples.
    else:
        # List of 1d numpy arrays
        sig=[]
        # Transfer over samples
        for ch in range(n_sig):
            # Indices of the flat signal that belong to the channel
            ch_indices = np.concatenate([np.array(range(samps_per_frame[ch])) + sum([0]+samps_per_frame[:ch]) + tsamps_per_frame*framenum for framenum in range(int(len(sigbytes)/tsamps_per_frame))])
            sig.append(sigbytes[ch_indices])
        # Skew the signal
        sig = skewsig(sig, skew, n_sig, readlen, fmt, nanreplace, samps_per_frame)

    # Integrity check of signal shape after reading
    checksigdims(sig, readlen, n_sig, samps_per_frame)

    return sig

def calc_read_params(fmt, sig_len, byte_offset, skew, tsamps_per_frame, sampfrom, sampto):
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
    sig_len=100, t = 4 (total samples/frame), skew = [0, 2, 4, 5]
    sampfrom=0, sampto=100 --> readlen = 100, nsampread = 100*t, extralen = 5, nanreplace = [0, 2, 4, 5]
    sampfrom=50, sampto=100 --> readlen = 50, nsampread = 50*t, extralen = 5, nanreplace = [0, 2, 4, 5]
    sampfrom=0, sampto=50 --> readlen = 50, nsampread = 55*t, extralen = 0, nanreplace = [0, 0, 0, 0]
    sampfrom=95, sampto=99 --> readlen = 4, nsampread = 5*t, extralen = 4, nanreplace = [0, 1, 3, 4]
    """

    # First flat sample number to read (if all channels were flattened)
    startflatsample = sampfrom * tsamps_per_frame

    #endflatsample = min((sampto + max(skew)-sampfrom), sig_len) * tsamps_per_frame

    # Calculate the last flat sample number to read.
    # Cannot exceed sig_len * tsamps_per_frame, the number of samples stored in the file.
    # If extra 'samples' are desired by the skew, keep track.
    # Where was the -sampfrom derived from? Why was it in the formula?
    if (sampto + max(skew))>sig_len:
        endflatsample = sig_len*tsamps_per_frame
        extraflatsamples = (sampto + max(skew) - sig_len) * tsamps_per_frame
    else:
        endflatsample = (sampto + max(skew)) * tsamps_per_frame
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
    startbyte = byte_offset + int(startflatsample * bytespersample[fmt])

    # The number of samples to read
    nreadsamples = endflatsample - startflatsample

    # The number of samples to replace with nan at the end of each signal
    # due to skew wanting samples beyond the file

    # Calculate this using the above statement case: if (sampto + max(skew))>sig_len:
    nanreplace = [max(0, sampto + s - sig_len) for s in skew]

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


def getdatbytes(file_name, dirname, pb_dir, fmt, startbyte, nsamp):
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
    if pb_dir is None:
        fp = open(os.path.join(dirname, file_name), 'rb')
        fp.seek(startbyte)

        # Read file using corresponding dtype
        sigbytes = np.fromfile(fp, dtype=np.dtype(dataloadtypes[fmt]), count=elementcount)

        fp.close()

    # Stream dat file from physiobank
    # Same output as above np.fromfile.
    else:
        sigbytes = download.stream_dat(file_name, pb_dir, fmt, bytecount, startbyte, dataloadtypes)

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

        sigbytes = sigbytes.astype('int16')
        sig = np.zeros(nsamp, dtype='int16')

        # One sample pair is stored in one byte triplet.

        # Even numbered samples
        sig[0::2] = sigbytes[0::3] + 256 * np.bitwise_and(sigbytes[1::3], 0x0f)
        # Odd numbered samples (len(sig) always >1 due to processing of whole blocks)
        sig[1::2] = sigbytes[2::3] + 256*np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)

        # Remove trailing sample read within the byte block if originally odd sampled
        if addedsamps:
            sig = sig[:-addedsamps]

        # Loaded values as un_signed. Convert to 2's complement form:
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

        sigbytes = sigbytes.astype('int16')
        sig = np.zeros(nsamp, dtype='int16')

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

        # Loaded values as un_signed. Convert to 2's complement form:
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

        sigbytes = sigbytes.astype('int16')
        sig = np.zeros(nsamp, dtype='int16')

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

        # Loaded values as un_signed. Convert to 2's complement form:
        # values > 2^9-1 are negative.
        sig[sig > 511] -= 1024
    return sig


def skewsig(sig, skew, n_sig, readlen, fmt, nanreplace, samps_per_frame=None):
    """
    Skew the signal, insert nans and shave off end of array if needed.

    fmt is just for the correct nan value.
    samps_per_frame is only used for skewing expanded signals.
    """
    if max(skew)>0:

        # Expanded frame samples. List of arrays.
        if isinstance(sig, list):
            # Shift the channel samples
            for ch in range(n_sig):
                if skew[ch]>0:
                    sig[ch][:readlen*samps_per_frame[ch]] = sig[ch][skew[ch]*samps_per_frame[ch]:]

            # Shave off the extra signal length at the end
            for ch in range(n_sig):
                sig[ch] = sig[ch][:readlen*samps_per_frame[ch]]

            # Insert nans where skewed signal overran dat file
            for ch in range(n_sig):
                if nanreplace[ch]>0:
                    sig[ch][-nanreplace[ch]:] = digi_nan(fmt)
        # Uniform array
        else:
            # Shift the channel samples
            for ch in range(n_sig):
                if skew[ch]>0:
                    sig[:readlen, ch] = sig[skew[ch]:, ch]
            # Shave off the extra signal length at the end
            sig = sig[:readlen, :]

            # Insert nans where skewed signal overran dat file
            for ch in range(n_sig):
                if nanreplace[ch]>0:
                    sig[-nanreplace[ch]:, ch] = digi_nan(fmt)

    return sig


# Integrity check of signal shape after reading
def checksigdims(sig, readlen, n_sig, samps_per_frame):
    if isinstance(sig, np.ndarray):
        if sig.shape != (readlen, n_sig):
            raise ValueError('Samples were not loaded correctly')
    else:
        if len(sig) != n_sig:
            raise ValueError('Samples were not loaded correctly')
        for ch in range(n_sig):
            if len(sig[ch]) != samps_per_frame[ch] * readlen:
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
    if isinstance(fmt, list):
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
    if isinstance(fmt, list):
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


def est_res(signals):
    """
    def est_res(signals):

    Estimate the resolution of each signal in a multi-channel signal in
    bits. Maximum of 32 bits.

    Parameters
    ----------
    signals : numpy array, or list
        A 2d numpy array representing a uniform multichannel signal, or
        a list of 1d numpy arrays representing multiple channels of
        signals with different numbers of samples per frame.

    Returns
    -------
    res : list
        A list of estimated integer resolutions for each channel
    """
    res_levels = np.power(2, np.arange(0, 33))
    # Expanded sample signals. List of numpy arrays
    if isinstance(signals, list):
        n_sig = len(signals)
    # Uniform numpy array
    else:
        if signals.ndim ==1:
            n_sig = 1
        else:
            n_sig = signals.shape[1]
    res = []

    for ch in range(n_sig):
        # Estimate the number of steps as the range divided by the
        # minimum increment.
        if isinstance(signals, list):
            sortedsig = np.sort(signals[ch])
        else:
            sortedsig = np.sort(signals[:,ch])

        min_inc = min(np.diff(sortedsig))

        if min_inc == 0:
            # Case where signal is flat. Resolution is 0.
            res.append(0)
        else:
            nlevels = 1 + (sortedsig[-1]-sortedsig[0])/min_inc
            if nlevels>=res_levels[-1]:
                res.append(32)
            else:
                res.append(np.where(res_levels>=nlevels)[0][0])

    return res



def wfdbfmt(res, single_fmt=True):
    """
    Return the most suitable wfdb format(s) to use given signal
    resolutions.
    If single_fmt is True, the format for the maximum resolution will be returned.

    Parameters

    """
    if isinstance(res, list):
        # Return a single format
        if single_fmt:
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


def fmt_res(fmt, max_res=False):
    """
    Return the resolution of the WFDB format(s).

    Parameters
    ----------
    fmt : str
        The wfdb format. Can be a list of valid fmts. If it is a list,
        and `max_res` is True, the list may contain None.
    max_res : bool, optional
        If given a list of fmts, whether to return the highest
        resolution.

    """
    if isinstance(fmt, list):
        if max_res:
            # Allow None
            res = np.max([fmt_res(f) for f in fmt if f is not None])
        else:
            res = [fmt_res(f) for f in fmt]
        return res

    res = {'8':8, '80':8, '310':10, '311':10, '212':12, '16':16, '61':16,
           '24':24, '32':32}

    if fmt in res:
        return res[fmt]
    else:
        raise ValueError('Invalid WFDB format.')



def np_dtype(res, discrete):
    """
    Given the resolution of a signal, return the minimum
    dtype to store it

    Parameters
    ----------
    res : int
        The resolution.
    discrete : bool
        Whether the dtype is to be discrete or floating.
    """
    if not hasattr(res, '__index__') or res > 64:
        raise TypeError('res must be integer based and <= 64')

    for np_res in [8, 16, 32, 64]:
        if res <= np_res:
            break

    if discrete is True:
        return 'int' + str(np_res)
    else:
        # No float8 dtype
        if np_res == 8:
            np_res = 16
        return 'float' + str(np_res)


def wr_dat_file(file_name, fmt, d_signal, byte_offset, expanded=False,
                e_d_signal=None, samps_per_frame=None, write_dir=''):
    """
    Write a dat file. All bytes are written one at a time to avoid
    endianness issues.

    """
    # Combine list of arrays into single array
    if expanded:
        n_sig = len(e_d_signal)
        sig_len = int(len(e_d_signal[0])/samps_per_frame[0])
        # Effectively create MxN signal, with extra frame samples acting like extra channels
        d_signal = np.zeros((sig_len, sum(samps_per_frame)), dtype = 'int64')
        # Counter for channel number
        expand_ch = 0
        for ch in range(n_sig):
            spf = samps_per_frame[ch]
            for framenum in range(spf):
                d_signal[:, expand_ch] = e_d_signal[ch][framenum::spf]
                expand_ch = expand_ch + 1

    # This n_sig is used for making list items.
    # Does not necessarily represent number of signals (ie. for expanded=True)
    n_sig = d_signal.shape[1]

    if fmt == '80':
        # convert to 8 bit offset binary form
        d_signal = d_signal + 128
        # Concatenate into 1D
        d_signal = d_signal.reshape(-1)
        # Convert to un_signed 8 bit dtype to write
        bwrite = d_signal.astype('uint8')

    elif fmt == '212':
        # Each sample is represented by a 12 bit two's complement amplitude.
        # The first sample is obtained from the 12 least significant bits of the first byte pair (stored least significant byte first).
        # The second sample is formed from the 4 remaining bits of the first byte pair (which are the 4 high bits of the 12-bit sample)
        # and the next byte (which contains the remaining 8 bits of the second sample).
        # The process is repeated for each successive pair of samples.

        # convert to 12 bit two's complement
        d_signal[d_signal<0] = d_signal[d_signal<0] + 4096

        # Concatenate into 1D
        d_signal = d_signal.reshape(-1)

        nsamp = len(d_signal)
        # use this for byte processing
        processnsamp = nsamp

        # Odd numbered number of samples. Fill in extra blank for following byte calculation.
        if processnsamp % 2:
            d_signal = np.concatenate([d_signal, np.array([0])])
            processnsamp +=1

        # The individual bytes to write
        bwrite = np.zeros([int(1.5*processnsamp)], dtype = 'uint8')

        # Fill in the byte triplets

        # Triplet 1 from lowest 8 bits of sample 1
        bwrite[0::3] = d_signal[0::2] & 255
        # Triplet 2 from highest 4 bits of samples 1 (lower) and 2 (upper)
        bwrite[1::3] = ((d_signal[0::2] & 3840) >> 8) + ((d_signal[1::2] & 3840) >> 4)
        # Triplet 3 from lowest 8 bits of sample 2
        bwrite[2::3] = d_signal[1::2] & 255

        # If we added an extra sample for byte calculation, remove the last byte (don't write)
        if nsamp % 2:
            bwrite = bwrite[:-1]

    elif fmt == '16':
        # convert to 16 bit two's complement
        d_signal[d_signal<0] = d_signal[d_signal<0] + 65536
        # Split samples into separate bytes using binary masks
        b1 = d_signal & [255]*n_sig
        b2 = ( d_signal & [65280]*n_sig ) >> 8
        # Interweave the bytes so that the same samples' bytes are consecutive
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        bwrite = np.concatenate((b1, b2), axis=1)
        bwrite = bwrite.reshape((1,-1))[0]
        # Convert to un_signed 8 bit dtype to write
        bwrite = bwrite.astype('uint8')
    elif fmt == '24':
        # convert to 24 bit two's complement
        d_signal[d_signal<0] = d_signal[d_signal<0] + 16777216
        # Split samples into separate bytes using binary masks
        b1 = d_signal & [255]*n_sig
        b2 = ( d_signal & [65280]*n_sig ) >> 8
        b3 = ( d_signal & [16711680]*n_sig ) >> 16
        # Interweave the bytes so that the same samples' bytes are consecutive
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        b3 = b3.reshape((-1, 1))
        bwrite = np.concatenate((b1, b2, b3), axis=1)
        bwrite = bwrite.reshape((1,-1))[0]
        # Convert to un_signed 8 bit dtype to write
        bwrite = bwrite.astype('uint8')

    elif fmt == '32':
        # convert to 32 bit two's complement
        d_signal[d_signal<0] = d_signal[d_signal<0] + 4294967296
        # Split samples into separate bytes using binary masks
        b1 = d_signal & [255]*n_sig
        b2 = ( d_signal & [65280]*n_sig ) >> 8
        b3 = ( d_signal & [16711680]*n_sig ) >> 16
        b4 = ( d_signal & [4278190080]*n_sig ) >> 24
        # Interweave the bytes so that the same samples' bytes are consecutive
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        b3 = b3.reshape((-1, 1))
        b4 = b4.reshape((-1, 1))
        bwrite = np.concatenate((b1, b2, b3, b4), axis=1)
        bwrite = bwrite.reshape((1,-1))[0]
        # Convert to un_signed 8 bit dtype to write
        bwrite = bwrite.astype('uint8')
    else:
        raise ValueError('This library currently only supports writing the following formats: 80, 16, 24, 32')

    # Byte offset in the file
    if byte_offset is not None and byte_offset>0:
        print('Writing file '+file_name+' with '+str(byte_offset)+' empty leading bytes')
        bwrite = np.append(np.zeros(byte_offset, dtype = 'uint8'), bwrite)

    f=open(os.path.join(write_dir, file_name),'wb')

    # Write the file
    bwrite.tofile(f)

    f.close()


def describe_list_indices(full_list):
    """
    Parameters
    ----------
    full_list : list
        The list of items to order and

    Returns
    -------
    unique_elements : list
        A list of the unique elements of the list, in the order in which
        they first appear.
    element_indices : dict
        A dictionary of lists for each unique element, giving all the
        indices in which they appear in the original list.

    """
    unique_elements = []
    element_indices = {}

    for i in range(len(full_list)):
        item = full_list[i]
        # new item
        if item not in unique_elements:
            unique_elements.append(item)
            element_indices[item] = [i]
        # previously seen item
        else:
            element_indices[item].append(i)
    return unique_elements, element_indices


def downround(x, base):
    """
    Round <x> down to nearest <base>
    """
    return base * math.floor(float(x)/base)


def upround(x, base):
    """
    Round <x> up to nearest <base>
    """
    return base * math.ceil(float(x)/base)
