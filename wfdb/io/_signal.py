import math
import os

import numpy as np

from wfdb.io import download
import pdb


MAX_I32 = 2147483647
MIN_I32 = -2147483648

# Formats in which all samples align with byte boundaries
ALIGNED_FMTS = ['8', '16', '24', '32', '61', '80', '160']
# Formats in which not all samples align with byte boundaries
UNALIGNED_FMTS = ['212', '310', '311']
# Formats which are stored in offset binary form
OFFSET_FMTS = ['80', '160']
# All WFDB dat formats - https://www.physionet.org/physiotools/wag/signal-5.htm
DAT_FMTS = ALIGNED_FMTS + UNALIGNED_FMTS

# Bytes required to hold each sample (including wasted space) for each
# WFDB dat formats
BYTES_PER_SAMPLE = {'8': 1, '16': 2, '24': 3, '32': 4, '61': 2, '80': 1,
                    '160': 2, '212': 1.5, '310': 4 / 3., '311': 4 / 3.}

# The bit resolution of each WFDB dat format
BIT_RES = {'8': 8, '16': 16, '24': 24, '32': 32, '61': 16, '80': 8,
           '160': 16, '212': 12, '310': 10, '311': 10}

# Numpy dtypes used to load dat files of each format.
DATA_LOAD_TYPES = {'8': '<i1', '16': '<i2', '24': '<i3', '32': '<i4',
                   '61': '>i2', '80': '<u1', '160': '<u2', '212': '<u1',
                   '310': '<u1', '311': '<u1'}


class SignalMixin(object):
    """
    Mixin class with signal methods. Inherited by Record class.

    Attributes
    ----------
    N/A


    """
    def wr_dats(self, expanded, write_dir):
        """
        Write all dat files associated with a record
        expanded=True to use e_d_signal instead of d_signal.

        Parameters
        ----------
        expanded : bool
            Whether to transform the `e_d_signal` attribute (True) or
            the `d_signal` attribute (False).
        write_dir : str
            The directory to write the output file to.

        Returns
        -------
        N/A

        """
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
        fields used to write the record.

        Parameters
        ----------
        write_fields : list
            All the fields used to write the header.
        expanded : bool
            Whether to transform the `e_d_signal` attribute (True) or
            the `d_signal` attribute (False).

        Returns
        -------
        N/A

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
                dmin, dmax = _digi_bounds(self.fmt[ch])

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
                dmin, dmax = _digi_bounds(self.fmt[ch])

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
        do_dac : bool, optional
            Whether to use the digital signal field to perform dac
            conversion to get the physical signal field beforehand.
        expanded : bool, optional
            Whether to transform the `e_p_signal` attribute (True) or
            the `p_signal` attribute (False). If True, the `samps_per_frame` 
            attribute is also required.

        Returns
        -------
        N/A

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
        do_adc : bools, optional
            Whether to use the physical signal field to perform adc
            conversion to get the digital signal field beforehand.
        single_fmt : bool, optional
            Whether to use a single digital format during adc, if it is
            performed.
        expanded : bool, optional
            Whether to transform the `e_p_signal` attribute (True) or
            the `p_signal` attribute (False).

        Returns
        -------
        N/A

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
                    self.fmt = _wfdb_fmt(res, single_fmt)
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
                    self.fmt = _wfdb_fmt(res, single_fmt)
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
        d_signal : ndarray, optional
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
        # The digital NAN values for each channel
        d_nans = _digi_nan(self.fmt)

        # To do: choose the minimum return res needed
        intdtype = 'int64'

        # Do inplace conversion and set relevant variables.
        if inplace:
            if expanded:
                for ch in range(self.n_sig):
                    # NAN locations for the channel
                    ch_nanlocs = np.isnan(self.e_p_signal[ch])
                    np.multiply(self.e_p_signal[ch], self.adc_gain[ch],
                                self.e_p_signal[ch])
                    np.add(e_p_signal[ch], self.baseline[ch],
                           self.e_p_signal[ch])
                    self.e_p_signal[ch] = self.e_p_signal[ch].astype(intdtype,
                                                                     copy=False)
                    self.e_p_signal[ch][ch_nanlocs] = d_nans[ch]
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
                    # NAN locations for the channel
                    ch_nanlocs = np.isnan(self.e_p_signal[ch])
                    ch_d_signal = self.e_p_signal[ch].copy()
                    np.multiply(ch_d_signal, self.adc_gain[ch], ch_d_signal)
                    np.add(ch_d_signal, self.baseline[ch], ch_d_signal)
                    ch_d_signal = ch_d_signal.astype(intdtype, copy=False)
                    ch_d_signal[ch_nanlocs] = d_nans[ch]
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
                            d_signal[nanlocs[:,ch],ch] = d_nans[ch]

            return d_signal


    def dac(self, expanded=False, return_res=64, inplace=False):
        """
        Performs the digital to analogue conversion of the signal stored
        in `d_signal` if expanded is False, or `e_d_signal` if expanded
        is True.

        The d_signal/e_d_signal, fmt, gain, and baseline fields must all be
        valid.

        If inplace is True, the dac will be performed inplace on the
        variable, the p_signal/e_p_signal attribute will be set, and the
        d_signal/e_d_signal field will be set to None.

        Parameters
        ----------
        expanded : bool, optional
            Whether to transform the `e_d_signal attribute` (True) or
            the `d_signal` attribute (False).
        return_res : int, optional
            The numpy array dtype of the returned signals. Options are: 64,
            32, 16, and 8, where the value represents the numpy int or float
            dtype. Note that the value cannot be 8 when physical is True
            since there is no float8 format.       
        inplace : bool, optional
            Whether to automatically set the object's corresponding
            physical signal attribute and set the digital signal
            attribute to None (True), or to return the converted
            signal as a separate variable without changing the original
            digital signal attribute (False).

        Returns
        -------
        p_signal : ndarray, optional
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
        # The digital NAN values for each channel
        d_nans = _digi_nan(self.fmt)

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
                    # NAN locations for the channel
                    ch_nanlocs = self.e_d_signal[ch] == d_nans[ch]
                    self.e_d_signal[ch] = self.e_d_signal[ch].astype(floatdtype, copy=False)
                    np.subtract(self.e_d_signal[ch], self.baseline[ch], self.e_d_signal[ch])
                    np.divide(self.e_d_signal[ch], self.adc_gain[ch], self.e_d_signal[ch])
                    self.e_d_signal[ch][ch_nanlocs] = np.nan
                self.e_p_signal = self.e_d_signal
                self.e_d_signal = None
            else:
                nanlocs = self.d_signal == d_nans
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
                    # NAN locations for the channel
                    ch_nanlocs = self.e_d_signal[ch] == d_nans[ch]
                    ch_p_signal = self.e_d_signal[ch].astype(floatdtype, copy=False)
                    np.subtract(ch_p_signal, self.baseline[ch], ch_p_signal)
                    np.divide(ch_p_signal, self.adc_gain[ch], ch_p_signal)
                    ch_p_signal[ch_nanlocs] = np.nan
                    p_signal.append(ch_p_signal)
            else:
                nanlocs = self.d_signal == d_nans
                p_signal = self.d_signal.astype(floatdtype, copy=False)
                np.subtract(p_signal, self.baseline, p_signal)
                np.divide(p_signal, self.adc_gain, p_signal)
                p_signal[nanlocs] = np.nan

            return p_signal


    def calc_adc_params(self):
        """
        Compute appropriate adc_gain and baseline parameters for adc
        conversion, given the physical signal and the fmts.

        Parameters
        ----------
        N/A

        Returns
        -------
        adc_gains : list
            List of calculated `adc_gain` values for each channel.
        baselines : list
            List of calculated `baseline` values for each channel.

        Notes
        -----
        This is the mapping equation:
            `digital - baseline / adc_gain = physical`
            `physical * adc_gain + baseline = digital`

        The original WFDB library stores `baseline` as int32.
        Constrain abs(adc_gain) <= 2**31 == 2147483648.

        This function does carefully deal with overflow for calculated
        int32 `baseline` values, but does not consider over/underflow
        for calculated float `adc_gain` values.

        """
        adc_gains = []
        baselines = []

        if np.where(np.isinf(self.p_signal))[0].size:
            raise ValueError('Signal contains inf. Cannot perform adc.')

        # min and max ignoring nans, unless whole channel is NAN.
        # Should suppress warning message.
        minvals = np.nanmin(self.p_signal, axis=0)
        maxvals = np.nanmax(self.p_signal, axis=0)

        for ch in range(np.shape(self.p_signal)[1]):
            # Get the minimum and maximum (valid) storage values
            dmin, dmax = _digi_bounds(self.fmt[ch])
            # add 1 because the lowest value is used to store nans
            dmin = dmin + 1

            pmin = minvals[ch]
            pmax = maxvals[ch]

            # Figure out digital samples used to store physical samples

            # If the entire signal is NAN, gain/baseline won't be used
            if pmin == np.nan:
                adc_gain = 1
                baseline = 1
            # If the signal is just one value, store one digital value.
            elif pmin == pmax:
                if pmin == 0:
                    adc_gain = 1
                    baseline = 1
                else:
                    # All digital values are +1 or -1. Keep adc_gain > 0
                    adc_gain = abs(1 / pmin)
                    baseline = 0
            # Regular varied signal case.
            else:
                # The equation is: p = (d - b) / g

                # Approximately, pmax maps to dmax, and pmin maps to
                # dmin. Gradient will be equal to, or close to
                # delta(d) / delta(p), since intercept baseline has
                # to be an integer.

                # Constraint: baseline must be between +/- 2**31
                adc_gain = (dmax-dmin) / (pmax-pmin)
                baseline = dmin - adc_gain*pmin

                # Make adjustments for baseline to be an integer
                # This up/down round logic of baseline is to ensure
                # there is no overshoot of dmax. Now pmax will map
                # to dmax or dmax-1 which is also fine.
                if pmin > 0:
                    baseline = int(np.ceil(baseline))
                else:
                    baseline = int(np.floor(baseline))

                # After baseline is set, adjust gain correspondingly.Set
                # the gain to map pmin to dmin, and p==0 to baseline.
                # In the case where pmin == 0 and dmin == baseline,
                # adc_gain is already correct. Avoid dividing by 0.
                if dmin != baseline:
                    adc_gain = (dmin - baseline) / pmin

            # Remap signal if baseline exceeds boundaries.
            # This may happen if pmax < 0
            if baseline > MAX_I32:
                # pmin maps to dmin, baseline maps to 2**31 - 1
                # pmax will map to a lower value than before
                adc_gain = (MAX_I32) - dmin / abs(pmin)
                baseline = MAX_I32
            # This may happen if pmin > 0
            elif baseline < MIN_I32:
                # pmax maps to dmax, baseline maps to -2**31 + 1
                adc_gain = (dmax - MIN_I32) / pmax
                baseline = MIN_I32

            adc_gains.append(adc_gain)
            baselines.append(baseline)

        return (adc_gains, baselines)


    def convert_dtype(self, physical, return_res, smooth_frames):
        """
        Convert the dtype of the signal.

        Parameters
        ----------
        physical : bool
            Specifies whether to return dtype in physical (float) units in the
            `p_signal` field (True), or digital (int) units in the `d_signal`
            field (False).
        return_res : int
            The numpy array dtype of the returned signals. Options are: 64,
            32, 16, and 8, where the value represents the numpy int or float
            dtype. Note that the value cannot be 8 when physical is True
            since there is no float8 format.
        smooth_frames : bool
            Used when reading records with signals having multiple samples
            per frame. Specifies whether to smooth the samples in signals
            with more than one sample per frame and return an (MxN) uniform
            numpy array as the `d_signal` or `p_signal` field (True), or to
            return a list of 1d numpy arrays containing every expanded
            sample as the `e_d_signal` or `e_p_signal` field (False).

        Returns
        -------
        N/A

        """
        if physical is True:
            return_dtype = 'float'+str(return_res)
            if smooth_frames is True:
                current_dtype = self.p_signal.dtype
                if current_dtype != return_dtype:
                    self.p_signal = self.p_signal.astype(return_dtype, copy=False)
            else:
                if self.e_p_signal is not None:
                    for ch in range(self.n_sig):
                        if self.e_p_signal[ch].dtype != return_dtype:
                            self.e_p_signal[ch] = self.e_p_signal[ch].astype(return_dtype, copy=False)
                else:
                    for ch in range(self.n_sig):
                        if self.p_signal[ch].dtype != return_dtype:
                            self.p_signal[ch] = self.p_signal[ch].astype(return_dtype, copy=False)
        else:
            return_dtype = 'int'+str(return_res)
            if smooth_frames is True:
                current_dtype = self.d_signal.dtype
                if current_dtype != return_dtype:
                    # Do not allow changing integer dtype to lower value due to over/underflow
                    if int(str(current_dtype)[3:])>int(str(return_dtype)[3:]):
                        raise Exception('Cannot convert digital samples to lower dtype. Risk of overflow/underflow.')
                    self.d_signal = self.d_signal.astype(return_dtype, copy=False)
            else:
                for ch in range(self.n_sig):
                    current_dtype = self.e_d_signal[ch].dtype
                    if current_dtype != return_dtype:
                        # Do not allow changing integer dtype to lower value due to over/underflow
                        if int(str(current_dtype)[3:])>int(str(return_dtype)[3:]):
                            raise Exception('Cannot convert digital samples to lower dtype. Risk of overflow/underflow.')
                        self.e_d_signal[ch] = self.e_d_signal[ch].astype(return_dtype, copy=False)
        return


    def calc_checksum(self, expanded=False):
        """
        Calculate the checksum(s) of the input signal.

        Parameters
        ----------
        expanded : bool, optional
            Whether to transform the `e_d_signal` attribute (True) or
            the `d_signal` attribute (False).

        Returns
        -------
        cs : list
            The resulting checksum-ed signal.

        """
        if expanded:
            cs = [int(np.sum(self.e_d_signal[ch]) % 65536) for ch in range(self.n_sig)]
        else:
            cs = np.sum(self.d_signal, 0) % 65536
            cs = [int(c) for c in cs]
        return cs


    def wr_dat_files(self, expanded=False, write_dir=''):
        """
        Write each of the specified dat files.

        Parameters
        ----------
        expanded : bool, optional
            Whether to transform the `e_d_signal` attribute (True) or
            the `d_signal` attribute (False).
        write_dir : str, optional
            The directory to write the output file to.

        Returns
        -------
        N/A

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

        Parameters
        ----------
        sigtype (default='physical') : str 
            Specifies whether to mooth the e_p_signal field ('physical'), or the e_d_signal
            field ('digital').

        Returns
        -------
        signal : ndarray
            Tranformed expanded signal into uniform signal.

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


def _rd_segment(file_name, dir_name, pn_dir, fmt, n_sig, sig_len, byte_offset,
                samps_per_frame, skew, sampfrom, sampto, channels,
                smooth_frames, ignore_skew, no_file=False, sig_data=None, return_res=64):
    """
    Read the digital samples from a single segment record's associated
    dat file(s).

    Parameters
    ----------
    file_name : list
        The names of the dat files to be read.
    dir_name : str
        The full directory where the dat file(s) are located, if the dat
        file(s) are local.
    pn_dir : str
        The PhysioNet directory where the dat file(s) are located, if
        the dat file(s) are remote.
    fmt : list
        The formats of the dat files.
    n_sig : int
        The number of signals contained in the dat file.
    sig_len : int
        The signal length (per channel) of the dat file.
    byte_offset : int
        The byte offset of the dat file.
    samps_per_frame : list
        The samples/frame for each signal of the dat file.
    skew : list
        The skew for the signals of the dat file.
    sampfrom : int
        The starting sample number to be read from the signals.
    sampto : int
        The final sample number to be read from the signals.
    smooth_frames : bool
        Whether to smooth channels with multiple samples/frame.
    ignore_skew : bool
        Used when reading records with at least one skewed signal.
        Specifies whether to apply the skew to align the signals in the
        output variable (False), or to ignore the skew field and load in
        all values contained in the dat files unaligned (True).
    no_file : bool, optional
        Used when using this function with just an array of signal data
        and no associated file to read the data from.
    sig_data : ndarray, optional
        The signal data that would normally be imported using the associated
        .dat and .hea files. Should only be used when no_file is set to True.
    return_res : int, optional
        The numpy array dtype of the returned signals. Options are: 64,
        32, 16, and 8, where the value represents the numpy int or float
        dtype. Note that the value cannot be 8 when physical is True
        since there is no float8 format.

    Returns
    -------
    signals : ndarray, list
        The signals read from the dat file(s). A 2d numpy array is
        returned if the signals have uniform samples/frame or if
        `smooth_frames` is True. Otherwise a list of 1d numpy arrays
        is returned.

    Notes
    -----
    'channels', 'sampfrom', 'sampto', 'smooth_frames', and 'ignore_skew'
    are user desired input fields. All other parameters are
    specifications of the segment.

    """
    # Check for valid inputs
    if no_file and sig_data is None:
        raise Exception('signal_dat empty: No signal data provided')

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
    out_dat_channel = {}
    for fn in w_channel:
        r_w_channel[fn] = [c - min(datchannel[fn]) for c in w_channel[fn]]
        out_dat_channel[fn] = [channels.index(c) for c in w_channel[fn]]

    # Signals with multiple samples/frame are smoothed, or all signals have 1 sample/frame.
    # Return uniform numpy array
    if smooth_frames or sum(samps_per_frame) == n_sig:
        # Figure out the largest required dtype for the segment to minimize memory usage
        max_dtype = _np_dtype(_fmt_res(fmt, max_res=True), discrete=True)
        # Allocate signal array. Minimize dtype
        signals = np.zeros([sampto-sampfrom, len(channels)], dtype=max_dtype)

        # Read each wanted dat file and store signals
        for fn in w_file_name:
            if no_file:
                signals[:, out_dat_channel[fn]] = _rd_dat_signals(fn, dir_name,
                    pn_dir, w_fmt[fn], len(datchannel[fn]), sig_len,
                    w_byte_offset[fn], w_samps_per_frame[fn], w_skew[fn],
                    sampfrom, sampto, smooth_frames, no_file=True,
                    sig_data=sig_data)[:, r_w_channel[fn]]
            else:
                signals[:, out_dat_channel[fn]] = _rd_dat_signals(fn, dir_name,
                    pn_dir, w_fmt[fn], len(datchannel[fn]), sig_len,
                    w_byte_offset[fn], w_samps_per_frame[fn], w_skew[fn],
                    sampfrom, sampto, smooth_frames)[:, r_w_channel[fn]]

    # Return each sample in signals with multiple samples/frame, without smoothing.
    # Return a list of numpy arrays for each signal.
    else:
        signals = [None] * len(channels)

        for fn in w_file_name:
            # Get the list of all signals contained in the dat file
            if no_file:
                datsignals = _rd_dat_signals(fn, dir_name, pn_dir, w_fmt[fn],
                    len(datchannel[fn]), sig_len, w_byte_offset[fn],
                    w_samps_per_frame[fn], w_skew[fn], sampfrom, sampto,
                    smooth_frames, no_file=True, sig_data=sig_data)
            else:
                datsignals = _rd_dat_signals(fn, dir_name, pn_dir, w_fmt[fn],
                    len(datchannel[fn]), sig_len, w_byte_offset[fn],
                    w_samps_per_frame[fn], w_skew[fn], sampfrom, sampto,
                    smooth_frames)

            # Copy over the wanted signals
            for cn in range(len(out_dat_channel[fn])):
                signals[out_dat_channel[fn][cn]] = datsignals[r_w_channel[fn][cn]]

    return signals


def _rd_dat_signals(file_name, dir_name, pn_dir, fmt, n_sig, sig_len,
                   byte_offset, samps_per_frame, skew, sampfrom, sampto,
                   smooth_frames, no_file=False, sig_data=None):
    """
    Read all signals from a WFDB dat file.

    Parameters
    ----------
    file_name : str
        The name of the dat file.
    dir_name : str
        The full directory where the dat file(s) are located, if the dat
        file(s) are local.
    pn_dir : str
        The PhysioNet directory where the dat file(s) are located, if
        the dat file(s) are remote.
    fmt : list
        The formats of the dat files.
    n_sig : int
        The number of signals contained in the dat file.
    sig_len : int
        The signal length (per channel) of the dat file.
    byte_offset : int
        The byte offset of the dat file.
    samps_per_frame : list
        The samples/frame for each signal of the dat file.
    skew : list
        The skew for the signals of the dat file.
    sampfrom : int
        The starting sample number to be read from the signals.
    sampto : int
        The final sample number to be read from the signals.
    smooth_frames : bool
        Whether to smooth channels with multiple samples/frame.
    no_file : bool, optional
        Used when using this function with just an array of signal data
        and no associated file to read the data from.
    sig_data : ndarray, optional
        The signal data that would normally be imported using the associated
        .dat and .hea files. Should only be used when no_file is set to True.

    Returns
    -------
    signal : ndarray, list
        The signals read from the dat file(s). A 2d numpy array is
        returned if the signals have uniform samples/frame or if
        `smooth_frames` is True. Otherwise a list of 1d numpy arrays
        is returned.

    Notes
    -----
    'channels', 'sampfrom', 'sampto', 'smooth_frames', and 'ignore_skew'
    are user desired input fields. All other parameters are
    specifications of the segment.

    """
    # Check for valid inputs
    if no_file and sig_data is None:
        raise Exception('signal_dat empty: No signal data provided')

    # Total number of samples per frame
    tsamps_per_frame = sum(samps_per_frame)
    # The signal length to read (per channel)
    read_len = sampto - sampfrom

    # Calculate parameters used to read and process the dat file
    (start_byte, n_read_samples, block_floor_samples,
     extra_flat_samples, nan_replace) = _dat_read_params(fmt, sig_len,
                                                         byte_offset, skew,
                                                         tsamps_per_frame,
                                                         sampfrom, sampto)

    # Number of bytes to be read from the dat file
    total_read_bytes = _required_byte_num('read', fmt, n_read_samples)

    # Total samples to be processed in intermediate step. Includes extra
    # padded samples beyond dat file
    total_process_samples = n_read_samples + extra_flat_samples

    # Total number of bytes to be processed in intermediate step.
    total_process_bytes = _required_byte_num('read', fmt,
                                             total_process_samples)

    # Get the intermediate bytes or samples to process. Bit of a
    # discrepancy. Recall special formats load uint8 bytes, other formats
    # already load samples.

    # Read values from dat file. Append bytes/samples if needed.
    if no_file:
        data_to_read = sig_data
    else:
        data_to_read = _rd_dat_file(file_name, dir_name, pn_dir, fmt,
                                    start_byte, n_read_samples)

    if extra_flat_samples:
        if fmt in UNALIGNED_FMTS:
            # Extra number of bytes to append onto the bytes read from
            # the dat file.
            n_extra_bytes = total_process_bytes - total_read_bytes

            sig_data = np.concatenate((data_to_read,
                                        np.zeros(n_extra_bytes,
                                                dtype=np.dtype(DATA_LOAD_TYPES[fmt]))))
        else:
            sig_data = np.concatenate((data_to_read,
                                        np.zeros(extra_flat_samples,
                                                dtype=np.dtype(DATA_LOAD_TYPES[fmt]))))
    else:
        sig_data = data_to_read

    # Finish processing the read data into proper samples if not already

    # For unaligned fmts, turn the uint8 blocks into actual samples
    if fmt in UNALIGNED_FMTS:
        sig_data = _blocks_to_samples(sig_data, total_process_samples, fmt)
        # Remove extra leading sample read within the byte block if any
        if block_floor_samples:
            sig_data = sig_data[block_floor_samples:]

    # Adjust samples values for byte offset formats
    if fmt in OFFSET_FMTS:
        if fmt == '80':
            sig_data = (sig_data.astype('int16') - 128).astype('int8')
        elif fmt == '160':
            sig_data = (sig_data.astype('int32') - 32768).astype('int16')

    # At this point, dtype of sig_data is the minimum integer format
    # required for storing the final digital samples.

    # No extra samples/frame. Obtain original uniform numpy array
    if tsamps_per_frame == n_sig:
        # Reshape into multiple channels
        signal = sig_data.reshape(-1, n_sig)
        # Skew the signal
        signal = _skew_sig(signal, skew, n_sig, read_len, fmt, nan_replace)
    # Extra frames present to be smoothed. Obtain averaged uniform numpy array
    elif smooth_frames:
        # Allocate memory for smoothed signal.
        signal = np.zeros((int(len(sig_data) / tsamps_per_frame) , n_sig),
                       dtype=sig_data.dtype)

        # Transfer and average samples
        for ch in range(n_sig):
            if samps_per_frame[ch] == 1:
                signal[:, ch] = sig_data[sum(([0] + samps_per_frame)[:ch + 1])::tsamps_per_frame]
            else:
                if ch == 0:
                    startind = 0
                else:
                    startind = np.sum(samps_per_frame[:ch])
                signal[:,ch] = [np.average(sig_data[ind:ind+samps_per_frame[ch]]) for ind in range(startind,len(sig_data),tsamps_per_frame)]
        # Skew the signal
        signal = _skew_sig(signal, skew, n_sig, read_len, fmt, nan_replace)

    # Extra frames present without wanting smoothing. Return all
    # expanded samples.
    else:
        # List of 1d numpy arrays
        signal = []
        # Transfer over samples
        for ch in range(n_sig):
            # Indices of the flat signal that belong to the channel
            ch_indices = np.concatenate([np.array(range(samps_per_frame[ch]))
                                         + sum([0] + samps_per_frame[:ch])
                                         + tsamps_per_frame * framenum for framenum in range(int(len(sig_data)/tsamps_per_frame))])
            signal.append(sig_data[ch_indices])
        # Skew the signal
        signal = _skew_sig(signal, skew, n_sig, read_len, fmt, nan_replace, samps_per_frame)

    # Integrity check of signal shape after reading
    _check_sig_dims(signal, read_len, n_sig, samps_per_frame)

    return signal


def _dat_read_params(fmt, sig_len, byte_offset, skew, tsamps_per_frame,
                     sampfrom, sampto):
    """
    Calculate the parameters used to read and process a dat file, given
    its layout, and the desired sample range.

    Parameters
    ----------
    fmt : str
        The format of the dat file.
    sig_len : int
        The signal length (per channel) of the dat file.
    byte_offset : int
        The byte offset of the dat file.
    skew : list
        The skew for the signals of the dat file.
    tsamps_per_frame : int
        The total samples/frame for all channels of the dat file.
    sampfrom : int
        The starting sample number to be read from the signals.
    sampto : int
        The final sample number to be read from the signals.

    Returns
    -------
    start_byte : int
        The starting byte to read the dat file from. Always points to
        the start of a byte block for special formats.
    n_read_samples : int
        The number of flat samples to read from the dat file.
    block_floor_samples : int
        The extra samples read prior to the first desired sample, for
        special formats, in order to ensure entire byte blocks are read.
    extra_flat_samples : int
        The extra samples desired beyond what is contained in the file.
    nan_replace : list
        The number of samples to replace with NAN at the end of each
        signal, due to skew wanting samples beyond the file.

    Examples
    --------
    sig_len=100, t = 4 (total samples/frame), skew = [0, 2, 4, 5]
    sampfrom=0, sampto=100 --> read_len = 100, n_sampread = 100*t, extralen = 5, nan_replace = [0, 2, 4, 5]
    sampfrom=50, sampto=100 --> read_len = 50, n_sampread = 50*t, extralen = 5, nan_replace = [0, 2, 4, 5]
    sampfrom=0, sampto=50 --> read_len = 50, n_sampread = 55*t, extralen = 0, nan_replace = [0, 0, 0, 0]
    sampfrom=95, sampto=99 --> read_len = 4, n_sampread = 5*t, extralen = 4, nan_replace = [0, 1, 3, 4]

    """
    # First flat sample number to read (if all channels were flattened)
    start_flat_sample = sampfrom * tsamps_per_frame

    # Calculate the last flat sample number to read.
    # Cannot exceed sig_len * tsamps_per_frame, the number of samples
    # stored in the file. If extra 'samples' are desired by the skew,
    # keep track.
    # Where was the -sampfrom derived from? Why was it in the formula?
    if (sampto + max(skew)) > sig_len:
        end_flat_sample = sig_len * tsamps_per_frame
        extra_flat_samples = (sampto + max(skew) - sig_len) * tsamps_per_frame
    else:
        end_flat_sample = (sampto + max(skew)) * tsamps_per_frame
        extra_flat_samples = 0

    # Adjust the starting sample number to read from start of blocks for special fmts.
    # Keep track of how many preceeding samples are read, to be discarded later.
    if fmt == '212':
        # Samples come in groups of 2, in 3 byte blocks
        block_floor_samples = start_flat_sample % 2
        start_flat_sample = start_flat_sample - block_floor_samples
    elif fmt in ['310', '311']:
        # Samples come in groups of 3, in 4 byte blocks
        block_floor_samples = start_flat_sample % 3
        start_flat_sample = start_flat_sample - block_floor_samples
    else:
        block_floor_samples = 0

    # The starting byte to read from
    start_byte = byte_offset + int(start_flat_sample * BYTES_PER_SAMPLE[fmt])

    # The number of samples to read
    n_read_samples = end_flat_sample - start_flat_sample

    # The number of samples to replace with NAN at the end of each signal
    # due to skew wanting samples beyond the file
    nan_replace = [max(0, sampto + s - sig_len) for s in skew]

    return (start_byte, n_read_samples, block_floor_samples,
            extra_flat_samples, nan_replace)


def _required_byte_num(mode, fmt, n_samp):
    """
    Determine how many signal bytes are needed to read or write a
    number of desired samples from a dat file.

    Parameters
    ----------
    mode : str
        Whether the file is to be read or written: 'read' or 'write'.
    fmt : str
        The WFDB dat format.
    n_samp : int
        The number of samples wanted.

    Returns
    -------
    n_bytes : int
        The number of bytes required to read or write the file.

    Notes
    -----
    Read and write require the same number in most cases. An exception
    is fmt 311 for n_extra==2.

    """
    if fmt == '212':
        n_bytes = math.ceil(n_samp*1.5)
    elif fmt in ['310', '311']:
        n_extra = n_samp % 3

        if n_extra == 2:
            if fmt == '310':
                n_bytes = upround(n_samp * 4/3, 4)
            # 311
            else:
                if mode == 'read':
                    n_bytes = math.ceil(n_samp * 4/3)
                # Have to write more bytes for WFDB c to work
                else:
                    n_bytes = upround(n_samp * 4/3, 4)
        # 0 or 1
        else:
            n_bytes = math.ceil(n_samp * 4/3 )
    else:
        n_bytes = n_samp * BYTES_PER_SAMPLE[fmt]

    return int(n_bytes)


def _rd_dat_file(file_name, dir_name, pn_dir, fmt, start_byte, n_samp):
    """
    Read data from a dat file, either local or remote, into a 1d numpy
    array.

    This is the lowest level dat reading function (along with
    `_stream_dat` which this function may call), and is called by
    `_rd_dat_signals`.

    Parameters
    ----------
    file_name : str
        The name of the dat file.
    dir_name : str
        The full directory where the dat file(s) are located, if the dat
        file(s) are local.
    pn_dir : str
        The PhysioNet directory where the dat file(s) are located, if
        the dat file(s) are remote.
    fmt : list
        The formats of the dat files.
    start_byte : int
        The starting byte number to read from.
    n_samp : int
        The total number of samples to read. Does NOT need to create
        whole blocks for special format. Any number of samples should be
        readable.

    Returns
    -------
    sig_data : ndarray
        The data read from the dat file. The dtype varies depending on
        fmt. Byte aligned fmts are read in their final required format.
        Unaligned formats are read as uint8 to be further processed.

    Notes
    -----
    'channels', 'sampfrom', 'sampto', 'smooth_frames', and 'ignore_skew'
    are user desired input fields. All other parameters are
    specifications of the segment.

    """
    # element_count is the number of elements to read using np.fromfile
    # for local files
    # byte_count is the number of bytes to read for streaming files
    if fmt == '212':
        byte_count = _required_byte_num('read', '212', n_samp)
        element_count = byte_count
    elif fmt in ['310', '311']:
        byte_count = _required_byte_num('read', fmt, n_samp)
        element_count = byte_count
    else:
        element_count = n_samp
        byte_count = n_samp * BYTES_PER_SAMPLE[fmt]

    # Local dat file
    if pn_dir is None:
        with open(os.path.join(dir_name, file_name), 'rb') as fp:
            fp.seek(start_byte)
            # Numpy doesn't really like 24-bit data but we can make it work
            if DATA_LOAD_TYPES[fmt] == '<i3':
                raw_data_map = np.memmap(fp,
                                         dtype=np.dtype('i2'),
                                         mode='r')
                temp_data = np.frombuffer(raw_data_map, 'b').reshape(-1,3)[:,1:].flatten().view('i2')
                sig_data = np.fromstring(temp_data, dtype='i2')
            else:
                sig_data = np.fromfile(fp, 
                                       dtype=np.dtype(DATA_LOAD_TYPES[fmt]),
                                       count=element_count)
    # Stream dat file from Physionet
    else:
        if DATA_LOAD_TYPES[fmt] == '<i3':
            dtype_in = '<i3'
        else:
            dtype_in = np.dtype(DATA_LOAD_TYPES[fmt])

        sig_data = download._stream_dat(file_name, pn_dir, byte_count,
                                        start_byte, dtype_in)

    return sig_data


def _blocks_to_samples(sig_data, n_samp, fmt):
    """
    Convert uint8 blocks into signal samples for unaligned dat formats.

    Parameters
    ----------
    sig_data : ndarray
        The uint8 data blocks.
    n_samp : int
        The number of samples contained in the bytes.
    fmt : list
        The formats of the dat files.

    Returns
    -------
    sig : ndarray
        The numpy array of digital samples.

    """
    if fmt == '212':
        # Easier to process when dealing with whole blocks
        if n_samp % 2:
            n_samp += 1
            added_samps = 1
            sig_data = np.append(sig_data, np.zeros(1, dtype='uint8'))
        else:
            added_samps = 0

        sig_data = sig_data.astype('int16')
        sig = np.zeros(n_samp, dtype='int16')

        # One sample pair is stored in one byte triplet.

        # Even numbered samples
        sig[0::2] = sig_data[0::3] + 256 * np.bitwise_and(sig_data[1::3], 0x0f)
        # Odd numbered samples (len(sig) always > 1 due to processing of
        # whole blocks)
        sig[1::2] = sig_data[2::3] + 256*np.bitwise_and(sig_data[1::3] >> 4, 0x0f)

        # Remove trailing sample read within the byte block if
        # originally odd sampled
        if added_samps:
            sig = sig[:-added_samps]

        # Loaded values as un_signed. Convert to 2's complement form:
        # values > 2^11-1 are negative.
        sig[sig > 2047] -= 4096

    elif fmt == '310':
        # Easier to process when dealing with whole blocks
        if n_samp % 3:
            n_samp = upround(n_samp,3)
            added_samps = n_samp % 3
            sig_data = np.append(sig_data, np.zeros(added_samps, dtype='uint8'))
        else:
            added_samps = 0

        sig_data = sig_data.astype('int16')
        sig = np.zeros(n_samp, dtype='int16')

        # One sample triplet is stored in one byte quartet
        # First sample is 7 msb of first byte and 3 lsb of second byte.
        sig[0::3] = (sig_data[0::4] >> 1)[0:len(sig[0::3])] + 128 * np.bitwise_and(sig_data[1::4], 0x07)[0:len(sig[0::3])]
        # Second signal is 7 msb of third byte and 3 lsb of forth byte
        sig[1::3] = (sig_data[2::4] >> 1)[0:len(sig[1::3])] + 128 * np.bitwise_and(sig_data[3::4], 0x07)[0:len(sig[1::3])]
        # Third signal is 5 msb of second byte and 5 msb of forth byte
        sig[2::3] = np.bitwise_and((sig_data[1::4] >> 3), 0x1f)[0:len(sig[2::3])] + 32 * np.bitwise_and(sig_data[3::4] >> 3, 0x1f)[0:len(sig[2::3])]

        # Remove trailing samples read within the byte block if
        # originally not 3n sampled
        if added_samps:
            sig = sig[:-added_samps]

        # Loaded values as un_signed. Convert to 2's complement form:
        # values > 2^9-1 are negative.
        sig[sig > 511] -= 1024

    elif fmt == '311':
        # Easier to process when dealing with whole blocks
        if n_samp % 3:
            n_samp = upround(n_samp,3)
            added_samps = n_samp % 3
            sig_data = np.append(sig_data, np.zeros(added_samps, dtype='uint8'))
        else:
            added_samps = 0

        sig_data = sig_data.astype('int16')
        sig = np.zeros(n_samp, dtype='int16')

        # One sample triplet is stored in one byte quartet
        # First sample is first byte and 2 lsb of second byte.
        sig[0::3] = sig_data[0::4][0:len(sig[0::3])] + 256 * np.bitwise_and(sig_data[1::4], 0x03)[0:len(sig[0::3])]
        # Second sample is 6 msb of second byte and 4 lsb of third byte
        sig[1::3] = (sig_data[1::4] >> 2)[0:len(sig[1::3])] + 64 * np.bitwise_and(sig_data[2::4], 0x0f)[0:len(sig[1::3])]
        # Third sample is 4 msb of third byte and 6 msb of forth byte
        sig[2::3] = (sig_data[2::4] >> 4)[0:len(sig[2::3])] + 16 * np.bitwise_and(sig_data[3::4], 0x7f)[0:len(sig[2::3])]

        # Remove trailing samples read within the byte block if
        # originally not 3n sampled
        if added_samps:
            sig = sig[:-added_samps]

        # Loaded values as un_signed. Convert to 2's complement form.
        # Values > 2^9-1 are negative.
        sig[sig > 511] -= 1024
    return sig


def _skew_sig(sig, skew, n_sig, read_len, fmt, nan_replace, samps_per_frame=None):
    """
    Skew the signal, insert nans, and shave off end of array if needed.

    Parameters
    ----------
    sig : ndarray
        The original signal.
    skew : list
        List of samples to skew for each signal.
    n_sig : int
        The number of signals.
    read_len : int
        The total number of samples: Calculated by `sampto - sampfrom`
    fmt : list
        The formats of the dat files.
    nan_replace : list
        The indices to replace values with NAN.
    samps_per_frame : list, optional
        The number of samples of the orignal signal per channel.

    Returns
    -------
    sig : ndarray
        The new skewed and trimmed signal.

    Notes
    -----
    `fmt` is just for the correct NAN value.
    `samps_per_frame` is only used for skewing expanded signals.

    """
    if max(skew)>0:

        # Expanded frame samples. List of arrays.
        if isinstance(sig, list):
            # Shift the channel samples
            for ch in range(n_sig):
                if skew[ch]>0:
                    sig[ch][:read_len*samps_per_frame[ch]] = sig[ch][skew[ch]*samps_per_frame[ch]:]

            # Shave off the extra signal length at the end
            for ch in range(n_sig):
                sig[ch] = sig[ch][:read_len*samps_per_frame[ch]]

            # Insert nans where skewed signal overran dat file
            for ch in range(n_sig):
                if nan_replace[ch]>0:
                    sig[ch][-nan_replace[ch]:] = _digi_nan(fmt)
        # Uniform array
        else:
            # Shift the channel samples
            for ch in range(n_sig):
                if skew[ch]>0:
                    sig[:read_len, ch] = sig[skew[ch]:, ch]
            # Shave off the extra signal length at the end
            sig = sig[:read_len, :]

            # Insert nans where skewed signal overran dat file
            for ch in range(n_sig):
                if nan_replace[ch]>0:
                    sig[-nan_replace[ch]:, ch] = _digi_nan(fmt)

    return sig


def _check_sig_dims(sig, read_len, n_sig, samps_per_frame):
    """
    Integrity check of a signal's shape after reading.

    Parameters
    ----------
    sig : ndarray
        The original signal.
    read_len : int
        The signal length to read per channel. Calculated 
        by `sampto - sampfrom`.
    n_sig : int
        The number of signals.  
    samps_per_frame : list
        The number of samples of the orignal signal per channel.

    Returns
    -------
    N/A

    """
    if isinstance(sig, np.ndarray):
        if sig.shape != (read_len, n_sig):
            raise ValueError('Samples were not loaded correctly')
    else:
        if len(sig) != n_sig:
            raise ValueError('Samples were not loaded correctly')
        for ch in range(n_sig):
            if len(sig[ch]) != samps_per_frame[ch] * read_len:
                raise ValueError('Samples were not loaded correctly')


#------------------- /Reading Signals -------------------#


def _digi_bounds(fmt):
    """
    Return min and max digital values for each format type.

    Parmeters
    ---------
    fmt : str, list
        The WFDB dat format, or a list of them.

    Returns
    -------
    tuple (int, int)
        The min and max WFDB digital value per format type.

    """
    if isinstance(fmt, list):
        return [_digi_bounds(f) for f in fmt]

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


def _digi_nan(fmt):
    """
    Return the WFDB digital value used to store NAN for the format type.

    Parmeters
    ---------
    fmt : str, list
        The WFDB dat format, or a list of them.

    Returns
    -------
    int
        The WFDB digital value per format type.

    """
    if isinstance(fmt, list):
        return [_digi_nan(f) for f in fmt]

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
    Estimate the resolution of each signal in a multi-channel signal in
    bits. Maximum of 32 bits.

    Parameters
    ----------
    signals : ndarray, list
        A 2d numpy array representing a uniform multichannel signal, or
        a list of 1d numpy arrays representing multiple channels of
        signals with different numbers of samples per frame.

    Returns
    -------
    res : list
        A list of estimated integer resolutions for each channel.

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
            sorted_sig = np.sort(np.unique(signals[ch]))
        else:
            if signals.ndim == 1:
                sorted_sig = np.sort(np.unique(signals))
            else:
                sorted_sig = np.sort(np.unique(signals[:,ch]))

        min_inc = min(np.diff(sorted_sig))

        if min_inc == 0:
            # Case where signal is flat. Resolution is 0.
            res.append(0)
        else:
            nlevels = 1 + (sorted_sig[-1]-sorted_sig[0]) / min_inc
            if nlevels >= res_levels[-1]:
                res.append(32)
            else:
                res.append(np.where(res_levels>=nlevels)[0][0])

    return res


def _wfdb_fmt(bit_res, single_fmt=True):
    """
    Return the most suitable WFDB format(s) to use given signal
    resolutions.

    Parameters
    ----------
    bit_res : int, list
        The resolution of the signal, or a list of resolutions, in bits.
    single_fmt : bool, optional
        Whether to return the format for the maximum resolution signal.

    Returns
    -------
    fmt : str, list
        The most suitable WFDB format(s) used to encode the signal(s).

    """
    if isinstance(bit_res, list):
        # Return a single format
        if single_fmt:
            bit_res = [max(bit_res)] * len(bit_res)

        return [_wfdb_fmt(r) for r in bit_res]

    if bit_res <= 8:
        return '80'
    elif bit_res <= 12:
        return '212'
    elif bit_res <= 16:
        return '16'
    elif bit_res <= 24:
        return '24'
    else:
        return '32'


def _fmt_res(fmt, max_res=False):
    """
    Return the resolution of the WFDB dat format(s). Uses the BIT_RES
    dictionary, but accepts lists and other options.

    Parameters
    ----------
    fmt : str
        The WFDB format. Can be a list of valid fmts. If it is a list,
        and `max_res` is True, the list may contain None.
    max_res : bool, optional
        If given a list of fmts, whether to return the highest
        resolution.

    Returns
    -------
    bit_res : int, list
        The resolution(s) of the dat format(s) in bits.

    """
    if isinstance(fmt, list):
        if max_res:
            # Allow None
            bit_res = np.max([_fmt_res(f) for f in fmt if f is not None])
        else:
            bit_res = [_fmt_res(f) for f in fmt]
        return bit_res

    return BIT_RES[fmt]


def _np_dtype(bit_res, discrete):
    """
    Given the bit resolution of a signal, return the minimum numpy dtype
    used to store it.

    Parameters
    ----------
    bit_res : int
        The bit resolution.
    discrete : bool
        Whether the dtype is to be int or float.

    Returns
    -------
    dtype : str
        String numpy dtype used to store the signal of the given
        resolution.

    """
    bit_res = min(bit_res, 64)

    for np_res in [8, 16, 32, 64]:
        if bit_res <= np_res:
            break

    if discrete is True:
        return 'int' + str(np_res)
    else:
        # No float8 dtype
        return 'float' + str(max(np_res, 16))


def wr_dat_file(file_name, fmt, d_signal, byte_offset, expanded=False,
                e_d_signal=None, samps_per_frame=None, write_dir=''):
    """
    Write a dat file. All bytes are written one at a time to avoid
    endianness issues.

    Parameters
    ----------
    file_name : str
        Name of the dat file.
    fmt : str
        WFDB fmt of the dat file.
    d_signal : ndarray
        The digital conversion of the signal. Either a 2d numpy
        array or a list of 1d numpy arrays.
    byte_offset : int
        The byte offset of the dat file.
    expanded : bool, optional
        Whether to transform the `e_d_signal` attribute (True) or
        the `d_signal` attribute (False).
    d_signal : ndarray, optional
        The expanded digital conversion of the signal. Either a 2d numpy
        array or a list of 1d numpy arrays.
    samps_per_frame : list, optional
        The samples/frame for each signal of the dat file.
    write_dir : str, optional
        The directory to write the output file to.

    Returns
    -------
    N/A

    """
    # Combine list of arrays into single array
    if expanded:
        n_sig = len(e_d_signal)
        sig_len = int(len(e_d_signal[0])/samps_per_frame[0])
        # Effectively create MxN signal, with extra frame samples acting
        # like extra channels
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
        b_write = d_signal.astype('uint8')

    elif fmt == '212':
        # Each sample is represented by a 12 bit two's complement
        # amplitude. The first sample is obtained from the 12 least
        # significant bits of the first byte pair (stored least
        # significant byte first). The second sample is formed from the
        # 4 remaining bits of the first byte pair (which are the 4 high
        # bits of the 12-bit sample) and the next byte (which contains
        # the remaining 8 bits of the second sample). The process is
        # repeated for each successive pair of samples.

        # convert to 12 bit two's complement
        d_signal[d_signal<0] = d_signal[d_signal<0] + 4096

        # Concatenate into 1D
        d_signal = d_signal.reshape(-1)

        n_samp = len(d_signal)
        # use this for byte processing
        processn_samp = n_samp

        # Odd numbered number of samples. Fill in extra blank for
        # following byte calculation.
        if processn_samp % 2:
            d_signal = np.concatenate([d_signal, np.array([0])])
            processn_samp +=1

        # The individual bytes to write
        b_write = np.zeros([int(1.5*processn_samp)], dtype = 'uint8')

        # Fill in the byte triplets

        # Triplet 1 from lowest 8 bits of sample 1
        b_write[0::3] = d_signal[0::2] & 255
        # Triplet 2 from highest 4 bits of samples 1 (lower) and 2 (upper)
        b_write[1::3] = ((d_signal[0::2] & 3840) >> 8) + ((d_signal[1::2] & 3840) >> 4)
        # Triplet 3 from lowest 8 bits of sample 2
        b_write[2::3] = d_signal[1::2] & 255

        # If we added an extra sample for byte calculation, remove the last byte (don't write)
        if n_samp % 2:
            b_write = b_write[:-1]

    elif fmt == '16':
        # convert to 16 bit two's complement
        d_signal[d_signal<0] = d_signal[d_signal<0] + 65536
        # Split samples into separate bytes using binary masks
        b1 = d_signal & [255]*n_sig
        b2 = ( d_signal & [65280]*n_sig ) >> 8
        # Interweave the bytes so that the same samples' bytes are consecutive
        b1 = b1.reshape((-1, 1))
        b2 = b2.reshape((-1, 1))
        b_write = np.concatenate((b1, b2), axis=1)
        b_write = b_write.reshape((1,-1))[0]
        # Convert to un_signed 8 bit dtype to write
        b_write = b_write.astype('uint8')
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
        b_write = np.concatenate((b1, b2, b3), axis=1)
        b_write = b_write.reshape((1,-1))[0]
        # Convert to un_signed 8 bit dtype to write
        b_write = b_write.astype('uint8')

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
        b_write = np.concatenate((b1, b2, b3, b4), axis=1)
        b_write = b_write.reshape((1,-1))[0]
        # Convert to un_signed 8 bit dtype to write
        b_write = b_write.astype('uint8')
    else:
        raise ValueError('This library currently only supports writing the following formats: 80, 16, 24, 32')

    # Byte offset in the file
    if byte_offset is not None and byte_offset>0:
        print('Writing file '+file_name+' with '+str(byte_offset)+' empty leading bytes')
        b_write = np.append(np.zeros(byte_offset, dtype = 'uint8'), b_write)

    # Write the bytes to the file
    with open(os.path.join(write_dir, file_name),'wb') as f:
        b_write.tofile(f)


def describe_list_indices(full_list):
    """
    Describe the indices of the given list.

    Parameters
    ----------
    full_list : list
        The list of items to order.

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


def _infer_sig_len(file_name, fmt, n_sig, dir_name, pn_dir=None):
    """
    Infer the length of a signal from a dat file.

    Parameters
    ----------
    file_name : str
        Name of the dat file.
    fmt : str
        WFDB fmt of the dat file.
    n_sig : int
        Number of signals contained in the dat file.
    dir_name : str
        The full directory where the dat file(s) are located, if the dat
        file(s) are local.
    pn_dir : str, optional
        The PhysioNet directory where the dat file(s) are located, if
        the dat file(s) are remote.

    Returns
    -------
    sig_len : int
        The length of the signal.

    Notes
    -----
    sig_len * n_sig * bytes_per_sample == file_size

    """
    if pn_dir is None:
        file_size = os.path.getsize(os.path.join(dir_name, file_name))
    else:
        file_size = download._remote_file_size(file_name=file_name,
                                               pn_dir=pn_dir)

    sig_len = int(file_size / (BYTES_PER_SAMPLE[fmt] * n_sig))

    return sig_len


def downround(x, base):
    """
    Round <x> down to nearest <base>.

    Parameters
    ---------
    x : str, int, float
        The number that will be rounded down.
    base : int, float
        The base to be rounded down to.

    Returns
    -------
    float
        The rounded down result of <x> down to nearest <base>.

    """
    return base * math.floor(float(x)/base)


def upround(x, base):
    """
    Round <x> up to nearest <base>.

    Parameters
    ---------
    x : str, int, float
        The number that will be rounded up.
    base : int, float
        The base to be rounded up to.

    Returns
    -------
    float
        The rounded up result of <x> up to nearest <base>.

    """
    return base * math.ceil(float(x)/base)
