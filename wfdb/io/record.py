import datetime
import multiprocessing
import posixpath
import re

import numpy as np
import os
import pandas as pd
import requests
import math
import functools
import struct
import pdb

from wfdb.io import _header
from wfdb.io import _signal
from wfdb.io import download


# -------------- WFDB Signal Calibration and Classification ---------- #


# Unit scales used for default display scales. The unit scale that the
# class should measure. 'No Unit' will also be allowed in all cases.
# * Will it always be 1?
unit_scale = {
    'voltage': ['pV', 'nV', 'uV', 'mV', 'V', 'kV'],
    'temperature': ['C', 'F'],
    'pressure': ['mmHg'],
    'no_unit': ['NU'],
    'percentage': ['%'],
    'heart_rate': ['bpm'],
}

"""
Signal classes that WFDB signals should fall under. The indexes are the
abbreviated class names.

Notes
-----
This will be used to automatically classify signals in classes based
on their names.

"""

SIGNAL_CLASSES = pd.DataFrame(
    index=['bp', 'co2', 'co', 'ecg', 'eeg', 'emg', 'eog', 'hr', 'mmg',
           'o2', 'pleth', 'resp', 'scg', 'stat', 'st', 'temp', 'unknown'],
    columns=['description', 'unit_scale', 'signal_names'],
    data=[['Blood Pressure', 'pressure', ['bp','abp','pap','cvp']], # bp
          ['Carbon Dioxide', 'percentage', ['co2', 'pco2']], # co2
          ['Carbon Monoxide', 'percentage', ['co']], # co
          ['Electrocardiogram', 'voltage', ['i','ii','iii','iv','v','avr']], # ecg
          ['Electroencephalogram', 'voltage', ['eeg']], # eeg
          ['Electromyograph', 'voltage', ['emg']], # emg
          ['Electrooculograph', 'voltage', ['eog']], # eog
          ['Heart Rate', 'heart_rate', ['hr']], # hr
          ['Magnetomyograph', 'voltage', ['mmg']], # mmg
          ['Oxygen', 'percentage', ['o2', 'spo2']], # o2
          ['Plethysmograph', 'pressure', ['pleth']], # pleth
          ['Respiration', 'no_unit', ['resp']], # resp
          ['Seismocardiogram', 'no_unit', ['scg']], # scg
          ['Status', 'no_unit', ['stat', 'status']], # stat
          ['ST Segment', '', ['st']], # st. This is not a signal?
          ['Temperature', 'temperature', ['temp']], # temp
          ['Unknown Class', 'no_unit', []], # unknown. special class.
    ]
)

"""
All of the default units to be used if the value for unit
while reading files returns "N/A".

Note
----
All of the key values here are in all lowercase characters
to remove duplicates by different cases.

"""

SIG_UNITS = {
    'a': 'uV',
    'abdomen': 'uV',
    'abdo': 'V',
    'abp': 'mmHg',
    'airflow': 'V',
    'ann': 'units',
    'art': 'mmHg',
    'atip': 'mV',
    'av': 'mV',
    'bp': 'mmHg',
    'c': 'uV',
    'c.o.': 'lpm',
    'co': 'Lpm',
    'cs': 'mV',
    'cvp': 'mmHg',
    'direct': 'uV',
    'ecg': 'mV',
    'edr': 'units',
    'eeg': 'mV',
    'emg': 'mV',
    'eog': 'mV',
    'event': 'mV',
    'f': 'uV',
    'fecg': 'mV',
    'fhr': 'bpm',
    'foobar': 'mmHg',
    'hr': 'bpm',
    'hva': 'mV',
    'i': 'mV',
    'ibp': 'mmHg',
    'mcl': 'mV',
    'nbp': 'mmHg',
    'o': 'uV',
    'p': 'mmHg',
    'pap': 'mmHg',
    'pawp': 'mmHg',
    'pcg': 'mV',
    'pleth': 'mV',
    'pr': 'bpm',
    'pulse': 'bpm',
    'record': 'mV',
    'resp': 'l',
    'sao2': '%',
    'so2': '%',
    'spo2': '%',
    'sv': 'ml',
    't': 'uV',
    'tblood': 'degC',
    'temp': 'degC',
    'thorax': 'mV',
    'thor': 'V',
    'v': 'mV',
    'uc': 'nd',
    'vtip': 'mV'
}


class BaseRecord(object):
    """
    The base WFDB class extended by the Record and MultiRecord classes.
    
    Attributes
    ----------
    record_name : str, optional
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    n_sig : int, optional
        Total number of signals.
    fs : int, float, optional
        The sampling frequency of the record.
    counter_freq : float, optional
        The frequency used to start counting.
    base_counter : float, optional
        The counter used at the start of the file.
    sig_len : int, optional
        The total length of the signal.
    base_time : str, optional
        A string of the record's start time in 24h 'HH:MM:SS(.ms)' format.
    base_date : str, optional
        A string of the record's start date in 'DD/MM/YYYY' format.
    comments : list, optional
        A list of string comments to be written to the header file.
    sig_name : str, optional
        A list of strings giving the signal name of each signal channel.

    """
    # The base WFDB class extended by the Record and MultiRecord classes.
    def __init__(self, record_name=None, n_sig=None,
                 fs=None, counter_freq=None, base_counter=None,
                 sig_len=None, base_time=None, base_date=None,
                 comments=None, sig_name=None):
        self.record_name = record_name
        self.n_sig = n_sig
        self.fs = fs
        self.counter_freq = counter_freq
        self.base_counter = base_counter
        self.sig_len = sig_len
        self.base_time = base_time
        self.base_date = base_date
        self.comments = comments
        self.sig_name = sig_name


    def check_field(self, field, required_channels='all'):
        """
        Check whether a single field is valid in its basic form. Does
        not check compatibility with other fields.

        Parameters
        ----------
        field : str
            The field name.
        required_channels : list, optional
            Used for signal specification fields. All channels are
            checked for their integrity if present, but channels that do
            not lie in this field may be None.

        Returns
        -------
        N/A

        Notes
        -----
        This function is called from wrheader to check fields before
        writing. It is also supposed to be usable at any point to
        check a specific field.

        """
        item = getattr(self, field)
        if item is None:
            raise Exception('Missing field required: %s' % field)

        # We should have a list specifying these automatically.

        # Whether the item should be a list. Watch out for required_channels for `segments`
        expect_list = True if field in LIST_FIELDS else False

        # Check the type of the field (and of its elements if it should
        # be a list)
        _check_item_type(item, field_name=field,
                        allowed_types=ALLOWED_TYPES[field],
                        expect_list=expect_list,
                        required_channels=required_channels)

        # Individual specific field checks
        if field in ['d_signal', 'p_signal']:
            check_np_array(item=item, field_name=field, ndim=2,
                           parent_class=(lambda f: np.integer if f == 'd_signal' else np.floating)(field))
        elif field in ['e_d_signal', 'e_p_signal']:
            for ch in range(len(item)):
                check_np_array(item=item[ch], field_name=field,
                               ndim=1, parent_class=(lambda f: np.integer if f == 'e_d_signal' else np.floating)(field),
                               channel_num=ch)

        # Record specification fields
        elif field == 'record_name':
            # Allow letters, digits, hyphens, and underscores.
            accepted_string = re.match('[-\w]+', self.record_name)
            if not accepted_string or accepted_string.string != self.record_name:
                raise ValueError('record_name must only comprise of letters, digits, hyphens, and underscores.')
        elif field == 'n_seg':
            if self.n_seg <= 0:
                raise ValueError('n_seg must be a positive integer')
        elif field == 'n_sig':
            if self.n_sig <= 0:
                raise ValueError('n_sig must be a positive integer')
        elif field == 'fs':
            if self.fs <= 0:
                raise ValueError('fs must be a positive number')
        elif field == 'counter_freq':
            if self.counter_freq <= 0:
                raise ValueError('counter_freq must be a positive number')
        elif field == 'base_counter':
            if self.base_counter <= 0:
                raise ValueError('base_counter must be a positive number')
        elif field == 'sig_len':
            if self.sig_len < 0:
                raise ValueError('sig_len must be a non-negative integer')

        # Signal specification fields
        elif field in _header.SIGNAL_SPECS.index:
            if required_channels == 'all':
                required_channels = range(len(item))

            for ch in range(len(item)):
                # If the element is allowed to be None
                if ch not in required_channels:
                    if item[ch] is None:
                        continue

                if field == 'file_name':
                    # Check for file_name characters
                    accepted_string = re.match('[-\w]+\.?[\w]+', item[ch])
                    if not accepted_string or accepted_string.string != item[ch]:
                        raise ValueError('File names should only contain alphanumerics, hyphens, and an extension. eg. record-100.dat')
                    # Check that dat files are grouped together
                    if not is_monotonic(self.file_name):
                        raise ValueError('Signals in a record that share a given file must be consecutive.')
                elif field == 'fmt':
                    if item[ch] not in _signal.DAT_FMTS:
                        raise ValueError('File formats must be valid WFDB dat formats:', _signal.DAT_FMTS)
                elif field == 'samps_per_frame':
                    if item[ch] < 1:
                        raise ValueError('samps_per_frame values must be positive integers')
                elif field == 'skew':
                    if item[ch] < 0:
                        raise ValueError('skew values must be non-negative integers')
                elif field == 'byte_offset':
                    if item[ch] < 0:
                        raise ValueError('byte_offset values must be non-negative integers')
                elif field == 'adc_gain':
                    if item[ch] <= 0:
                        raise ValueError('adc_gain values must be positive')
                elif field == 'baseline':
                    # Original WFDB library 10.5.24 only has 4 bytes for baseline.
                    if item[ch] < -2147483648 or item[ch] > 2147483648:
                        raise ValueError('baseline values must be between -2147483648 (-2^31) and 2147483647 (2^31 -1)')
                elif field == 'units':
                    if re.search('\s', item[ch]):
                        raise ValueError('units strings may not contain whitespaces.')
                elif field == 'adc_res':
                    if item[ch] < 0:
                        raise ValueError('adc_res values must be non-negative integers')
                elif field == 'block_size':
                    if item[ch] < 0:
                        raise ValueError('block_size values must be non-negative integers')
                elif field == 'sig_name':
                    if re.search('\s', item[ch]):
                        raise ValueError('sig_name strings may not contain whitespaces.')
                    if len(set(item)) != len(item):
                        raise ValueError('sig_name strings must be unique.')

        # Segment specification fields and comments
        elif field in _header.SEGMENT_SPECS.index:
            for ch in range(len(item)):
                if field == 'seg_name':
                    # Segment names must be alphanumerics or just a single '~'
                    if item[ch] == '~':
                        continue
                    accepted_string = re.match('[-\w]+', item[ch])
                    if not accepted_string or accepted_string.string != item[ch]:
                        raise ValueError("Non-null segment names may only contain alphanumerics and dashes. Null segment names must be set to '~'")
                elif field == 'seg_len':
                    # For records with more than 1 segment, the first
                    # segment may be the layout specification segment
                    # with a length of 0
                    min_len = 0 if ch == 0 else 1
                    if item[ch] < min_len:
                        raise ValueError('seg_len values must be positive integers. Only seg_len[0] may be 0 to indicate a layout segment')
                # Comment field
                elif field == 'comments':
                    if item[ch].startswith('#'):
                        print("Note: comment strings do not need to begin with '#'. This library adds them automatically.")
                    if re.search('[\t\n\r\f\v]', item[ch]):
                        raise ValueError('comments may not contain tabs or newlines (they may contain spaces and underscores).')


    def check_read_inputs(self, sampfrom, sampto, channels, physical,
                          smooth_frames, return_res):
        """
        Ensure that input read parameters (from rdsamp) are valid for
        the record.

        Parameters
        ----------
        sampfrom : int
            The starting sample number to read for all channels.
        sampto : int, 'end'
            The sample number at which to stop reading for all channels.
            Reads the entire duration by default.
        channels : list
            List of integer indices specifying the channels to be read.
            Reads all channels by default.
        physical : bool
            Specifies whether to return signals in physical units in the
            `p_signal` field (True), or digital units in the `d_signal`
            field (False).
        smooth_frames : bool
            Used when reading records with signals having multiple samples
            per frame. Specifies whether to smooth the samples in signals
            with more than one sample per frame and return an (MxN) uniform
            numpy array as the `d_signal` or `p_signal` field (True), or to
            return a list of 1d numpy arrays containing every expanded
            sample as the `e_d_signal` or `e_p_signal` field (False).
        return_res : int
            The numpy array dtype of the returned signals. Options are: 64,
            32, 16, and 8, where the value represents the numpy int or float
            dtype. Note that the value cannot be 8 when physical is True
            since there is no float8 format.

        Returns
        -------
        N/A

        """
        # Data Type Check
        if not hasattr(sampfrom, '__index__'):
            raise TypeError('sampfrom must be an integer')
        if not hasattr(sampto, '__index__'):
            raise TypeError('sampto must be an integer')
        if not isinstance(channels, list):
            raise TypeError('channels must be a list of integers')

        # Duration Ranges
        if sampfrom < 0:
            raise ValueError('sampfrom must be a non-negative integer')
        if sampfrom > self.sig_len:
            raise ValueError('sampfrom must be shorter than the signal length')
        if sampto < 0:
            raise ValueError('sampto must be a non-negative integer')
        if sampto > self.sig_len:
            raise ValueError('sampto must be shorter than the signal length')
        if sampto <= sampfrom:
            raise ValueError('sampto must be greater than sampfrom')

        # Channel Ranges
        if len(channels):
            if min(channels) < 0:
                raise ValueError('Input channels must all be non-negative integers')
            if max(channels) > self.n_sig - 1:
                raise ValueError('Input channels must all be lower than the total number of channels')

        if return_res not in [64, 32, 16, 8]:
            raise ValueError("return_res must be one of the following: 64, 32, 16, 8")
        if physical is True and return_res == 8:
            raise ValueError("return_res must be one of the following when physical is True: 64, 32, 16")

        # Cannot expand multiple samples/frame for multi-segment records
        if isinstance(self, MultiRecord):
            if smooth_frames is False:
                raise ValueError('This package version cannot expand all samples when reading multi-segment records. Must enable frame smoothing.')


    def _adjust_datetime(self, sampfrom):
        """
        Adjust date and time fields to reflect user input if possible.

        Helper function for the `_arrange_fields` of both Record and
        MultiRecord objects.

        Parameters
        ----------
        sampfrom : int
            The starting sample number to read for all channels.

        Returns
        -------
        N/A

        """
        if sampfrom:
            dt_seconds = sampfrom / self.fs
            if self.base_date and self.base_time:
                self.base_datetime = datetime.datetime.combine(self.base_date,
                                                               self.base_time)
                self.base_datetime += datetime.timedelta(seconds=dt_seconds)
                self.base_date = self.base_datetime.date()
                self.base_time = self.base_datetime.time()
            # We can calculate the time even if there is no date
            elif self.base_time:
                tmp_datetime = datetime.datetime.combine(
                    datetime.datetime.today().date(), self.base_time)
                self.base_time = (tmp_datetime
                                  + datetime.timedelta(seconds=dt_seconds)).time()
            # Cannot calculate date or time if there is only date


class Record(BaseRecord, _header.HeaderMixin, _signal.SignalMixin):
    """
    The class representing single segment WFDB records.

    Record objects can be created using the initializer, by reading a WFDB
    header with `rdheader`, or a WFDB record (header and associated dat files)
    with `rdrecord`.

    The attributes of the Record object give information about the record as
    specified by: https://www.physionet.org/physiotools/wag/header-5.htm

    In addition, the d_signal and p_signal attributes store the digital and
    physical signals of WFDB records with at least one channel.

    Attributes
    ----------
    p_signal : ndarray, optional
        An (MxN) 2d numpy array, where M is the signal length. Gives the
        physical signal values intended to be written. Either p_signal or
        d_signal must be set, but not both. If p_signal is set, this method will
        use it to perform analogue-digital conversion, writing the resultant
        digital values to the dat file(s). If fmt is set, gain and baseline must
        be set or unset together. If fmt is unset, gain and baseline must both
        be unset.
    d_signal : ndarray, optional
        An (MxN) 2d numpy array, where M is the signal length. Gives the
        digital signal values intended to be directly written to the dat
        file(s). The dtype must be an integer type. Either p_signal or d_signal
        must be set, but not both. In addition, if d_signal is set, fmt, gain
        and baseline must also all be set.
    e_p_signal : ndarray, optional
        The expanded physical conversion of the signal. Either a 2d numpy
        array or a list of 1d numpy arrays.
    e_d_signal : ndarray, optional
        The expanded digital conversion of the signal. Either a 2d numpy
        array or a list of 1d numpy arrays.
    record_name : str, optional
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    n_sig : int, optional
        Total number of signals.
    fs : float, optional
        The sampling frequency of the record.
    counter_freq : float, optional
        The frequency used to start counting.
    base_counter : float, optional
        The counter used at the start of the file.
    sig_len : int, optional
        The total length of the signal.
    base_time : str, optional
        A string of the record's start time in 24h 'HH:MM:SS(.ms)' format.
    base_date : str, optional
        A string of the record's start date in 'DD/MM/YYYY' format.
    file_name : str, optional
        The name of the file used for analysis.
    fmt : list, optional
        A list of strings giving the WFDB format of each file used to store each
        channel. Accepted formats are: '80','212','16','24', and '32'. There are
        other WFDB formats as specified by:
        https://www.physionet.org/physiotools/wag/signal-5.htm
        but this library will not write (though it will read) those file types.
    samps_per_frame : int, optional
        The total number of samples per frame.
    skew : float, optional
        The offset used to allign signals.
    byte_offset : int, optional
        The byte offset used to allign signals.
    adc_gain : list, optional
        A list of numbers specifying the ADC gain.
    baseline : list, optional
        A list of integers specifying the digital baseline.
    units : list, optional
        A list of strings giving the units of each signal channel.  
    adc_res: int, optional
        The value produced by the ADC given a given Volt input.  
    adc_zero: int, optional
        The value produced by the ADC given a 0 Volt input.
    init_value : list, optional
        The initial value of the signal.
    checksum : list, int, optional
        The checksum of the signal.
    block_size : str, optional
        The dimensions of the field data.
    sig_name : list, optional
        A list of strings giving the signal name of each signal channel.
    comments : list, optional
        A list of string comments to be written to the header file.

    Examples
    --------
    >>> record = wfdb.Record(record_name='r1', fs=250, n_sig=2, sig_len=1000,
                         file_name=['r1.dat','r1.dat'])

    """
    def __init__(self, p_signal=None, d_signal=None,
                 e_p_signal=None, e_d_signal=None,
                 record_name=None, n_sig=None,
                 fs=None, counter_freq=None, base_counter=None,
                 sig_len=None, base_time=None, base_date=None,
                 file_name=None, fmt=None, samps_per_frame=None,
                 skew=None, byte_offset=None, adc_gain=None,
                 baseline=None, units=None, adc_res=None,
                 adc_zero=None, init_value=None, checksum=None,
                 block_size=None, sig_name=None, comments=None):

        # Note the lack of the 'n_seg' field. Single segment records cannot
        # have this field. Even n_seg = 1 makes the header a multi-segment
        # header.
        super(Record, self).__init__(record_name, n_sig,
                    fs, counter_freq, base_counter, sig_len,
                    base_time, base_date, comments, sig_name)

        self.p_signal = p_signal
        self.d_signal = d_signal
        self.e_p_signal = e_p_signal
        self.e_d_signal = e_d_signal

        self.file_name = file_name
        self.fmt = fmt
        self.samps_per_frame = samps_per_frame
        self.skew = skew
        self.byte_offset = byte_offset
        self.adc_gain = adc_gain
        self.baseline = baseline
        self.units = units
        self.adc_res = adc_res
        self.adc_zero = adc_zero
        self.init_value = init_value
        self.checksum = checksum
        self.block_size = block_size


    # Equal comparison operator for objects of this type
    def __eq__(self, other, verbose=False):
        """
        Equal comparison operator for objects of this type.

        Parameters
        ----------
        other : object
            The object that is being compared to self.
        verbose : bool, optional
            Whether to print details about equality (True) or not (False).

        Returns
        -------
        bool
            Determines if the objects are equal (True) or not equal (False).

        """
        att1 = self.__dict__
        att2 = other.__dict__

        if set(att1.keys()) != set(att2.keys()):
            if verbose:
                print('Attributes members mismatch.')
            return False

        for k in att1.keys():

            v1 = att1[k]
            v2 = att2[k]

            if type(v1) != type(v2):
                if verbose:
                    print('Mismatch in attribute: %s' % k, v1, v2)
                return False

            if type(v1) == np.ndarray:
                # Necessary for nans
                np.testing.assert_array_equal(v1, v2)
            else:
                if v1 != v2:
                    if verbose:
                        print('Mismatch in attribute: %s' % k, v1, v2)
                    return False

        return True


    def wrsamp(self, expanded=False, write_dir=''):
        """
        Write a WFDB header file and any associated dat files from this
        object.

        Parameters
        ----------
        expanded : bool, optional
            Whether to write the expanded signal (e_d_signal) instead
            of the uniform signal (d_signal).
        write_dir : str, optional
            The directory in which to write the files.

        Returns
        -------
        N/A

        """
        # Perform field validity and cohesion checks, and write the
        # header file.
        self.wrheader(write_dir=write_dir)
        if self.n_sig > 0:
            # Perform signal validity and cohesion checks, and write the
            # associated dat files.
            self.wr_dats(expanded=expanded, write_dir=write_dir)


    def _arrange_fields(self, channels, sampfrom=0, expanded=False):
        """
        Arrange/edit object fields to reflect user channel and/or signal
        range input.

        Parameters
        ----------
        channels : list
            List of channel numbers specified.
        sampfrom : int, optional
            Starting sample number read.
        expanded : bool, optional
            Whether the record was read in expanded mode.

        Returns
        -------
        N/A

        """
        # Rearrange signal specification fields
        for field in _header.SIGNAL_SPECS.index:
            item = getattr(self, field)
            setattr(self, field, [item[c] for c in channels])

        # Expanded signals - multiple samples per frame.
        if expanded:
            # Checksum and init_value to be updated if present
            # unless the whole signal length was input
            if self.sig_len != int(len(self.e_d_signal[0]) / self.samps_per_frame[0]):
                self.checksum = self.calc_checksum(expanded)
                self.init_value = [s[0] for s in self.e_d_signal]

            self.n_sig = len(channels)
            self.sig_len = int(len(self.e_d_signal[0]) / self.samps_per_frame[0])

        # MxN numpy array d_signal
        else:
            # Checksum and init_value to be updated if present
            # unless the whole signal length was input
            if self.sig_len != self.d_signal.shape[0]:

                if self.checksum is not None:
                    self.checksum = self.calc_checksum()
                if self.init_value is not None:
                    ival = list(self.d_signal[0, :])
                    self.init_value = [int(i) for i in ival]

            # Update record specification parameters
            # Important that these get updated after^^
            self.n_sig = len(channels)
            self.sig_len = self.d_signal.shape[0]

        # Adjust date and time if necessary
        self._adjust_datetime(sampfrom=sampfrom)


class MultiRecord(BaseRecord, _header.MultiHeaderMixin):
    """
    The class representing multi-segment WFDB records.

    MultiRecord objects can be created using the initializer, or by reading a
    multi-segment WFDB record using 'rdrecord' with the `m2s` (multi to single)
    input parameter set to False.

    The attributes of the MultiRecord object give information about the entire
    record as specified by: https://www.physionet.org/physiotools/wag/header-5.htm

    In addition, the `segments` parameter is a list of Record objects
    representing each individual segment, or None representing empty segments,
    of the entire multi-segment record.

    Notably, this class has no attribute representing the signals as a whole.
    The 'multi_to_single' instance method can be called on MultiRecord objects
    to return a single segment representation of the record as a Record object.
    The resulting Record object will have its 'p_signal' field set.

    Attributes
    ----------
    segments : list, optional
        The segments to be read.
    layout : str, optional
        Whether the record will be 'fixed' or 'variable'.
    record_name : str, optional
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    n_sig : int, optional
        Total number of signals.
    fs : int, float, optional
        The sampling frequency of the record.
    counter_freq : float, optional
        The frequency used to start counting.
    base_counter : float, optional
        The counter used at the start of the file.
    sig_len : int, optional
        The total length of the signal.
    base_time : str, optional
        A string of the record's start time in 24h 'HH:MM:SS(.ms)' format.
    base_date : str, optional
        A string of the record's start date in 'DD/MM/YYYY' format.
    seg_name : str, optional
        The name of the segment.
    seg_len : int, optional
        The length of the segment.
    comments : list, optional
        A list of string comments to be written to the header file.
    sig_name : str, optional
        A list of strings giving the signal name of each signal channel.
    sig_segments : list, optional
        The signal segments to be read.

    Examples
    --------
    >>> record_m = wfdb.MultiRecord(record_name='rm', fs=50, n_sig=8,
                                    sig_len=9999, seg_name=['rm_1', '~', rm_2'],
                                    seg_len=[800, 200, 900])
    >>> # Get a MultiRecord object
    >>> record_s = wfdb.rdsamp('s00001-2896-10-10-00-31', m2s=False)
    >>> # Turn it into a single record
    >>> record_s = record_s.multi_to_single()

    record_s initially stores a `MultiRecord` object, and is then converted into
    a `Record` object.

    """
    def __init__(self, segments=None, layout=None,
                 record_name=None, n_sig=None, fs=None,
                 counter_freq=None, base_counter=None,
                 sig_len=None, base_time=None, base_date=None,
                 seg_name=None, seg_len=None, comments=None,
                 sig_name=None, sig_segments=None):


        super(MultiRecord, self).__init__(record_name, n_sig,
                    fs, counter_freq, base_counter, sig_len,
                    base_time, base_date, comments, sig_name)

        self.layout = layout
        self.segments = segments
        self.seg_name = seg_name
        self.seg_len = seg_len
        self.sig_segments = sig_segments


    def wrsamp(self, write_dir=''):
        """
        Write a multi-segment header, along with headers and dat files
        for all segments, from this object.

        Parameters
        ----------
        write_dir : str, optional
            The directory in which to write the files.

        Returns
        -------
        N/A

        """
        # Perform field validity and cohesion checks, and write the
        # header file.
        self.wrheader(write_dir=write_dir)
        # Perform record validity and cohesion checks, and write the
        # associated segments.
        for seg in self.segments:
            seg.wrsamp(write_dir=write_dir)


    def _check_segment_cohesion(self):
        """
        Check the cohesion of the segments field with other fields used
        to write the record.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        """
        if self.n_seg != len(self.segments):
            raise ValueError("Length of segments must match the 'n_seg' field")

        for i in range(n_seg):
            s = self.segments[i]

            # If segment 0 is a layout specification record, check that its file names are all == '~''
            if i == 0 and self.seg_len[0] == 0:
                for file_name in s.file_name:
                    if file_name != '~':
                        raise ValueError("Layout specification records must have all file_names named '~'")

            # Sampling frequencies must all match the one in the master header
            if s.fs != self.fs:
                raise ValueError("The 'fs' in each segment must match the overall record's 'fs'")

            # Check the signal length of the segment against the corresponding seg_len field
            if s.sig_len != self.seg_len[i]:
                raise ValueError('The signal length of segment '+str(i)+' does not match the corresponding segment length')

            totalsig_len = totalsig_len + getattr(s, 'sig_len')

        # No need to check the sum of sig_lens from each segment object against sig_len
        # Already effectively done it when checking sum(seg_len) against sig_len


    def _required_segments(self, sampfrom, sampto):
        """
        Determine the segments and the samples within each segment in a
        multi-segment record, that lie within a sample range.

        Parameters
        ----------
        sampfrom : int
            The starting sample number to read for each channel.
        sampto : int
            The sample number at which to stop reading for each channel.

        Returns
        -------
        seg_numbers : list
            List of segment numbers to read.
        readsamps : list
            List of sample numbers to be read.

        """
        # The starting segment with actual samples
        if self.layout == 'fixed':
            startseg = 0
        else:
            startseg = 1

        # Cumulative sum of segment lengths (ignoring layout segment)
        cumsumlengths = list(np.cumsum(self.seg_len[startseg:]))
        # Get first segment
        seg_numbers = [[sampfrom < cs for cs in cumsumlengths].index(True)]
        # Get final segment
        if sampto == cumsumlengths[len(cumsumlengths) - 1]:
            seg_numbers.append(len(cumsumlengths) - 1)
        else:
            seg_numbers.append([sampto <= cs for cs in cumsumlengths].index(True))

        # Add 1 for variable layout records
        seg_numbers = list(np.add(seg_numbers,startseg))

        # Obtain the sampfrom and sampto to read for each segment
        if seg_numbers[1] == seg_numbers[0]:
            # Only one segment to read
            seg_numbers = [seg_numbers[0]]
            # The segment's first sample number relative to the entire record
            segstartsamp = sum(self.seg_len[0:seg_numbers[0]])
            readsamps = [[sampfrom-segstartsamp, sampto-segstartsamp]]

        else:
            # More than one segment to read
            seg_numbers = list(range(seg_numbers[0], seg_numbers[1]+1))
            readsamps = [[0, self.seg_len[s]] for s in seg_numbers]

            # Starting sample for first segment.
            readsamps[0][0] = sampfrom - ([0] + cumsumlengths)[seg_numbers[0]-startseg]

            # End sample for last segment
            readsamps[-1][1] = sampto - ([0] + cumsumlengths)[seg_numbers[-1]-startseg]

        return (seg_numbers, readsamps)


    def _required_channels(self, seg_numbers, channels, dir_name, pn_dir):
        """
        Get the channel numbers to be read from each specified segment,
        given the channel numbers specified for the entire record.

        Parameters
        ----------
        seg_numbers : list
            List of segment numbers to read.
        channels : list
            The channel indices to read for the whole record. Same one
            specified by user input.
        dir_name : str
            The local directory location of the header file. This parameter
            is ignored if `pn_dir` is set.
        pn_dir : str
            Option used to stream data from Physionet. The Physionet
            database directory from which to find the required record files.
            eg. For record '100' in 'http://physionet.org/content/mitdb'
            pn_dir='mitdb'.

        Returns
        -------
        required_channels : list
            List of lists, containing channel indices to read for each
            desired segment.

        """
        # Fixed layout. All channels are the same.
        if self.layout == 'fixed':
            required_channels = [channels] * len(seg_numbers)
        # Variable layout: figure out channels by matching record names
        else:
            required_channels = []
            # The overall layout signal names
            l_sig_names = self.segments[0].sig_name
            # The wanted signals
            w_sig_names = [l_sig_names[c] for c in channels]

            # For each segment
            for i in range(len(seg_numbers)):
                # Skip empty segments
                if self.seg_name[seg_numbers[i]] == '~':
                    required_channels.append([])
                else:
                    # Get the signal names of the current segment
                    s_sig_names = rdheader(
                        os.path.join(dir_name, self.seg_name[seg_numbers[i]]),
                        pn_dir=pn_dir).sig_name
                    required_channels.append(_get_wanted_channels(
                        w_sig_names, s_sig_names))

        return required_channels


    def _arrange_fields(self, seg_numbers, seg_ranges, channels,
                        sampfrom=0, force_channels=True):
        """
        Arrange/edit object fields to reflect user channel and/or
        signal range inputs. Updates layout specification header if
        necessary.

        Parameters
        ----------
        seg_numbers : list
            List of integer segment numbers read.
        seg_ranges: list
            List of integer pairs, giving the sample ranges for each
            segment number read.
        channels : list
            List of channel numbers specified
        sampfrom : int
            Starting sample read.
        force_channels : bool, optional
            Used when reading multi-segment variable layout records.
            Whether to update the layout specification record to match
            the input `channels` argument, or to omit channels in which
            no read segment contains the signals.

        Returns
        -------
        N/A

        """
        # Update seg_len values for relevant segments
        for i in range(len(seg_numbers)):
            self.seg_len[seg_numbers[i]] = seg_ranges[i][1] - seg_ranges[i][0]

        # Get rid of the segments and segment line parameters
        # outside the desired segment range
        if self.layout == 'fixed':
            self.n_sig = len(channels)
            self.segments = self.segments[seg_numbers[0]:seg_numbers[-1]+1]
            self.seg_name = self.seg_name[seg_numbers[0]:seg_numbers[-1]+1]
            self.seg_len = self.seg_len[seg_numbers[0]:seg_numbers[-1]+1]
        else:
            self.segments = [self.segments[0]] + self.segments[seg_numbers[0]:seg_numbers[-1]+1]
            self.seg_name = [self.seg_name[0]] + self.seg_name[seg_numbers[0]:seg_numbers[-1]+1]
            self.seg_len = [self.seg_len[0]] + self.seg_len[seg_numbers[0]:seg_numbers[-1]+1]

            # Update the layout specification segment. At this point it
            # should match the full original header

            # Have to inspect existing channels of segments; requested
            # input channels will not be enough on its own because not
            # all signals may be present, depending on which section of
            # the signal was read.
            if not force_channels:
                # The desired signal names.
                desired_sig_names = [self.segments[0].sig_name[ch] for ch in channels]
                # Actual contained signal names of individual segments
                #contained_sig_names = [seg.sig_name for seg in self.segments[1:]]
                contained_sig_names = set([name for seg in self.segments[1:] if seg is not None for name in seg.sig_name])
                # Remove non-present names. Keep the order.
                sig_name = [name for name in desired_sig_names if name in contained_sig_names]
                # Channel indices to keep for signal specification fields
                channels = [self.segments[0].sig_name.index(name) for name in sig_name]

            # Rearrange signal specification fields
            for field in _header.SIGNAL_SPECS.index:
                item = getattr(self.segments[0], field)
                setattr(self.segments[0], field, [item[c] for c in channels])

            self.segments[0].n_sig = self.n_sig = len(channels)
            if self.n_sig == 0:
                print('No signals of the desired channels are contained in the specified sample range.')

        # Update record specification parameters
        self.sig_len = sum([sr[1]-sr[0] for sr in seg_ranges])
        self.n_seg = len(self.segments)
        self._adjust_datetime(sampfrom=sampfrom)


    def multi_to_single(self, physical, return_res=64):
        """
        Create a Record object from the MultiRecord object. All signal
        segments will be combined into the new object's `p_signal` or
        `d_signal` field. For digital format, the signals must have
        the same storage format, baseline, and adc_gain in all segments.

        Parameters
        ----------
        physical : bool
            Whether to convert the physical or digital signal.
        return_res : int, optional
            The return resolution of the `p_signal` field. Options are:
            64, 32, and 16.

        Returns
        -------
        record : WFDB Record
            The single segment record created.

        """
        # The fields to transfer to the new object
        fields = self.__dict__.copy()

        # Remove multirecord fields
        for attr in ['segments', 'seg_name', 'seg_len', 'n_seg']:
            del(fields[attr])

        # Figure out single segment fields to set for the new Record
        if self.layout == 'fixed':
            # Get the fields from the first segment
            for attr in ['fmt', 'adc_gain', 'baseline', 'units', 'sig_name']:
                fields[attr] = getattr(self.segments[0], attr)
        else:
            # For variable layout records, inspect the segments for the
            # attribute values.

            # Coincidentally, if physical=False, figure out if this
            # conversion can be performed. All signals of the same name
            # must have the same fmt, gain, baseline, and units for all
            # segments.

            # The layout header should be updated at this point to
            # reflect channels. We can depend on it for sig_name, but
            # not for fmt, adc_gain, units, and baseline.

            # These signal names will be the key
            signal_names = self.segments[0].sig_name
            n_sig = len(signal_names)

            # This will be the field dictionary to copy over.
            reference_fields = {'fmt':n_sig*[None], 'adc_gain':n_sig*[None],
                                'baseline':n_sig*[None],
                                'units':n_sig*[None]}

            # For physical signals, mismatched fields will not be copied
            # over. For digital, mismatches will cause an exception.
            mismatched_fields = []
            for seg in self.segments[1:]:
                if seg is None:
                    continue
                # For each signal, check fmt, adc_gain, baseline, and
                # units of each signal
                for seg_ch in range(seg.n_sig):
                    sig_name = seg.sig_name[seg_ch]
                    # The overall channel
                    ch = signal_names.index(sig_name)

                    for field in reference_fields:
                        item_ch = getattr(seg, field)[seg_ch]
                        if reference_fields[field][ch] is None:
                            reference_fields[field][ch] = item_ch
                        # mismatch case
                        elif reference_fields[field][ch] != item_ch:
                            if physical:
                                mismatched_fields.append(field)
                            else:
                                raise Exception('This variable layout multi-segment record cannot be converted to single segment, in digital format.')
            # Remove mismatched signal fields for physical signals
            for field in set(mismatched_fields):
                del(reference_fields[field])
            # At this point, the fields should be set for all channels
            fields.update(reference_fields)
            fields['sig_name'] = signal_names

        # Figure out signal attribute to set, and its dtype.
        if physical:
            sig_attr = 'p_signal'
            # Figure out the largest required dtype
            dtype = _signal._np_dtype(return_res, discrete=False)
            nan_vals = np.array([self.n_sig * [np.nan]], dtype=dtype)
        else:
            sig_attr = 'd_signal'
            # Figure out the largest required dtype
            dtype = _signal._np_dtype(return_res, discrete=True)
            nan_vals = np.array([_signal._digi_nan(fields['fmt'])], dtype=dtype)

        # Initialize the full signal array
        combined_signal = np.repeat(nan_vals, self.sig_len, axis=0)

        # Start and end samples in the overall array to place the
        # segment samples into
        start_samps = [0] + list(np.cumsum(self.seg_len)[0:-1])
        end_samps = list(np.cumsum(self.seg_len))

        if self.layout == 'fixed':
            # Copy over the signals directly. Recall there are no
            # empty segments in fixed layout records.
            for i in range(self.n_seg):
                combined_signal[start_samps[i]:end_samps[i], :] = getattr(self.segments[i], sig_attr)
        else:
            # Copy over the signals into the matching channels
            for i in range(1, self.n_seg):
                seg = self.segments[i]
                if seg is not None:
                    # Get the segment channels to copy over for each
                    # overall channel
                    segment_channels = _get_wanted_channels(fields['sig_name'],
                                                           seg.sig_name,
                                                           pad=True)
                    for ch in range(self.n_sig):
                        # Copy over relevant signal
                        if segment_channels[ch] is not None:
                            combined_signal[start_samps[i]:end_samps[i], ch] = getattr(seg, sig_attr)[:, segment_channels[ch]]

        # Create the single segment Record object and set attributes
        record = Record()
        for field in fields:
            setattr(record, field, fields[field])
        setattr(record, sig_attr, combined_signal)

        # Use the signal to set record features
        if physical:
            record.set_p_features()
        else:
            record.set_d_features()

        return record


# ---------------------- Type Specifications ------------------------- #


# Allowed types of WFDB header fields, and also attributes defined in
# this library
ALLOWED_TYPES = dict([[index, _header.FIELD_SPECS.loc[index, 'allowed_types']] for index in _header.FIELD_SPECS.index])
ALLOWED_TYPES.update({'comments': (str,), 'p_signal': (np.ndarray,),
                      'd_signal':(np.ndarray,), 'e_p_signal':(np.ndarray,),
                      'e_d_signal':(np.ndarray,),
                      'segments':(Record, type(None))})

# Fields that must be lists
LIST_FIELDS = tuple(_header.SIGNAL_SPECS.index) + ('comments', 'e_p_signal',
                                                   'e_d_signal', 'segments')

def get_version(pn_dir):
    """
    Get the version number of the desired project.

    Parameters
    ----------
    pn_dir : str
        The PhysioNet database directory from which to find the
        required version number. eg. For the project 'mitdb' in
        'http://physionet.org/content/mitdb', pn_dir='mitdb'.

    Returns
    -------
    version_number : str
        The version number of the most recent database.

    """
    db_dir = pn_dir.split('/')[0]
    url = posixpath.join(download.PN_CONTENT_URL, db_dir)
    response = requests.get(url)
    contents = [line.decode('utf-8').strip() for line in response.content.splitlines()]
    version_number = [v for v in contents if 'Version:' in v]
    version_number = version_number[0].split(':')[-1].strip().split('<')[0]

    return version_number


def _check_item_type(item, field_name, allowed_types, expect_list=False,
                    required_channels='all'):
    """
    Check the item's type against a set of allowed types.
    Vary the print message regarding whether the item can be None.
    Helper to `BaseRecord.check_field`.

    Parameters
    ----------
    item : any
        The item to check.
    field_name : str
        The field name.
    allowed_types : iterable
        Iterable of types the item is allowed to be.
    expect_list : bool, optional
        Whether the item is expected to be a list.
    required_channels : list, optional
        List of integers specifying which channels of the item must be
        present. May be set to 'all' to indicate all channels. Only used
        if `expect_list` is True, ie. item is a list, and its
        subelements are to be checked.

    Returns
    -------
    N/A

    Notes
    -----
    This is called by `check_field`, which determines whether the item
    should be a list or not. This function should generally not be
    called by the user directly.

    """
    if expect_list:
        if not isinstance(item, list):
            raise TypeError('Field `%s` must be a list.' % field_name)

        # All channels of the field must be present.
        if required_channels == 'all':
            required_channels = list(range(len(item)))

        for ch in range(len(item)):
            # Check whether the field may be None
            if ch in required_channels:
                allowed_types_ch = allowed_types
            else:
                allowed_types_ch = allowed_types + (type(None),)

            if not isinstance(item[ch], allowed_types_ch):
                raise TypeError('Channel %d of field `%s` must be one of the following types:' % (ch, field_name),
                                allowed_types_ch)
    else:
        if not isinstance(item, allowed_types):
            raise TypeError('Field `%s` must be one of the following types:',
                            allowed_types)


def check_np_array(item, field_name, ndim, parent_class, channel_num=None):
    """
    Check a numpy array's shape and dtype against required
    specifications.

    Parameters
    ----------
    item : ndarray
        The numpy array to check
    field_name : str
        The name of the field to check
    ndim : int
        The required number of dimensions
    parent_class : type
        The parent class of the dtype. ie. np.integer, np.floating.
    channel_num : int, optional
        If not None, indicates that the item passed in is a subelement
        of a list. Indicate this in the error message if triggered.

    Returns
    -------
    N/A

    """
    # Check shape
    if item.ndim != ndim:
        error_msg = 'Field `%s` must have ndim == %d' % (field_name, ndim)
        if channel_num is not None:
            error_msg = ('Channel %d of f' % channel_num) + error_msg[1:]
        raise TypeError(error_msg)

    # Check dtype
    if not np.issubdtype(item.dtype, parent_class):
        error_msg = 'Field `%s` must have a dtype that subclasses %s' % (field_name, parent_class)
        if channel_num is not None:
            error_msg = ('Channel %d of f' % channel_num) + error_msg[1:]
        raise TypeError(error_msg)


def edf2mit(record_name, pn_dir=None, delete_file=True, record_only=True,
            header_only=False, verbose=False):
    """
    Convert EDF formatted files to MIT format.

    Many EDF files contain signals at widely varying sampling frequencies.
    `edf2mit` handles these properly, but the default behavior of most WFDB
    applications is to read such data in low-resolution mode (in which all
    signals are resampled at the lowest sampling frequency used for any signal
    in the record). This is almost certainly not what you want if, for
    example, the record contains EEG signals sampled at 200 Hz and body
    temperature sampled at 1 Hz; by default, applications such as `rdsamp`
    will resample the EEGs (and any other signals in the record) at 1 Hz. To
    avoid this behavior, you can set `smooth_frames` to False (high resolution)
    provided by `rdrecord` and a few other WFDB applications.

    Note that applications built using version 3.1.0 and later versions of
    the WFDB-Python library can read EDF files directly, so that the conversion
    performed by `edf2mit` is no longer necessary. However, one can still use
    this function to produce WFDB-compatible files from EDF files if desired.

    Parameters
    ----------
    record_name : str
        The name of the input EDF record to be read.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    delete_file : bool, optional
        Whether to delete the saved EDF file (False) or not (True)
        after being imported.
    record_only : bool, optional
        Whether to only return the record information (True) or not (False).
        If false, this function will generate both a .dat and .hea file.
    header_only : bool, optional
        Whether to only return the header information (True) or not (False).
        If true, this function will only return `['fs', 'sig_len', 'n_sig',
        'base_date', 'base_time', 'units', 'sig_name', 'comments']`.
    verbose : bool, optional
        Whether to print all the information read about the file (True) or
        not (False).

    Returns
    -------
    record : dict, optional
        All of the record information needed to generate MIT formatted files.
        Only returns if 'record_only' is set to True, else generates the
        corresponding .dat and .hea files. This record file will not match the
        `rdrecord` output since it will only give us the digital signal for now.

    Notes
    -----
    The entire file is composed of (seen here: https://www.edfplus.info/specs/edf.html):

    HEADER RECORD (we suggest to also adopt the 12 simple additional EDF+ specs)
    8 ascii : version of this data format (0)
    80 ascii : local patient identification (mind item 3 of the additional EDF+ specs)
    80 ascii : local recording identification (mind item 4 of the additional EDF+ specs)
    8 ascii : startdate of recording (dd.mm.yy) (mind item 2 of the additional EDF+ specs)
    8 ascii : starttime of recording (hh.mm.ss)
    8 ascii : number of bytes in header record
    44 ascii : reserved
    8 ascii : number of data records (-1 if unknown, obey item 10 of the additional EDF+ specs)
    8 ascii : duration of a data record, in seconds
    4 ascii : number of signals (ns) in data record
    ns * 16 ascii : ns * label (e.g. EEG Fpz-Cz or Body temp) (mind item 9 of the additional EDF+ specs)
    ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)
    ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)
    ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)
    ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)
    ns * 8 ascii : ns * digital minimum (e.g. -2048)
    ns * 8 ascii : ns * digital maximum (e.g. 2047)
    ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
    ns * 8 ascii : ns * nr of samples in each data record
    ns * 32 ascii : ns * reserved

    DATA RECORD
    nr of samples[1] * integer : first signal in the data record
    nr of samples[2] * integer : second signal
    ..
    ..
    nr of samples[ns] * integer : last signal

    Bytes   0 - 127: descriptive text
    Bytes 128 - 131: master tag (data type = matrix)
    Bytes 132 - 135: master tag (data size)
    Bytes 136 - 151: array flags (4 byte tag with data type, 4 byte
                     tag with subelement size, 8 bytes of content)
    Bytes 152 - 167: array dimension (4 byte tag with data type, 4
                     byte tag with subelement size, 8 bytes of content)
    Bytes 168 - 183: array name (4 byte tag with data type, 4 byte
                     tag with subelement size, 8 bytes of content)
    Bytes 184 - ...: array content (4 byte tag with data type, 4 byte
                     tag with subelement size, ... bytes of content)

    Examples
    --------
    >>> edf_record = wfdb.edf2mit('x001_FAROS.edf',
                                  pn_dir='simultaneous-measurements/raw_data')

    """
    if pn_dir is not None:

        if '.' not in pn_dir:
            dir_list = pn_dir.split('/')
            pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]), *dir_list[1:])

        file_url = posixpath.join(download.PN_INDEX_URL, pn_dir, record_name)
        # Currently must download file for MNE to read it though can give the
        # user the option to delete it immediately afterwards
        r = requests.get(file_url, allow_redirects=False)
        open(record_name, 'wb').write(r.content)

    # Open the desired file
    edf_file = open(record_name, mode='rb')

    # Remove the file if the `delete_file` flag is set
    if pn_dir is not None and delete_file:
        os.remove(record_name)

    # Version of this data format (8 bytes)
    version = struct.unpack('<8s', edf_file.read(8))[0].decode()

    # Check to see that the input is an EDF file. (This check will detect
    # most but not all other types of files.)
    if version != '0       ':
        raise Exception('Input does not appear to be EDF -- no conversion attempted')
    else:
        if verbose:
            print('EDF version number: {}'.format(version.strip()))

    # Local patient identification (80 bytes)
    patient_id = struct.unpack('<80s', edf_file.read(80))[0].decode()
    if verbose:
        print('Patient ID: {}'.format(patient_id))

    # Local recording identification (80 bytes)
    # Bob Kemp recommends using this field to encode the start date
    # including an abbreviated month name in English and a full (4-digit)
    # year, as is done here if this information is available in the input
    # record. EDF+ requires this.
    record_id = struct.unpack('<80s', edf_file.read(80))[0].decode()
    if verbose:
        print('Recording ID: {}'.format(record_id))

    # Start date of recording (dd.mm.yy) (8 bytes)
    start_date = struct.unpack('<8s', edf_file.read(8))[0].decode()
    if verbose:
        print('Recording Date: {}'.format(start_date))
    start_day, start_month, start_year = [int(i) for i in start_date.split('.')]
    # This should work for a while
    if start_year < 1970:
        start_year += 1900
    if start_year < 1970:
        start_year += 100

    # Start time of recording (hh.mm.ss) (8 bytes)
    start_time = struct.unpack('<8s', edf_file.read(8))[0].decode()
    if verbose:
        print('Recording Time: {}'.format(start_time))
    start_hour, start_minute, start_second = [int(i) for i in start_time.split('.')]

    # Number of bytes in header (8 bytes)
    header_bytes = int(struct.unpack('<8s', edf_file.read(8))[0].decode())
    if verbose:
        print('Number of bytes in header record: {}'.format(header_bytes))

    # Reserved (44 bytes)
    reserved_notes = struct.unpack('<44s', edf_file.read(44))[0].decode().strip()
    if reserved_notes == 'EDF+C':
        raise Exception('EDF+ File: uninterrupted data records (not currently supported)')
    elif reserved_notes == 'EDF+D':
        raise Exception('EDF+ File: interrupted data records (not currently supported)')
    else:
        if verbose:
            print('Free Space: {}'.format(reserved_notes))

    # Number of blocks (-1 if unknown) (8 bytes)
    num_blocks = int(struct.unpack('<8s', edf_file.read(8))[0].decode())
    if verbose:
        print('Number of data records: {}'.format(num_blocks))
    if num_blocks == -1:
        raise Exception('Number of data records in unknown (not currently supported)')

    # Duration of a block, in seconds (8 bytes)
    block_duration = float(struct.unpack('<8s', edf_file.read(8))[0].decode())
    if verbose:
        print('Duration of each data record in seconds: {}'.format(block_duration))
    if block_duration <= 0.0:
        block_duration = 1.0

    # Number of signals (4 bytes)
    n_sig = int(struct.unpack('<4s', edf_file.read(4))[0].decode())
    if verbose:
        print('Number of signals: {}'.format(n_sig))
    if n_sig < 1:
        raise Exception('Done: not any signals left to read')

    # Label (e.g., EEG FpzCz or Body temp) (16 bytes each)
    sig_name = []
    for _ in range(n_sig):
        sig_name.append(struct.unpack('<16s', edf_file.read(16))[0].decode().strip())
    if verbose:
        print('Signal Labels: {}'.format(sig_name))

    # Transducer type (e.g., AgAgCl electrode) (80 bytes each)
    transducer_types = []
    for _ in range(n_sig):
        transducer_types.append(struct.unpack('<80s', edf_file.read(80))[0].decode().strip())
    if verbose:
        print('Transducer Types: {}'.format(transducer_types))

    # Physical dimension (e.g., uV or degreeC) (8 bytes each)
    physical_dims = []
    for _ in range(n_sig):
        physical_dims.append(struct.unpack('<8s', edf_file.read(8))[0].decode().strip())
    if verbose:
        print('Physical Dimensions: {}'.format(physical_dims))

    # Physical minimum (e.g., -500 or 34) (8 bytes each)
    physical_min = np.array([])
    for _ in range(n_sig):
        physical_min = np.append(physical_min, float(struct.unpack('<8s', edf_file.read(8))[0].decode()))
    if verbose:
        print('Physical Minimums: {}'.format(physical_min))

    # Physical maximum (e.g., 500 or 40) (8 bytes each)
    physical_max = np.array([])
    for _ in range(n_sig):
        physical_max = np.append(physical_max, float(struct.unpack('<8s', edf_file.read(8))[0].decode()))
    if verbose:
        print('Physical Maximums: {}'.format(physical_max))

    # Digital minimum (e.g., -2048) (8 bytes each)
    digital_min = np.array([])
    for _ in range(n_sig):
        digital_min = np.append(digital_min, float(struct.unpack('<8s', edf_file.read(8))[0].decode()))
    if verbose:
        print('Digital Minimums: {}'.format(digital_min))

    # Digital maximum (e.g., 2047) (8 bytes each)
    digital_max = np.array([])
    for _ in range(n_sig):
        digital_max = np.append(digital_max, float(struct.unpack('<8s', edf_file.read(8))[0].decode()))
    if verbose:
        print('Digital Maximums: {}'.format(digital_max))

    # Prefiltering (e.g., HP:0.1Hz LP:75Hz) (80 bytes each)
    prefilter_info = []
    for _ in range(n_sig):
        prefilter_info.append(struct.unpack('<80s', edf_file.read(80))[0].decode().strip())
    if verbose:
        print('Prefiltering Information: {}'.format(prefilter_info))

    # Number of samples per block (8 bytes each)
    samps_per_block = []
    for _ in range(n_sig):
        samps_per_block.append(int(struct.unpack('<8s', edf_file.read(8))[0].decode()))
    if verbose:
        print('Number of Samples per Record: {}'.format(samps_per_block))

    # The last 32*nsig bytes in the header are unused
    for _ in range(n_sig):
        struct.unpack('<32s', edf_file.read(32))[0].decode()

    # Pre-process the acquired data before creating the record
    record_name_out = record_name.split(os.sep)[-1].replace('-','_').replace('.edf','')
    sample_rate = [int(i/block_duration) for i in samps_per_block]
    fs = functools.reduce(math.gcd, sample_rate)
    samps_per_frame = [int(s/min(samps_per_block)) for s in samps_per_block]
    sig_len = int(fs * num_blocks * block_duration)
    base_time = datetime.time(start_hour, start_minute, start_second)
    base_date = datetime.date(start_year, start_month, start_day)
    file_name = n_sig * [record_name_out + '.dat']
    fmt = n_sig * ['16']
    skew = n_sig * [None]
    byte_offset = n_sig * [None]
    adc_gain_all = (digital_max - digital_min) / (physical_max - physical_min)
    adc_gain =  [float(format(a,'.12g')) for a in adc_gain_all]
    baseline = (digital_max - (physical_max * adc_gain_all) + 1).astype('int64')

    units = n_sig * ['']
    for i,f in enumerate(physical_dims):
        if f == 'n/a':
            label = sig_name[i].lower().split()[0]
            if label in list(SIG_UNITS.keys()):
                units[i] = SIG_UNITS[label]
            else:
                units[i] = 'n/a'
        else:
            f = f.replace('µ','u')  # Maybe more weird symbols to check for?
            if f == '':
                units[i] = 'mV'
            else:
                units[i] = f

    adc_res = [int(math.log2(f)) for f in (digital_max - digital_min)]
    adc_zero = [int(f) for f in ((digital_max + 1 + digital_min) / 2)]
    block_size = n_sig * [0]
    base_datetime = datetime.datetime(start_year, start_month, start_day,
                                      start_hour, start_minute, start_second)

    sig_data = np.empty((sig_len, n_sig))
    temp_sig_data = np.fromfile(edf_file, dtype=np.int16)
    temp_sig_data = temp_sig_data.reshape((-1,sum(samps_per_block)))
    temp_all_sigs = np.hsplit(temp_sig_data, np.cumsum(samps_per_block)[:-1])
    for i in range(n_sig):
        # Check if `samps_per_frame` has all equal values
        if samps_per_frame.count(samps_per_frame[0]) == len(samps_per_frame):
            sig_data[:,i] = (temp_all_sigs[i].flatten() - baseline[i]) / adc_gain_all[i]
        else:
            temp_sig_data = temp_all_sigs[i].flatten()
            if samps_per_frame[i] == 1:
                sig_data[:,i] = (temp_sig_data - baseline[i]) / adc_gain_all[i]
            else:
                for j in range(sig_len):
                    start_ind = j * samps_per_frame[i]
                    stop_ind = start_ind + samps_per_frame[i]
                    sig_data[j,i] = np.mean((temp_sig_data[start_ind:stop_ind] - baseline[i]) / adc_gain_all[i])

    # This is the closest I can get to the original implementation
    # NOTE: This is done using `np.testing.assert_array_equal()`
    # Mismatched elements: 15085545 / 15400000 (98%)
    # Max absolute difference: 3.75166564e-12
    # Max relative difference: 5.41846079e-15
    #  x: array([[  -3.580728,   42.835293, -102.818048,   54.978632,  -52.354247],
    #        [  -8.340205,   43.079939, -102.106351,   56.402027,  -44.992626],
    #        [  -5.004123,   43.546991,  -99.481966,   51.64255 ,  -43.079939],...
    #  y: array([[  -3.580728,   42.835293, -102.818048,   54.978632,  -52.354247],
    #        [  -8.340205,   43.079939, -102.106351,   56.402027,  -44.992626],
    #        [  -5.004123,   43.546991,  -99.481966,   51.64255 ,  -43.079939],...

    init_value = [int(s[0,0]) for s in temp_all_sigs]
    checksum = [int(np.sum(v) % 65536) for v in np.transpose(sig_data)]  # not all values correct?

    record = Record(
        record_name = record_name_out,
        n_sig = n_sig,
        fs = fs,
        samps_per_frame = samps_per_frame,
        counter_freq = None,
        base_counter = None,
        sig_len = sig_len,
        base_time = datetime.time(base_datetime.hour,
                                  base_datetime.minute,
                                  base_datetime.second),
        base_date = datetime.date(base_datetime.year,
                                  base_datetime.month,
                                  base_datetime.day),
        comments = [],
        sig_name = sig_name,    # Remove whitespace to make compatible later?
        p_signal = sig_data,
        d_signal = None,
        e_p_signal = None,
        e_d_signal = None,
        file_name = n_sig * [record_name_out + '.dat'],
        fmt = n_sig * ['16'],
        skew = n_sig * [None],
        byte_offset = n_sig * [None],
        adc_gain = adc_gain,
        baseline = baseline,
        units = units,
        adc_res =  [int(math.log2(f)) for f in (digital_max - digital_min)],
        adc_zero = [int(f) for f in ((digital_max + 1 + digital_min) / 2)],
        init_value = init_value,
        checksum = checksum,
        block_size = n_sig * [0]
    )

    record.base_datetime = base_datetime

    if record_only:
        return record
    else:
        # TODO: Generate the .dat and .hea files
        pass


def mit2edf(record_name, pn_dir=None, sampfrom=0, sampto=None, channels=None,
             output_filename='', edf_plus=False):
    """
    These programs convert EDF (European Data Format) files into
    WFDB-compatible files (as used in PhysioNet) and vice versa. European
    Data Format (EDF) was originally designed for storage of polysomnograms.

    Note that WFDB format does not include a standard way to specify the
    transducer type or the prefiltering specification; these parameters are
    not preserved by these conversion programs. Also note that use of the
    standard signal and unit names specified for EDF is permitted but not
    enforced by `mit2edf`.

    Parameters
    ----------
    record_name : str
        The name of the input WFDB record to be read. Can also work with both
        EDF and WAV files.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    sampfrom : int, optional
        The starting sample number to read for all channels.
    sampto : int, 'end', optional
        The sample number at which to stop reading for all channels.
        Reads the entire duration by default.
    channels : list, optional
        List of integer indices specifying the channels to be read.
        Reads all channels by default.
    output_filename : str, optional
        The desired name of the output file. If this value set to the
        default value of '', then the output filename will be 'REC.edf'.
    edf_plus : bool, optional
        Whether to write the output file in EDF (False) or EDF+ (True) format.

    Returns
    -------
    N/A

    Notes
    -----
    The entire file is composed of (seen here: https://www.edfplus.info/specs/edf.html):

    HEADER RECORD (we suggest to also adopt the 12 simple additional EDF+ specs)
    8 ascii : version of this data format (0)
    80 ascii : local patient identification (mind item 3 of the additional EDF+ specs)
    80 ascii : local recording identification (mind item 4 of the additional EDF+ specs)
    8 ascii : startdate of recording (dd.mm.yy) (mind item 2 of the additional EDF+ specs)
    8 ascii : starttime of recording (hh.mm.ss)
    8 ascii : number of bytes in header record
    44 ascii : reserved
    8 ascii : number of data records (-1 if unknown, obey item 10 of the additional EDF+ specs)
    8 ascii : duration of a data record, in seconds
    4 ascii : number of signals (ns) in data record
    ns * 16 ascii : ns * label (e.g. EEG Fpz-Cz or Body temp) (mind item 9 of the additional EDF+ specs)
    ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)
    ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)
    ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)
    ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)
    ns * 8 ascii : ns * digital minimum (e.g. -2048)
    ns * 8 ascii : ns * digital maximum (e.g. 2047)
    ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
    ns * 8 ascii : ns * nr of samples in each data record
    ns * 32 ascii : ns * reserved

    DATA RECORD
    nr of samples[1] * integer : first signal in the data record
    nr of samples[2] * integer : second signal
    ..
    ..
    nr of samples[ns] * integer : last signal

    Bytes   0 - 127: descriptive text
    Bytes 128 - 131: master tag (data type = matrix)
    Bytes 132 - 135: master tag (data size)
    Bytes 136 - 151: array flags (4 byte tag with data type, 4 byte
                     tag with subelement size, 8 bytes of content)
    Bytes 152 - 167: array dimension (4 byte tag with data type, 4
                     byte tag with subelement size, 8 bytes of content)
    Bytes 168 - 183: array name (4 byte tag with data type, 4 byte
                     tag with subelement size, 8 bytes of content)
    Bytes 184 - ...: array content (4 byte tag with data type, 4 byte
                     tag with subelement size, ... bytes of content)

    Examples
    --------
    >>> wfdb.mit2edf('100', pn_dir='pwave')

    The output file name is '100.edf'

    """
    record = rdrecord(record_name, pn_dir=pn_dir, sampfrom=sampfrom,
                      sampto=sampto, smooth_frames=False)
    record_name_out = record_name.split(os.sep)[-1].replace('-','_')

    # Maximum data block length, in bytes
    edf_max_block = 61440

    # Convert to the expected month name formatting
    month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
                   'SEP', 'OCT', 'NOV', 'DEC']

    # Calculate block duration. (In the EDF spec, blocks are called "records"
    # or "data records", but this would be confusing here since "record"
    # refers to the entire recording -- so here we say "blocks".)
    samples_per_frame = sum(record.samps_per_frame)

    # i.e., The number of frames per minute, divided by 60
    frames_per_minute = record.fs * 60 + 0.5
    frames_per_second = frames_per_minute / 60

    # Ten seconds
    frames_per_block = 10 * frames_per_second + 0.5

    # EDF specifies 2 bytes per sample
    bytes_per_block = int(2 * samples_per_frame * frames_per_block)

    # Blocks would be too long -- reduce their length by a factor of 10
    while (bytes_per_block > edf_max_block):
        frames_per_block /= 10
        bytes_per_block = samples_per_frame * 2 * frames_per_block

    seconds_per_block = int(frames_per_block / frames_per_second)

    if (frames_per_block < 1) and (bytes_per_block < edf_max_block/60):
        # The number of frames/minute
        frames_per_block = frames_per_minute
        bytes_per_block = 2 * samples_per_frame * frames_per_block
        seconds_per_block = 60

    if (bytes_per_block > edf_max_block):
        print(("Can't convert record %s to EDF: EDF blocks can't be larger "
               "than {} bytes, but each input frame requires {} bytes.  Use "
               "'channels' to select a subset of the input signals or trim "
               "using 'sampfrom' and 'sampto'.").format(edf_max_block,
                                                        samples_per_frame * 2))

    # Calculate the number of blocks to be written. The calculation rounds
    # up so that we don't lose any frames, even if the number of frames is not
    # an exact multiple of frames_per_block
    total_frames = record.sig_len
    num_blocks = int(total_frames / int(frames_per_block)) + 1

    digital_min = []
    digital_max = []
    physical_min = []
    physical_max = []
    # Calculate the physical and digital extrema
    for i in range(record.n_sig):
        # Invalid ADC resolution in input .hea file
        if (record.adc_res[i] < 1):
            # Guess the ADC resolution based on format
            if record.fmt[i] == '24':
                temp_adc_res = 24
            elif record.fmt[i] == '32':
                temp_adc_res = 32
            elif record.fmt[i] == '80':
                temp_adc_res = 8
            elif record.fmt[i] == '212':
                temp_adc_res = 12
            elif (record.fmt[i] == '310') or (record.fmt[i] == '311'):
                temp_adc_res = 10
            else:
                temp_adc_res = 16
        else:
            temp_adc_res = record.adc_res[i]
        # Determine the physical and digital extrema
        digital_max.append(int(record.adc_zero[i] + (1 << (temp_adc_res - 1)) - 1))
        digital_min.append(int(record.adc_zero[i] - (1 << (temp_adc_res - 1))))
        physical_max.append((digital_max[i] - record.baseline[i]) / record.adc_gain[i])
        physical_min.append((digital_min[i] - record.baseline[i]) / record.adc_gain[i])

    # The maximum record name length to write is 80 bytes
    if len(record_name_out) > 80:
        record_name_write = record_name_out[:79] + '\0'
    else:
        record_name_write = record_name_out

    # The maximum seconds per block length to write is 8 bytes
    if len(str(seconds_per_block)) > 8:
        seconds_per_block_write = seconds_per_block[:7] + '\0'
    else:
        seconds_per_block_write = seconds_per_block

    # The maximum signal name length to write is 16 bytes
    sig_name_write = len(record.sig_name) * []
    for s in record.sig_name:
        if len(s) > 16:
            sig_name_write.append(s[:15] + '\0')
        else:
            sig_name_write.append(s)

    # The maximum units length to write is 8 bytes
    units_write = len(record.units) * []
    for s in record.units:
        if len(s) > 8:
            units_write.append(s[:7] + '\0')
        else:
            units_write.append(s)

    # Configure the output datetime
    if hasattr('record', 'base_datetime'):
        start_second = int(record.base_datetime.second)
        start_minute = int(record.base_datetime.minute)
        start_hour = int(record.base_datetime.hour)
        start_day = int(record.base_datetime.day)
        start_month = int(record.base_datetime.month)
        start_year = int(record.base_datetime.year)
    else:
        # Set date to start of EDF epoch
        start_second = 0
        start_minute = 0
        start_hour = 0
        start_day = 1
        start_month = 1
        start_year = 1985

    # Determine the number of bytes in the header
    header_bytes = 256 * (record.n_sig + 1)

    # Determine the number of samples per data record
    samps_per_record = []
    for spf in record.samps_per_frame:
        samps_per_record.append(int(frames_per_block) * spf)

    # Determine the output data
    # NOTE: The output data will be close (+-1) but not equal due to the
    #       inappropriate rounding done by record.adc()
    #       For example...
    #           Mismatched elements: 862881 / 24168000 (3.57%)
    #           Max absolute difference: 1
    #           Max relative difference: 0.0212766
    #            x: array([ 53, -28,  14, ..., 884, 898, 898], dtype=int16)
    #            y: array([ 53, -28,  14, ..., 884, 898, 898], dtype=int16)
    if record.e_p_signal is not None:
        temp_data = record.adc(expanded=True)
    else:
        temp_data = record.adc()
        temp_data = [v for v in np.transpose(temp_data)]

    out_data = []
    for i in range(record.sig_len):
        for j,sig in enumerate(temp_data):
            ind_start = i * samps_per_record[j]
            ind_stop = (i+1) * samps_per_record[j]
            out_data.extend(sig[ind_start:ind_stop].tolist())
    out_data = np.array(out_data, dtype=np.int16)

    # Start writing the file
    if output_filename == '':
        output_filename = record_name_out + '.edf'

    with open(output_filename, 'wb') as f:

        print('Converting record {} to {} ({} mode)'.format(record_name, output_filename, 'EDF+' if edf_plus else 'EDF'))

        # Version of this data format (8 bytes)
        f.write(struct.pack('<8s', b'0').replace(b'\x00',b'\x20'))

        # Local patient identification (80 bytes)
        f.write(struct.pack('<80s', '{}'.format(record_name_write).encode('ascii')).replace(b'\x00',b'\x20'))

        # Local recording identification (80 bytes)
        # Bob Kemp recommends using this field to encode the start date
        # including an abbreviated month name in English and a full (4-digit)
        # year, as is done here if this information is available in the input
        # record. EDF+ requires this.
        if hasattr('record', 'base_datetime'):
            f.write(struct.pack('<80s', 'Startdate {}-{}-{}'.format(start_day, month_names[start_month-1], start_year).encode('ascii')).replace(b'\x00',b'\x20'))
        else:
            f.write(struct.pack('<80s', b'Startdate not recorded').replace(b'\x00',b'\x20'))
        if edf_plus:
            print('WARNING: EDF+ requires start date (not specified)')

        # Start date of recording (dd.mm.yy) (8 bytes)
        f.write(struct.pack('<8s', '{:02d}.{:02d}.{:02d}'.format(start_day, start_month, start_year%100).encode('ascii')).replace(b'\x00',b'\x20'))

        # Start time of recording (hh.mm.ss) (8 bytes)
        f.write(struct.pack('<8s', '{:02d}.{:02d}.{:02d}'.format(start_hour, start_minute, start_second).encode('ascii')).replace(b'\x00',b'\x20'))

        # Number of bytes in header (8 bytes)
        f.write(struct.pack('<8s', '{:d}'.format(header_bytes).encode('ascii')).replace(b'\x00',b'\x20'))

        # Reserved (44 bytes)
        if edf_plus:
            f.write(struct.pack('<44s', b'EDF+C').replace(b'\x00',b'\x20'))
        else:
            f.write(struct.pack('<44s', b'').replace(b'\x00',b'\x20'))

        # Number of blocks (-1 if unknown) (8 bytes)
        f.write(struct.pack('<8s', '{:d}'.format(num_blocks).encode('ascii')).replace(b'\x00',b'\x20'))

        # Duration of a block, in seconds (8 bytes)
        f.write(struct.pack('<8s', '{:g}'.format(seconds_per_block_write).encode('ascii')).replace(b'\x00',b'\x20'))

        # Number of signals (4 bytes)
        f.write(struct.pack('<4s', '{:d}'.format(record.n_sig).encode('ascii')).replace(b'\x00',b'\x20'))

        # Label (e.g., EEG FpzCz or Body temp) (16 bytes each)
        for i in sig_name_write:
            f.write(struct.pack('<16s', '{}'.format(i).encode('ascii')).replace(b'\x00',b'\x20'))

        # Transducer type (e.g., AgAgCl electrode) (80 bytes each)
        for _ in range(record.n_sig):
            f.write(struct.pack('<80s', b'transducer type not recorded').replace(b'\x00',b'\x20'))

        # Physical dimension (e.g., uV or degreeC) (8 bytes each)
        for i in units_write:
            f.write(struct.pack('<8s', '{}'.format(i).encode('ascii')).replace(b'\x00',b'\x20'))

        # Physical minimum (e.g., -500 or 34) (8 bytes each)
        for pmin in physical_min:
            f.write(struct.pack('<8s', '{:g}'.format(pmin).encode('ascii')).replace(b'\x00',b'\x20'))

        # Physical maximum (e.g., 500 or 40) (8 bytes each)
        for pmax in physical_max:
            f.write(struct.pack('<8s', '{:g}'.format(pmax).encode('ascii')).replace(b'\x00',b'\x20'))

        # Digital minimum (e.g., -2048) (8 bytes each)
        for dmin in digital_min:
            f.write(struct.pack('<8s', '{:d}'.format(dmin).encode('ascii')).replace(b'\x00',b'\x20'))

        # Digital maximum (e.g., 2047) (8 bytes each)
        for dmax in digital_max:
            f.write(struct.pack('<8s', '{:d}'.format(dmax).encode('ascii')).replace(b'\x00',b'\x20'))

        # Prefiltering (e.g., HP:0.1Hz LP:75Hz) (80 bytes each)
        for _ in range(record.n_sig):
            f.write(struct.pack('<80s', b'prefiltering not recorded').replace(b'\x00',b'\x20'))

        # Number of samples per block (8 bytes each)
        for spr in samps_per_record:
            f.write(struct.pack('<8s', '{:d}'.format(spr).encode('ascii')).replace(b'\x00',b'\x20'))

        # The last 32*nsig bytes in the header are unused
        for _ in range(record.n_sig):
            f.write(struct.pack('<32s', b'').replace(b'\x00',b'\x20'))

        # Write the data blocks
        out_data.tofile(f, format='%d')

        # Add the buffer
        correct_bytes = num_blocks * sum(samps_per_record)
        current_bytes = len(out_data)
        num_to_write = correct_bytes - current_bytes
        for i in range(num_to_write):
            f.write(b'\x00\x80')

    print('Header block size: {:d} bytes'.format((record.n_sig+1) * 256))
    print('Data block size: {:g} seconds ({:d} frames or {:d} bytes)'.format(seconds_per_block, int(frames_per_block), int(bytes_per_block)))
    print('Recording length: {:d} ({:d} data blocks, {:d} frames, {:d} bytes)'.format(sum([num_blocks, num_blocks*int(frames_per_block), num_blocks*bytes_per_block]), num_blocks, num_blocks*int(frames_per_block), num_blocks*bytes_per_block))
    print('Total length of file to be written: {:d} bytes'.format(int((record.n_sig+1)*256 + num_blocks*bytes_per_block)))

    if edf_plus:
        print(("WARNING: EDF+ requires the subject's gender, birthdate, and name, as "
               "well as additional information about the recording that is not usually "
               "available. This information is not saved in the output file even if "
               "available. EDF+ also requires the use of standard names for signals and "
               "for physical units;  these requirements are not enforced by this program. "
               "To make the output file fully EDF+ compliant, its header must be edited "
               "manually."))

        if 'EDF-Annotations' not in record.sig_name:
            print('WARNING: The output file does not include EDF annotations, which are required for EDF+.')

    # Check that all characters in the header are valid (printable ASCII
    # between 32 and 126 inclusive). Note that this test does not prevent
    # generation of files containing invalid characters; it merely warns
    # the user if this has happened.
    header_test = open(output_filename,'rb').read((record.n_sig + 1) * 256)
    for i,val in enumerate(header_test):
        if (val < 32) or (val > 126):
            print('WARNING: output contains an invalid character, {}, at byte {}'.format(val, i))


def mit2wav(record_name, pn_dir=None, sampfrom=0, sampto=None, channels=None,
            output_filename='', write_header=False):
    """
    This program converts a WFDB record into .wav format (format 16, multiplexed
    signals, with embedded header information).  Use 'wav2mit' to perform the
    reverse conversion.

    Parameters
    ----------
    record_name : str
        The name of the input WFDB record to be read. Can also work with both
        EDF and WAV files.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    sampfrom : int, optional
        The starting sample number to read for all channels.
    sampto : int, 'end', optional
        The sample number at which to stop reading for all channels.
        Reads the entire duration by default.
    channels : list, optional
        List of integer indices specifying the channels to be read.
        Reads all channels by default.
    output_filename : str, optional
        The desired name of the output file. If this value set to the
        default value of '', then the output filename will be 'REC.wav'.
    write_header : bool, optional
        Whether to write (True) or not to write (False) a header file to
        accompany the generated WAV file. The default value is 'False'.

    Returns
    -------
    N/A

    Notes
    -----
    Files that can be processed successfully using `wav2mit` always have exactly
    three chunks (a header chunk, a format chunk, and a data chunk).  In .wav
    files, binary data are always written in little-endian format (least
    significant byte first). The format of `wav2mit`'s input files is as follows:

    [Header chunk]
    Bytes  0 -  3: "RIFF" [4 ASCII characters]
    Bytes  4 -  7: L-8 (number of bytes to follow in the file, excluding bytes 0-7)
    Bytes  8 - 11: "WAVE" [4 ASCII characters]

    [Format chunk]
    Bytes 12 - 15: "fmt " [4 ASCII characters, note trailing space]
    Bytes 16 - 19: 16 (format chunk length in bytes, excluding bytes 12-19)
    Bytes 20 - 35: format specification, consisting of:
    Bytes 20 - 21: 1 (format tag, indicating no compression is used)
    Bytes 22 - 23: number of signals (1 - 65535)
    Bytes 24 - 27: sampling frequency in Hz (per signal)
                   Note that the sampling frequency in a .wav file must be an
                   integer multiple of 1 Hz, a restriction that is not imposed
                   by MIT (WFDB) format.
    Bytes 28 - 31: bytes per second (sampling frequency * frame size in bytes)
    Bytes 32 - 33: frame size in bytes
    Bytes 34 - 35: bits per sample (ADC resolution in bits)
                   Note that the actual ADC resolution (e.g., 12) is written in
                   this field, although each output sample is right-padded to fill
                   a full (16-bit) word. (.wav format allows for 8, 16, 24, and
                   32 bits per sample)

    [Data chunk]
    Bytes 36 - 39: "data" [4 ASCII characters]
    Bytes 40 - 43: L-44 (number of bytes to follow in the data chunk)
    Bytes 44 - L-1: sample data, consisting of:
    Bytes 44 - 45: sample 0, channel 0
    Bytes 46 - 47: sample 0, channel 1
    ... etc. (same order as in a multiplexed WFDB signal file)

    Examples
    --------
    >>> wfdb.mit2wav('100', pn_dir='pwave')

    The output file name is '100.wav'

    """
    record = rdrecord(record_name, pn_dir=pn_dir, sampfrom=sampfrom,
                      sampto=sampto, smooth_frames=False)
    record_name_out = record_name.split(os.sep)[-1].replace('-','_')

    # Get information needed for the header and format chunks
    num_samps = record.sig_len
    samps_per_second = record.fs
    frame_length = record.n_sig * 2
    chunk_bytes = num_samps * frame_length
    file_bytes = chunk_bytes + 36
    bits_per_sample = max(record.adc_res)
    offset = record.adc_zero
    shift = [(16 - v) for v in record.adc_res]

    # Start writing the file
    if output_filename != '':
        if not output_filename.endswith('.wav'):
            raise Exception("Name of output file must end in '.wav'")
    else:
        output_filename = record_name_out + '.wav'

    with open(output_filename, 'wb') as f:
        # Write the WAV file identifier
        f.write(struct.pack('>4s', b'RIFF'))
        # Write the number of bytes to follow in the file
        # (num_samps*frame_length) sample bytes, and 36 more bytes of miscellaneous embedded header
        f.write(struct.pack('<I', file_bytes))
        # Descriptor for the format of the file
        f.write(struct.pack('>8s', b'WAVEfmt '))
        # Number of bytes to follow in the format chunk
        f.write(struct.pack('<I', 16))
        # The format tag
        f.write(struct.pack('<H', 1))
        # The number of signals
        f.write(struct.pack('<H', record.n_sig))
        # The samples per second
        f.write(struct.pack('<I', samps_per_second))
        # The number of bytes per second
        f.write(struct.pack('<I', samps_per_second * frame_length))
        # The length of each frame
        f.write(struct.pack('<H', frame_length))
        # The number of bits per samples
        f.write(struct.pack('<H', bits_per_sample))
        # The descriptor to indicate that the data information is next
        f.write(struct.pack('>4s', b'data'))
        # The number of bytes in the signal data chunk
        f.write(struct.pack('<I', chunk_bytes))
        # Write the signal data... the closest I can get to the original implementation
        # Mismatched elements: 723881 / 15400000 (4.7%)
        # Max absolute difference: 2
        # Max relative difference: 0.00444444
        #  x: array([ -322,  3852, -9246, ...,     0,     0,     0], dtype=int16)
        #  y: array([ -322,  3852, -9246, ...,     0,     0,     0], dtype=int16)
        sig_data = np.left_shift(np.subtract(record.adc(), offset), shift).reshape((1, -1)).astype(np.int16)
        sig_data.tofile(f)

    # If asked to write the accompanying header file
    if write_header:
        record.adc_zero = record.n_sig * [0]
        record.adc_res = record.n_sig * [16]
        record.adc_gain = [(r * (1 << shift[i])) for i,r in enumerate(record.adc_gain)]
        record.baseline = [(b - offset[i]) for i,b in enumerate(record.baseline)]
        record.baseline = [(b * (1 << shift[i])) for i,b in enumerate(record.baseline)]
        record.file_name = record.n_sig * [record_name_out + '.wav']
        record.block_size = record.n_sig * [0]
        record.fmt = record.n_sig * ['16']
        record.samps_per_fram = record.n_sig * [1]
        record.init_value = sig_data[0][:record.n_sig].tolist()
        record.byte_offset = record.n_sig * [44]
        # Write the header file
        record.wrheader()


def wav2mit(record_name, pn_dir=None, delete_file=True, record_only=False):
    """
    Convert .wav (format 16, multiplexed signals, with embedded header
    information) formatted files to MIT format. See here for more details about
    the formatting of a .wav file: http://soundfile.sapp.org/doc/WaveFormat/.

    This process may not work with some .wav files that are encoded using
    variants of the original .wav format that are not WFDB-compatible. In
    principle, this program should be able to recognize such files by their
    format codes, and it will produce an error message in such cases. If
    the format code is incorrect, however, `wav2mit` may not recognize that
    an error has occurred.

    Parameters
    ----------
    record_name : str
        The name of the input .wav record to be read.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    delete_file : bool, optional
        Whether to delete the saved .wav file (False) or not (True)
        after being imported.
    record_only : bool, optional
        Whether to only return the record information (True) or not (False).
        If false, this function will generate both a .dat and .hea file.

    Returns
    -------
    record : dict, optional
        All of the record information needed to generate MIT formatted files.
        Only returns if 'record_only' is set to True, else generates the
        corresponding .dat and .hea files. This record file will not match the
        `rdrecord` output since it will only give us the digital signal for now.

    Notes
    -----
    Files that can be processed successfully using `wav2mit` always have exactly
    three chunks (a header chunk, a format chunk, and a data chunk).  In .wav
    files, binary data are always written in little-endian format (least
    significant byte first). The format of `wav2mit`'s input files is as follows:

    [Header chunk]
    Bytes  0 -  3: "RIFF" [4 ASCII characters]
    Bytes  4 -  7: L-8 (number of bytes to follow in the file, excluding bytes 0-7)
    Bytes  8 - 11: "WAVE" [4 ASCII characters]

    [Format chunk]
    Bytes 12 - 15: "fmt " [4 ASCII characters, note trailing space]
    Bytes 16 - 19: 16 (format chunk length in bytes, excluding bytes 12-19)
    Bytes 20 - 35: format specification, consisting of:
    Bytes 20 - 21: 1 (format tag, indicating no compression is used)
    Bytes 22 - 23: number of signals (1 - 65535)
    Bytes 24 - 27: sampling frequency in Hz (per signal)
                   Note that the sampling frequency in a .wav file must be an
                   integer multiple of 1 Hz, a restriction that is not imposed
                   by MIT (WFDB) format.
    Bytes 28 - 31: bytes per second (sampling frequency * frame size in bytes)
    Bytes 32 - 33: frame size in bytes
    Bytes 34 - 35: bits per sample (ADC resolution in bits)
                   Note that the actual ADC resolution (e.g., 12) is written in
                   this field, although each output sample is right-padded to fill
                   a full (16-bit) word. (.wav format allows for 8, 16, 24, and
                   32 bits per sample)

    [Data chunk]
    Bytes 36 - 39: "data" [4 ASCII characters]
    Bytes 40 - 43: L-44 (number of bytes to follow in the data chunk)
    Bytes 44 - L-1: sample data, consisting of:
    Bytes 44 - 45: sample 0, channel 0
    Bytes 46 - 47: sample 0, channel 1
    ... etc. (same order as in a multiplexed WFDB signal file)

    Examples
    --------
    >>> wav_record = wfdb.wav2mit('sample-data/SC4001E0-PSG.wav', record_only=True)

    """
    if not record_name.endswith('.wav'):
        raise Exception('Name of the input file must end in .wav')

    if pn_dir is not None:

        if '.' not in pn_dir:
            dir_list = pn_dir.split('/')
            pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]), *dir_list[1:])

        file_url = posixpath.join(download.PN_INDEX_URL, pn_dir, record_name)
        # Currently must download file to read it though can give the
        # user the option to delete it immediately afterwards
        r = requests.get(file_url, allow_redirects=False)
        open(record_name, 'wb').write(r.content)

    wave_file = open(record_name, mode='rb')
    record_name_out = record_name.split(os.sep)[-1].replace('-','_').replace('.wav','')

    chunk_ID = ''.join([s.decode() for s in struct.unpack('>4s', wave_file.read(4))])
    if chunk_ID != 'RIFF':
        raise Exception('{} is not a .wav-format file'.format(record_name))

    correct_chunk_size = os.path.getsize(record_name) - 8
    chunk_size = struct.unpack('<I', wave_file.read(4))[0]
    if chunk_size != correct_chunk_size:
        raise Exception('Header chunk has incorrect length (is {} should be {})'.format(chunk_size,correct_chunk_size))

    fmt = struct.unpack('>4s', wave_file.read(4))[0].decode()
    if fmt != 'WAVE':
        raise Exception('{} is not a .wav-format file'.format(record_name))

    subchunk1_ID = struct.unpack('>4s', wave_file.read(4))[0].decode()
    if subchunk1_ID != 'fmt ':
        raise Exception('Format chunk missing or corrupt')

    subchunk1_size = struct.unpack('<I', wave_file.read(4))[0]
    audio_format = struct.unpack('<H', wave_file.read(2))[0]
    if audio_format > 1:
        print('PCM has compression of {}'.format(audio_format))

    if (subchunk1_size != 16) or (audio_format != 1):
	    raise Exception('Unsupported format {}'.format(audio_format))

    num_channels = struct.unpack('<H', wave_file.read(2))[0]
    if num_channels == 1:
        print('Reading Mono formatted .wav file...')
    elif num_channels == 2:
        print('Reading Stereo formatted .wav file...')
    else:
        print('Reading {}-channel formatted .wav file...'.format(num_channels))

    sample_rate = struct.unpack('<I', wave_file.read(4))[0]
    print('Sample rate: {}'.format(sample_rate))
    byte_rate = struct.unpack('<I', wave_file.read(4))[0]
    print('Byte rate: {}'.format(byte_rate))
    block_align = struct.unpack('<H', wave_file.read(2))[0]
    print('Block align: {}'.format(block_align))
    bits_per_sample = struct.unpack('<H', wave_file.read(2))[0]
    print('Bits per sample: {}'.format(bits_per_sample))
    # I wish this were more precise but unfortunately some information
    # is lost in .wav files which is needed for these calculations
    if bits_per_sample <= 8:
        adc_res = 8
        adc_gain = 12.5
    elif bits_per_sample <= 16:
        adc_res = 16
        adc_gain = 6400
    else:
        raise Exception('Unsupported resolution ({} bits/sample)'.format(bits_per_sample))

    if block_align != (num_channels * int(adc_res / 8)):
	    raise Exception('Format chunk of {} has incorrect frame length'.format(block_align))

    subchunk2_ID = struct.unpack('>4s', wave_file.read(4))[0].decode()
    if subchunk2_ID != 'data':
        raise Exception('Format chunk missing or corrupt')

    correct_subchunk2_size = os.path.getsize(record_name) - 44
    subchunk2_size = struct.unpack('<I', wave_file.read(4))[0]
    if subchunk2_size != correct_subchunk2_size:
        raise Exception('Data chunk has incorrect length.. (is {} should be {})'.format(subchunk2_size, correct_subchunk2_size))
    sig_len = int(subchunk2_size / block_align)

    sig_data = (np.fromfile(wave_file, dtype=np.int16).reshape((-1,num_channels)) / (2*adc_res)).astype(np.int16)

    init_value = [int(s[0]) for s in np.transpose(sig_data)]
    checksum = [int(np.sum(v) % 65536) for v in np.transpose(sig_data)]  # not all values correct?

    if pn_dir is not None and delete_file:
        os.remove(record_name)

    record = Record(
        record_name = record_name_out,
        n_sig = num_channels,
        fs = num_channels * [sample_rate],
        samps_per_frame = num_channels * [1],
        counter_freq = None,
        base_counter = None,
        sig_len = sig_len,
        base_time = None,
        base_date = None,
        comments = [],
        sig_name = num_channels * [None],
        p_signal = None,
        d_signal = sig_data,
        e_p_signal = None,
        e_d_signal = None,
        file_name = num_channels * [record_name_out + '.dat'],
        fmt = num_channels * ['16' if (adc_res == 16) else '80'],
        skew = num_channels * [None],
        byte_offset = num_channels * [None],
        adc_gain = num_channels * [adc_gain],
        baseline = num_channels * [0 if (adc_res == 16) else 128],
        units = num_channels * [None],
        adc_res =  num_channels * [adc_res],
        adc_zero = num_channels * [0 if (adc_res == 16) else 128],
        init_value = init_value,
        checksum = checksum,
        block_size = num_channels * [0]
    )

    if record_only:
        return record
    else:
        # TODO: Generate the .dat and .hea files
        pass


def wfdb2mat(record_name, pn_dir=None, sampfrom=0, sampto=None, channels=None):
    """
    This program converts the signals of any PhysioNet record (or one in any
    compatible format) into a .mat file that can be read directly using any version
    of Matlab, and a short text file containing information about the signals
    (names, gains, baselines, units, sampling frequency, and start time/date if
    known). If the input record name is REC, the output files are RECm.mat and
    RECm.hea. The output files can also be read by any WFDB application as record
    RECm.

    This program does not convert annotation files; for that task, 'rdann' is
    recommended.

    The output .mat file contains a single matrix named `val` containing raw
    (unshifted, unscaled) samples from the selected record. Using various options,
    you can select any time interval within a record, or any subset of the signals,
    which can be rearranged as desired within the rows of the matrix. Since .mat
    files are written in column-major order (i.e., all of column n precedes all of
    column n+1), each vector of samples is written as a column rather than as a
    row, so that the column number in the .mat file equals the sample number in the
    input record (minus however many samples were skipped at the beginning of the
    record, as specified using the `start_time` option). If this seems odd, transpose
    your matrix after reading it!

    This program writes version 5 MAT-file format output files, as documented in
    http://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf
    The samples are written as 32-bit signed integers (mattype=20 below) in
    little-endian format if the record contains any format 24 or format 32 signals,
    as 8-bit unsigned integers (mattype=50) if the record contains only format 80
    signals, or as 16-bit signed integers in little-endian format (mattype=30)
    otherwise.

    The maximum size of the output variable is 2^31 bytes. `wfdb2mat` from versions
    10.5.24 and earlier of the original WFDB software package writes version 4 MAT-
    files which have the additional constraint of 100,000,000 elements per variable.

    The output files (recordm.mat + recordm.hea) are still WFDB-compatible, given
    the .hea file constructed by this program.

    Parameters
    ----------
    record_name : str
        The name of the input WFDB record to be read. Can also work with both
        EDF and WAV files.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    sampfrom : int, optional
        The starting sample number to read for all channels.
    sampto : int, 'end', optional
        The sample number at which to stop reading for all channels.
        Reads the entire duration by default.
    channels : list, optional
        List of integer indices specifying the channels to be read.
        Reads all channels by default.

    Returns
    -------
    N/A

    Notes
    -----
    The entire file is composed of:

    Bytes   0 - 127: descriptive text
    Bytes 128 - 131: master tag (data type = matrix)
    Bytes 132 - 135: master tag (data size)
    Bytes 136 - 151: array flags (4 byte tag with data type, 4 byte
                     tag with subelement size, 8 bytes of content)
    Bytes 152 - 167: array dimension (4 byte tag with data type, 4
                     byte tag with subelement size, 8 bytes of content)
    Bytes 168 - 183: array name (4 byte tag with data type, 4 byte
                     tag with subelement size, 8 bytes of content)
    Bytes 184 - ...: array content (4 byte tag with data type, 4 byte
                     tag with subelement size, ... bytes of content)

    Examples
    --------
    >>> wfdb.wfdb2mat('100', pn_dir='pwave')

    The output file name is 100m.mat and 100m.hea

    """
    record = rdrecord(record_name, pn_dir=pn_dir, sampfrom=sampfrom, sampto=sampto)
    record_name_out = record_name.split(os.sep)[-1].replace('-','_') + 'm'

    # Some variables describing the format of the .mat file
    field_version = 256         # 0x0100 or 256
    endian_indicator = b'IM'    # little endian
    master_type = 14            # matrix
    sub1_type = 6               # UINT32
    sub2_type = 5               # INT32
    sub3_type = 1               # INT8
    sub1_class = 6              # double precision array

    # Determine if we can write 8-bit unsigned samples, or if 16 or 32 bits
    # are needed per sample
    bytes_per_element = 1
    for i in range(record.n_sig):
        if (record.adc_res[i] > 0):
            if (record.adc_res[i] > 16):
                bytes_per_element = 4
            elif (record.adc_res[i] > 8) and (bytes_per_element < 2):
                bytes_per_element = 2
        else:
            # adc_res not specified.. try to guess from format
            if (record.fmt[i] == '24') or (record.fmt[i] == '32'):
                bytes_per_element = 4
            elif (record.fmt[i] != '80') and (bytes_per_element < 2):
                bytes_per_element = 2

    if (bytes_per_element == 1):
        sub4_type = 2       # MAT8
        out_type = '<u1'    # np.uint8
        wfdb_type = '80'    # Offset binary form (80)
        offset = 128        # Offset between sample values and the raw
                            # byte/word values as interpreted by Matlab/Octave
    elif (bytes_per_element == 2):
        sub4_type = 3       # MAT16
        out_type = '<i2'    # np.int16
        wfdb_type = '16'    # Align with byte boundary (16)
        offset = 0          # Offset between sample values and the raw
                            # byte/word values as interpreted by Matlab/Octave
    else:
        sub4_type = 5       # MAT32
        out_type = '<i4'    # np.int32
        wfdb_type = '32'    # Align with byte boundary (32)
        offset = 0          # Offset between sample values and the raw
                            # byte/word values as interpreted by Matlab/Octave

    # Ensure the signal size does not exceed the 2^31 byte limit
    max_length = int((2**31) / bytes_per_element / record.n_sig)
    if sampto is None:
        sampto = record.p_signal.shape[0]
    desired_length = sampto - sampfrom
    # Snip record
    if desired_length > max_length:
        raise Exception("Can't write .mat file: data size exceeds 2GB limit")

    # Bytes of actual data
    bytes_of_data = bytes_per_element * record.n_sig * desired_length
    # This is the remaining number of bytes that don't fit into integer
    # multiple of 8: i.e. if 18 bytes, bytes_remain = 2, from 17 to 18
    bytes_remain = bytes_of_data % 8

    # master_bytes = (8 + 8) + (8 + 8) + (8 + 8) + (8 + bytes_of_data) + padding
    # Must be integer multiple 8
    if bytes_remain == 0:
        master_bytes = bytes_of_data + 56
    else:
        master_bytes = bytes_of_data + 64 - (bytes_remain)

    # Start writing the file
    output_file = record_name_out + '.mat'
    with open(output_file, 'wb') as f:
        # Descriptive text (124 bytes)
        f.write(struct.pack('<124s', b'MATLAB 5.0'))
        # Version (2 bytes)
        f.write(struct.pack('<H', field_version))
        # Endian indicator (2 bytes)
        f.write(struct.pack('<2s', endian_indicator))

        # Master tag data type (4 bytes)
        f.write(struct.pack('<I', master_type))
        # Master tag number of bytes (4 bytes)
        # Number of bytes of data element
        #     = (8 + 8) + (8 + 8) + (8 + 8) + (8 + bytes_of_data)
        #     = 56 + bytes_of_data
        f.write(struct.pack('<I', master_bytes))

        # Matrix data has 4 subelements (5 if imaginary):
        #     Array flags, dimensions array, array name, real part
        # Each subelement has its own subtag, and subdata

        # Subelement 1: Array flags
        # Subtag 1: data type (4 bytes)
        f.write(struct.pack('<I', sub1_type))
        # Subtag 1: number of bytes (4 bytes)
        f.write(struct.pack('<I', 8))
        # Value class indication the MATLAB data type (8 bytes)
        f.write(struct.pack('<Q', sub1_class))

        # Subelement 2: Rows and columns
        # Subtag 2: data type (4 bytes)
        f.write(struct.pack('<I', sub2_type))
        # Subtag 2: number of bytes (4 bytes)
        f.write(struct.pack('<I', 8))
        # Number of signals (4 bytes)
        f.write(struct.pack('<I', record.n_sig))
        # Number of rows (4 bytes)
        f.write(struct.pack('<I', desired_length))

        # Subelement 3: Array name
        # Subtag 3: data type (4 bytes)
        f.write(struct.pack('<I', sub3_type))
        # Subtag 3: number of bytes (4 bytes)
        f.write(struct.pack('<I', 3))
        # Subtag 3: name of the array (8 bytes)
        f.write(struct.pack('<8s', b'val'))

        # Subelement 4: Signal data
        # Subtag 4: data type (4 bytes)
        f.write(struct.pack('<I', sub4_type))
        # Subtag 4: number of bytes (4 bytes)
        f.write(struct.pack('<I', bytes_of_data))

        # Total size of everything before actual data:
        #     128 byte header
        #     + 8 byte master tag
        #     + 56 byte subelements (48 byte default + 8 byte name)
        #     = 192

        # Copy the selected data into the .mat file
        out_data = record.p_signal * record.adc_gain + record.baseline - record.adc_zero
        # Cast the data to the correct type base on the bytes_per_element
        out_data = np.around(out_data).astype(out_type)
        # out_data should be [r1c1, r1c2, r2c1, r2c2, etc.]
        out_data = out_data.flatten()
        out_fmt = '<%sh' % len(out_data)
        f.write(struct.pack(out_fmt, *out_data))

    # Display some useful information
    if record.base_time is None:
        if record.base_date is None:
            datetime_string = '[None]'
        else:
            datetime_string = '[{}]'.format(record.base_date.strftime('%d/%m/%Y'))
    else:
        if record.base_date is None:
            datetime_string = '[{}]'.format(record.base_time.strftime('%H:%M:%S.%f'))
        else:
            datetime_string = '[{} {}]'.format(record.base_time.strftime('%H:%M:%S.%f'),
                                               record.base_date.strftime('%d/%m/%Y'))

    print('Source: record {}\t\tStart: {}'.format(record_name, datetime_string))
    print('val has {} rows (signals) and {} columns (samples/signal)'.format(record.n_sig,
                                                                             desired_length))
    duration_string = str(datetime.timedelta(seconds=desired_length/record.fs))
    print('Duration: {}'.format(duration_string))
    print('Sampling frequency: {} Hz\tSampling interval: {} sec'.format(record.fs,
                                                                        1/record.fs))
    print('{:<7}{:<20}{:<17}{:<10}{:<10}'.format('Row','Signal','Gain','Base','Units'))
    record.sig_name = [s.replace(' ','_') for s in record.sig_name]
    for i in range(record.n_sig):
        print('{:<7}{:<20}{:<17}{:<10}{:<10}'.format(i,
                                                     record.sig_name[i],
                                                     record.adc_gain[i],
                                                     record.baseline[i]-record.adc_zero[i]+offset,
                                                     record.units[i]))

    # Modify the record file to reflect the new data
    num_channels = record.n_sig if (channels is None) else len(channels)
    record.record_name = record_name_out
    record.n_sig = num_channels
    record.samps_per_frame = num_channels * [1]
    record.file_name = num_channels * [output_file]
    record.fmt = num_channels * [wfdb_type]
    record.byte_offset = num_channels * [192]
    record.baseline = [b - record.adc_zero[i] for i,b in enumerate(record.baseline)]
    record.adc_zero = num_channels * [0]
    record.init_value = out_data[:record.n_sig].tolist()

    # Write the header file RECm.hea
    record.wrheader()
    # Append the following lines to create a signature
    with open(record_name_out+'.hea','a') as f:
        f.write('#Creator: wfdb2mat\n')
        f.write('#Source: record {}\n'.format(record_name))


def csv2mit(file_name, fs, units, fmt=None, adc_gain=None, baseline=None,
            samps_per_frame=None, counter_freq=None, base_counter=None,
            base_time=None, base_date=None, comments=None, sig_name=None,
            dat_file_name=None, skew=None, byte_offset=None, adc_res=None,
            adc_zero=None, init_value=None, checksum=None, block_size=None,
            record_only=False, header=True, delimiter=',', verbose=False):
    """
    Read a WFDB header file and return either a `Record` object with the
    record descriptors as attributes or write a record and header file.

    Parameters
    ----------
    file_name : str
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    fs : float
        This number can be expressed in any format legal for a Python input of
        floating point numbers (thus '360', '360.', '360.0', and '3.6e2' are
        all legal and equivalent). The sampling frequency must be greater than 0;
        if it is missing, a value of 250 is assumed.
    units : list, str
        This will be applied as the passed list unless a single str is passed
        instead - in which case the str will be assigned for all channels.
        This field can be present only if the ADC gain is also present. It
        follows the baseline field if that field is present, or the gain field
        if the baseline field is absent. The units field is a list of character
        strings that specifies the type of physical unit. If the units field is
        absent, the physical unit may be assumed to be 1 mV.
    fmt : list, str, optional
        This will be applied as the passed list unless a single str is passed
        instead - in which case the str will be assigned for all
        channels. A list of strings giving the WFDB format of each file used to
        store each channel. Accepted formats are: '80','212','16','24', and
        '32'. There are other WFDB formats as specified by:
        https://www.physionet.org/physiotools/wag/signal-5.htm
        but this library will not write (though it will read) those file types.
        Each field is an integer that specifies the storage format of the signal.
        All signals in a given group are stored in the same format. The most
        common format is format `16` (sixteen-bit amplitudes). The parameters
        `samps_per_frame`, `skew`, and `byte_offset` are optional fields, and
        if present, are bound to the format field. In other words, they may be
        considered as format modifiers, since they further describe the encoding
        of samples within the signal file.
    adc_gain : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        This field is a list of numbers that specifies the difference in
        sample values that would be observed if a step of one physical unit
        occurred in the original analog signal. For ECGs, the gain is usually
        roughly equal to the R-wave amplitude in a lead that is roughly parallel
        to the mean cardiac electrical axis. If the gain is zero or missing, this
        indicates that the signal amplitude is uncalibrated; in such cases, a
        value of 200 ADC units per physical unit may be assumed.
    baseline : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels. This
        field can be present only if the ADC gain is also present. It is not
        separated by whitespace from the ADC gain field; rather, it is
        surrounded by parentheses, which delimit it. The baseline is an integer
        that specifies the sample value corresponding to 0 physical units. If
        absent, the baseline is taken to be equal to the ADC zero. Note that
        the baseline need not be a value within the ADC range; for example,
        if the ADC input range corresponds to 200-300 degrees Kelvin, the
        baseline is the (extended precision) value that would map to 0 degrees
        Kelvin.
    samps_per_frame : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        Normally, all signals in a given record are sampled at the (base)
        sampling frequency as specified by `fs`; in this case, the number of
        samples per frame is 1 for all signals, and this field is conventionally
        omitted. If the signal was sampled at some integer multiple, n, of the
        base sampling frequency, however, each frame contains n samples of the
        signal, and the value specified in this field is also n. (Note that
        non-integer multiples of the base sampling frequency are not supported).
    counter_freq : float, optional
        This field (a floating-point number, in the same format as `fs`) can be
        present only if `fs` is also present. Typically, the counter frequency
        may be derived from an analog tape counter, or from page numbers in a
        chart recording. If the counter frequency is absent or not positive,
        it is assumed to be equal to `fs`.
    base_counter : float, optional
        This field can be present only if the counter frequency is also present.
        The base counter value is a floating-point number that specifies the counter
        value corresponding to sample 0. If absent, the base counter value is
        taken to be 0.
    base_time : str, optional
        This field can be present only if the number of samples is also present.
        It gives the time of day that corresponds to the beginning of the
        record, in 'HH:MM:SS' format (using a 24-hour clock; thus '13:05:00', or
        '13:5:0', represent 1:05 pm). If this field is absent, the time-conversion
        functions assume a value of '0:0:0', corresponding to midnight.
    base_date : str, optional
        This field can be present only if the base time is also present. It contains
        the date that corresponds to the beginning of the record, in 'DD/MM/YYYY'
        format (e.g., '25/4/1989' is '25 April 1989').
    comments : list, optional
        A list of string comments to be written to the header file. Each string
        entry represents a new line to be appended to the bottom of the header
        file ('.hea').
    sig_name : list, optional
        A list of strings giving the signal name of each signal channel. This
        will be used for plotting the signal both in this package and
        LightWave. Note, this value will be used in preference to the CSV
        header, if applicable, to define custom signal names.
    dat_file_name : str, optional
        The name of the file in which samples of the signal are kept. Although the
        record name is usually part of the signal file name, this convention is
        not a requirement. Note that several signals can share the same file
        (i.e., they can belong to the same signal group); all entries for signals
        that share a given file must be consecutive, however. Note, the default
        behavior is to save the files in the current working directory, not the
        directory of the file being read.
    skew : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        Ideally, within a given record, samples of different signals with the
        same sample number are simultaneous (within one sampling interval).
        If this is not the case (as, for example, when a multitrack analog
        tape recording is digitized and the azimuth of the playback head does
        not match that of the recording head), the skew between signals can
        sometimes be determined (for example, by locating recorded waveform
        features with known time relationships, such as calibration signals).
        If this has been done, the skew field may be inserted into the header
        file to indicate the (positive) number of samples of the signal that
        are considered to precede sample 0. These samples, if any, are included
        in the checksum. (Note the checksum need not be changed if the skew field
        is inserted or modified).
    byte_offset : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        Normally, signal files include only sample data. If a signal file
        includes a preamble, however, this field specifies the offset in bytes
        from the beginning of the signal file to sample 0 (i.e., the length
        of the preamble). Data within the preamble is not included in the signal
        checksum. Note that the byte offset must be the same for all signals
        within a given group (use the skew field to correct for intersignal
        skew). This feature is provided only to simplify the task of reading
        signal files not generated using the WFDB library; the WFDB library
        does not support any means of writing such files, and byte offsets must
        be inserted into header files manually.
    adc_res: list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        This field can be present only if the ADC gain is also present. It
        specifies the resolution of the analog-to-digital converter used to
        digitize the signal. Typical ADCs have resolutions between 8 and 16
        bits. If this field is missing or zero, the default value is 12 bits
        for amplitude-format signals, or 10 bits for difference-format signals
        (unless a lower value is specified by the format field).
    adc_zero: list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        This field can be present only if the ADC resolution is also present.
        It is an integer that represents the amplitude (sample value) that
        would be observed if the analog signal present at the ADC inputs had
        a level that fell exactly in the middle of the input range of the ADC.
        For a bipolar ADC, this value is usually zero, but a unipolar (offset
        binary) ADC usually produces a non-zero value in the middle of its
        range. Together with the ADC resolution, the contents of this field
        can be used to determine the range of possible sample values. If this
        field is missing, a value of 0 is assumed.
    init_value : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        This field can be present only if the ADC zero is also present. It
        specifies the value of sample 0 in the signal, but is used only if the
        signal is stored in difference format. If this field is missing, a
        value equal to the ADC zero is assumed.
    checksum : list, optional
        This field can be present only if the initial value is also present. It
        is a 16-bit signed checksum of all samples in the signal. (Thus the
        checksum is independent of the storage format.) If the entire record
        is read without skipping samples, and the header’s record line specifies
        the correct number of samples per signal, this field is compared against
        a computed checksum to verify that the signal file has not been corrupted.
        A value of zero may be used as a field placeholder if the number of
        samples is unspecified.
    block_size : list, int, optional
        This will be applied as the passed list unless a single int is passed
        instead - in which case the int will be assigned for all channels.
        This field can be present only if the checksum is present. This field
        is an integer and is usually 0. If the signal is stored in a file
        that must be read in blocks of a specific size, however, this field
        specifies the block size in bytes. (On UNIX systems, this is the case
        only for character special files, corresponding to certain tape and
        raw disk files. If necessary, the block size may be given as a negative
        number to indicate that the associated file lacks I/O driver support for
        some operations.) All signals belonging to the same signal group have
        the same block size.
    record_only : bool, optional
        Whether to only return the record information (True) or not (False).
        If false, this function will generate both a .dat and .hea file.
    header : bool, optional
        Whether to assume the CSV has a first line header (True) or not (False)
        which defines the signal names. If false, this function will generate
        either the signal names provided by `sig_name` or set `[ch_1, ch_2, ...]`
        as the default.
    delimiter : str, optional
        What to use as the delimiter for the file to separate data. The default
        if a comma (','). Other common delimiters are tabs ('\t'), spaces (' '),
        pipes ('|'), and colons (':').
    verbose : bool, optional
        Whether to print all the information read about the file (True) or
        not (False).

    Returns
    -------
    record : Record or MultiRecord, optional
        The WFDB Record or MultiRecord object representing the contents
        of the CSV file read.

    Notes
    -----
    CSVs should be in the following format:

    sig_1_name,sig_2_name,...
    sig_1_val_1,sig_2_val_1,...
    sig_1_val_2,sig_2_val_2,...
    ...,...,...

    Or this format if `header=False` is defined:

    sig_1_val_1,sig_2_val_1,...
    sig_1_val_2,sig_2_val_2,...
    ...,...,...

    The signal will be saved defaultly as a `p_signal` so both floats and
    ints are acceptable.

    Examples
    --------
    Create the header ('.hea') and record ('.dat') files, specifies both
    units to be 'mV'
    >>> wfdb.csv2mit('sample-data/100.csv', fs=360, units='mV')

    Create the header ('.hea') and record ('.dat') files, change units for
    each signal
    >>> wfdb.csv2mit('sample-data/100.csv', fs=360, units=['mV','kV'])

    Return just the record, note the use of lists to specify which values should
    be applied to each signal
    >>> csv_record = wfdb.csv2mit('sample-data/100.csv', fs=360, units=['mV','mV'],
                                  fmt=['80',212'], adc_gain=[100,200],
                                  baseline=[1024,512], record_only=True)

    Return just the record, note the use of single strings and ints to specify
    when fields can be applied to all signals
    >>> csv_record = wfdb.csv2mit('sample-data/100.csv', fs=360, units='mV',
                                  fmt=['80','212'], adc_gain=200, baseline=1024,
                                  record_only=True)

    """
    # NOTE: No need to write input checks here since the Record class should
    # handle them (except verifying the CSV input format which is for Pandas)
    if header:
        df_CSV = pd.read_csv(file_name, delimiter=delimiter)
    else:
        df_CSV = pd.read_csv(file_name, delimiter=delimiter, header=None)
    if verbose:
        print('Successfully read CSV')
    # Extract the entire signal from the dataframe
    p_signal = df_CSV.values
    # The dataframe should be in (`sig_len`, `n_sig`) dimensions
    sig_len = p_signal.shape[0]
    if verbose:
        print('Signal length: {}'.format(sig_len))
    n_sig = p_signal.shape[1]
    if verbose:
        print('Number of signals: {}'.format(n_sig))
    # Check if signal names are valid and set defaults
    if not sig_name:
        if header:
            sig_name = df_CSV.columns.to_list()
            if any(map(str.isdigit, sig_name)):
                print("WARNING: One or more of your signal names are numbers, this "\
                      "is not recommended:\n- Does your CSV have a header line "\
                      "which defines the signal names?\n- If not, please set the "\
                      "parameter 'header' to False.\nSignal names: {}".format(sig_name))
        else:
            sig_name = ['ch_'+str(i) for i in range(n_sig)]
            if verbose:
                print('Signal names: {}'.format(sig_name))

    # Set the output header file name to be the same, remove path
    if os.sep in file_name:
        file_name = file_name.split(os.sep)[-1]
    record_name = file_name.replace('.csv','')
    if verbose:
        print('Output header: {}.hea'.format(record_name))

    # Replace the CSV file tag with DAT
    dat_file_name = file_name.replace('.csv','.dat')
    dat_file_name = [dat_file_name] * n_sig
    if verbose:
        print('Output record: {}'.format(dat_file_name[0]))

    # Convert `units` from string to list if necessary
    units = [units]*n_sig if type(units) is str else units

    # Set the default `fmt` if none exists
    if not fmt:
        fmt = ['16'] * n_sig
    fmt = [fmt]*n_sig if type(fmt) is str else fmt
    if verbose:
        print('Signal format: {}'.format(fmt))

    # Set the default `adc_gain` if none exists
    if not adc_gain:
        adc_gain = [200] * n_sig
    adc_gain = [adc_gain]*n_sig if type(adc_gain) is int else adc_gain
    if verbose:
        print('Signal ADC gain: {}'.format(adc_gain))

    # Set the default `baseline` if none exists
    if not baseline:
        if adc_zero:
            baseline = [adc_zero] * n_sig
        else:
            baseline = [0] * n_sig
    baseline = [baseline]*n_sig if type(baseline) is int else baseline
    if verbose:
        print('Signal baseline: {}'.format(baseline))

    # Convert `samps_per_frame` from int to list if necessary
    samps_per_frame = [samps_per_frame]*n_sig if type(samps_per_frame) is int else samps_per_frame

    # Convert `skew` from int to list if necessary
    skew = [skew]*n_sig if type(skew) is int else skew

    # Convert `byte_offset` from int to list if necessary
    byte_offset = [byte_offset]*n_sig if type(byte_offset) is int else byte_offset

    # Set the default `adc_res` if none exists
    if not adc_res:
        adc_res = [12] * n_sig
    adc_res = [adc_res]*n_sig if type(adc_res) is int else adc_res
    if verbose:
        print('Signal ADC resolution: {}'.format(adc_res))

    # Set the default `adc_zero` if none exists
    if not adc_zero:
        adc_zero = [0] * n_sig
    adc_zero = [adc_zero]*n_sig if type(adc_zero) is int else adc_zero
    if verbose:
        print('Signal ADC zero: {}'.format(adc_zero))

    # Set the default `init_value`
    # NOTE: Initial value (and subsequently the digital signal) won't be correct
    # unless the correct `baseline` and `adc_gain` are provided... this is just
    # the best approximation
    if not init_value:
        init_value = p_signal[0,:]
        init_value = baseline + (np.array(adc_gain) * init_value)
        init_value = [int(i) for i in init_value.tolist()]
    if verbose:
        print('Signal initial value: {}'.format(init_value))

    # Set the default `checksum`
    if not checksum:
        checksum = [int(np.sum(v) % 65536) for v in np.transpose(p_signal)]
    if verbose:
        print('Signal checksum: {}'.format(checksum))

    # Set the default `block_size`
    if not block_size:
        block_size = [0] * n_sig
    block_size = [block_size]*n_sig if type(block_size) is int else block_size
    if verbose:
        print('Signal block size: {}'.format(block_size))

    # Change the dates and times into `datetime` objects
    if base_time:
        base_time = _header.wfdb_strptime(base_time)
    if base_date:
        base_date = datetime.datetime.strptime(base_date, '%d/%m/%Y').date()

    # Convert array to floating point
    p_signal = p_signal.astype('float64')

    # Either return the record or generate the record and header files
    # if requested
    if record_only:
        # Create the record from the input and generated values
        record = Record(
            record_name = record_name,
            n_sig = n_sig,
            fs = fs,
            samps_per_frame = samps_per_frame,
            counter_freq = counter_freq,
            base_counter = base_counter,
            sig_len = sig_len,
            base_time = base_time,
            base_date = base_date,
            comments = comments,
            sig_name = sig_name,
            p_signal = p_signal,
            d_signal = None,
            e_p_signal = None,
            e_d_signal = None,
            file_name = dat_file_name,
            fmt = fmt,
            skew = skew,
            byte_offset = byte_offset,
            adc_gain = adc_gain,
            baseline = baseline,
            units = units,
            adc_res =  adc_res,
            adc_zero = adc_zero,
            init_value = init_value,
            checksum = checksum,
            block_size = block_size
        )
        if verbose:
            print('Record generated successfully')
        return record

    else:
        # Write the information to a record and header file
        wrsamp(
            record_name = record_name,
            fs = fs,
            units = units,
            sig_name = sig_name,
            p_signal = p_signal,
            fmt = fmt,
            adc_gain = adc_gain,
            baseline = baseline,
            comments = comments,
            base_time = base_time,
            base_date = base_date,
        )
        if verbose:
            print('File generated successfully')


#------------------------- Reading Records --------------------------- #


def rdheader(record_name, pn_dir=None, rd_segments=False):
    """
    Read a WFDB header file and return a `Record` or `MultiRecord`
    object with the record descriptors as attributes.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    rd_segments : bool, optional
        Used when reading multi-segment headers. If True, segment headers will
        also be read (into the record object's `segments` field).

    Returns
    -------
    record : Record or MultiRecord
        The WFDB Record or MultiRecord object representing the contents
        of the header read.

    Examples
    --------
    >>> ecg_record = wfdb.rdheader('sample-data/test01_00s', sampfrom=800,
                                   channels = [1,3])

    """
    dir_name, base_record_name = os.path.split(record_name)
    dir_name = os.path.abspath(dir_name)

    if (pn_dir is not None) and ('.' not in pn_dir):
        dir_list = pn_dir.split('/')
        pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]), *dir_list[1:])

    # Read the header file. Separate comment and non-comment lines
    header_lines, comment_lines = _header._read_header_lines(base_record_name,
                                                             dir_name, pn_dir)

    # Get fields from record line
    record_fields = _header._parse_record_line(header_lines[0])

    # Single segment header - Process signal specification lines
    if record_fields['n_seg'] is None:
        # Create a single-segment WFDB record object
        record = Record()

        # There are signals
        if len(header_lines)>1:
            # Read the fields from the signal lines
            signal_fields = _header._parse_signal_lines(header_lines[1:])
            # Set the object's signal fields
            for field in signal_fields:
                setattr(record, field, signal_fields[field])

        # Set the object's record line fields
        for field in record_fields:
            if field == 'n_seg':
                continue
            setattr(record, field, record_fields[field])
    # Multi segment header - Process segment specification lines
    else:
        # Create a multi-segment WFDB record object
        record = MultiRecord()
        # Read the fields from the segment lines
        segment_fields = _header._read_segment_lines(header_lines[1:])
        # Set the object's segment fields
        for field in segment_fields:
            setattr(record, field, segment_fields[field])
        # Set the objects' record fields
        for field in record_fields:
            setattr(record, field, record_fields[field])

        # Determine whether the record is fixed or variable
        if record.seg_len[0] == 0:
            record.layout = 'variable'
        else:
            record.layout = 'fixed'

        # If specified, read the segment headers
        if rd_segments:
            record.segments = []
            # Get the base record name (could be empty)
            for s in record.seg_name:
                if s == '~':
                    record.segments.append(None)
                else:
                    record.segments.append(rdheader(os.path.join(dir_name, s),
                                                    pn_dir))
            # Fill in the sig_name attribute
            record.sig_name = record.get_sig_name()
            # Fill in the sig_segments attribute
            record.sig_segments = record.get_sig_segments()

    # Set the comments field
    record.comments = [line.strip(' \t#') for line in comment_lines]

    return record


def rdrecord(record_name, sampfrom=0, sampto=None, channels=None,
             physical=True, pn_dir=None, m2s=True, smooth_frames=True,
             ignore_skew=False, return_res=64, force_channels=True,
             channel_names=None, warn_empty=False):
    """
    Read a WFDB record and return the signal and record descriptors as
    attributes in a Record or MultiRecord object.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
        Can also handle .edf and .wav files as long as the extension is
        provided in the `record_name`.
    sampfrom : int, optional
        The starting sample number to read for all channels.
    sampto : int, 'end', optional
        The sample number at which to stop reading for all channels.
        Reads the entire duration by default.
    channels : list, optional
        List of integer indices specifying the channels to be read.
        Reads all channels by default.
    physical : bool, optional
        Specifies whether to return signals in physical units in the
        `p_signal` field (True), or digital units in the `d_signal`
        field (False).
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    m2s : bool, optional
        Used when reading multi-segment records. Specifies whether to
        directly return a WFDB MultiRecord object (False), or to convert
        it into and return a WFDB Record object (True).
    smooth_frames : bool, optional
        Used when reading records with signals having multiple samples
        per frame. Specifies whether to smooth the samples in signals
        with more than one sample per frame and return an (MxN) uniform
        numpy array as the `d_signal` or `p_signal` field (True), or to
        return a list of 1d numpy arrays containing every expanded
        sample as the `e_d_signal` or `e_p_signal` field (False).
    ignore_skew : bool, optional
        Used when reading records with at least one skewed signal.
        Specifies whether to apply the skew to align the signals in the
        output variable (False), or to ignore the skew field and load in
        all values contained in the dat files unaligned (True).
    return_res : int, optional
        The numpy array dtype of the returned signals. Options are: 64,
        32, 16, and 8, where the value represents the numpy int or float
        dtype. Note that the value cannot be 8 when physical is True
        since there is no float8 format.
    force_channels : bool, optional
        Used when reading multi-segment variable layout records. Whether
        to update the layout specification record, and the converted
        Record object if `m2s` is True, to match the input `channels`
        argument, or to omit channels in which no read segment contains
        the signals.
    channel_names : list, optional
        List of channel names to return. If this parameter is specified,
        it takes precedence over `channels`.
    warn_empty : bool, optional
        Whether to display a warning if the specified channel indices
        or names are not contained in the record, and no signal is
        returned.

    Returns
    -------
    record : Record or MultiRecord
        The WFDB Record or MultiRecord object representing the contents
        of the record read.

    Notes
    -----
    If a signal range or channel selection is specified when calling
    this function, the resulting attributes of the returned object will
    be set to reflect the section of the record that is actually read,
    rather than necessarily the entire record. For example, if
    `channels=[0, 1, 2]` is specified when reading a 12 channel record,
    the 'n_sig' attribute will be 3, not 12.

    The `rdsamp` function exists as a simple alternative to `rdrecord`
    for the common purpose of extracting the physical signals and a few
    important descriptor fields.

    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/test01_00s', sampfrom=800,
                               channels=[1, 3])

    """
    dir_name, base_record_name = os.path.split(record_name)
    dir_name = os.path.abspath(dir_name)

    # Read the header fields
    if (pn_dir is not None) and ('.' not in pn_dir):
        dir_list = pn_dir.split('/')
        pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]), *dir_list[1:])

    if record_name.endswith('.edf'):
        record = edf2mit(record_name, pn_dir=pn_dir, record_only=True)
    elif record_name.endswith('.wav'):
        record = wav2mit(record_name, pn_dir=pn_dir, record_only=True)
    else:
        record = rdheader(record_name, pn_dir=pn_dir, rd_segments=False)

    # Set defaults for sampto and channels input variables
    if sampto is None:
        # If the header does not contain the signal length, figure it
        # out from the first dat file. This is only possible for single
        # segment records. If there are no signals, sig_len is 0.
        if record.sig_len is None:
            if record.n_sig == 0:
                record.sig_len = 0
            else:
                record.sig_len = _signal._infer_sig_len(
                    file_name=record.file_name[0], fmt=record.fmt[0],
                    n_sig=record.file_name.count(record.file_name[0]),
                    dir_name=dir_name, pn_dir=pn_dir)
        sampto = record.sig_len

    # channel_names takes precedence over channels
    if channel_names is not None:
        # Figure out the channel indices matching the record, if any.
        if isinstance(record, Record):
            reference_record = record
        else:
            if record.layout == 'fixed':
                # Find the first non-empty segment to get the signal
                # names
                first_seg_name = [n for n in record.seg_name if n != '~'][0]
                reference_record = rdheader(os.path.join(dir_name,
                                                         record.seg_name[0]),
                                            pn_dir=pn_dir)
            else:
                # Use the layout specification header to get the signal
                # names
                reference_record = rdheader(os.path.join(dir_name,
                                                         record.seg_name[0]),
                                            pn_dir=pn_dir)

        channels = _get_wanted_channels(wanted_sig_names=channel_names,
                                       record_sig_names=reference_record.sig_name)

    elif channels is None:
        channels = list(range(record.n_sig))

    # Ensure that input fields are valid for the record
    record.check_read_inputs(sampfrom, sampto, channels, physical,
                             smooth_frames, return_res)

    # If the signal doesn't have the specified channels, there will be
    # no signal. Recall that `rdsamp` is not called on segments of multi
    # segment records if the channels are not present, so this won't
    # break anything.
    if not len(channels):
        old_record = record
        record = Record()
        for attr in _header.RECORD_SPECS.index:
            if attr == 'n_seg':
                continue
            elif attr in ['n_sig', 'sig_len']:
                setattr(record, attr, 0)
            else:
                setattr(record, attr, getattr(old_record, attr))
        if warn_empty:
            print('None of the specified signals were contained in the record')

    # A single segment record
    elif isinstance(record, Record):

        # Only 1 sample/frame, or frames are smoothed. Return uniform numpy array
        if smooth_frames or max([record.samps_per_frame[c] for c in channels]) == 1:
            # Read signals from the associated dat files that contain
            # wanted channels
            if record_name.endswith('.edf') or record_name.endswith('.wav'):
                record.d_signal = _signal._rd_segment(record.file_name,
                                                      dir_name, pn_dir,
                                                      record.fmt,
                                                      record.n_sig,
                                                      record.sig_len,
                                                      record.byte_offset,
                                                      record.samps_per_frame,
                                                      record.skew, sampfrom,
                                                      sampto, channels,
                                                      smooth_frames,
                                                      ignore_skew,
                                                      no_file=True,
                                                      sig_data=record.d_signal,
                                                      return_res=return_res)
            else:
                record.d_signal = _signal._rd_segment(record.file_name,
                                                      dir_name, pn_dir,
                                                      record.fmt,
                                                      record.n_sig,
                                                      record.sig_len,
                                                      record.byte_offset,
                                                      record.samps_per_frame,
                                                      record.skew, sampfrom,
                                                      sampto, channels,
                                                      smooth_frames,
                                                      ignore_skew,
                                                      return_res=return_res)

            # Arrange/edit the object fields to reflect user channel
            # and/or signal range input
            record._arrange_fields(channels=channels, sampfrom=sampfrom,
                                   expanded=False)

            if physical:
                # Perform inplace dac to get physical signal
                record.dac(expanded=False, return_res=return_res, inplace=True)

        # Return each sample of the signals with multiple samples per frame
        else:
            if record_name.endswith('.edf') or record_name.endswith('.wav'):
                record.e_d_signal = _signal._rd_segment(record.file_name,
                                                      dir_name, pn_dir,
                                                      record.fmt,
                                                      record.n_sig,
                                                      record.sig_len,
                                                      record.byte_offset,
                                                      record.samps_per_frame,
                                                      record.skew, sampfrom,
                                                      sampto, channels,
                                                      smooth_frames,
                                                      ignore_skew,
                                                      no_file=True,
                                                      sig_data=record.d_signal,
                                                      return_res=return_res)
            else:
                record.e_d_signal = _signal._rd_segment(record.file_name,
                                                        dir_name, pn_dir,
                                                        record.fmt,
                                                        record.n_sig,
                                                        record.sig_len,
                                                        record.byte_offset,
                                                        record.samps_per_frame,
                                                        record.skew, sampfrom,
                                                        sampto, channels,
                                                        smooth_frames,
                                                        ignore_skew,
                                                        return_res=return_res)

            # Arrange/edit the object fields to reflect user channel
            # and/or signal range input
            record._arrange_fields(channels=channels, sampfrom=sampfrom,
                                   expanded=True)

            if physical:
                # Perform dac to get physical signal
                record.dac(expanded=True, return_res=return_res, inplace=True)

    # A multi segment record
    else:
        # Strategy:
        # 1. Read the required segments and store them in
        # Record objects.
        # 2. Update the parameters of the objects to reflect
        # the state of the sections read.
        # 3. Update the parameters of the overall MultiRecord
        # object to reflect the state of the individual segments.
        # 4. If specified, convert the MultiRecord object
        # into a single Record object.

        # Segments field is a list of Record objects
        # Empty segments store None.

        record.segments = [None] * record.n_seg

        # Variable layout, read the layout specification header
        if record.layout == 'variable':
            record.segments[0] = rdheader(os.path.join(dir_name,
                                                       record.seg_name[0]),
                                          pn_dir=pn_dir)

        # The segment numbers and samples within each segment to read.
        seg_numbers, seg_ranges  = record._required_segments(sampfrom, sampto)
        # The channels within each segment to read
        seg_channels = record._required_channels(seg_numbers, channels,
                                                 dir_name, pn_dir)

        # Read the desired samples in the relevant segments
        for i in range(len(seg_numbers)):
            seg_num = seg_numbers[i]
            # Empty segment or segment with no relevant channels
            if record.seg_name[seg_num] == '~' or len(seg_channels[i]) == 0:
                record.segments[seg_num] = None
            else:
                record.segments[seg_num] = rdrecord(
                    os.path.join(dir_name, record.seg_name[seg_num]),
                    sampfrom=seg_ranges[i][0], sampto=seg_ranges[i][1],
                    channels=seg_channels[i], physical=physical, pn_dir=pn_dir,
                    return_res=return_res)

        # Arrange the fields of the layout specification segment, and
        # the overall object, to reflect user input.
        record._arrange_fields(seg_numbers=seg_numbers, seg_ranges=seg_ranges,
                               channels=channels, sampfrom=sampfrom,
                               force_channels=force_channels)

        # Convert object into a single segment Record object
        if m2s:
            record = record.multi_to_single(physical=physical,
                                            return_res=return_res)

    # Perform dtype conversion if necessary
    if isinstance(record, Record) and record.n_sig > 0:
        record.convert_dtype(physical, return_res, smooth_frames)

    return record


def rdsamp(record_name, sampfrom=0, sampto=None, channels=None, pn_dir=None,
           channel_names=None, warn_empty=False, return_res=64):
    """
    Read a WFDB record, and return the physical signals and a few important
    descriptor fields.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read (without any file
        extensions). If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/baserecord
        and the data files will be searched for in the local path.
    sampfrom : int, optional
        The starting sample number to read for all channels.
    sampto : int, 'end', optional
        The sample number at which to stop reading for all channels.
        Reads the entire duration by default.
    channels : list, optional
        List of integer indices specifying the channels to be read.
        Reads all channels by default.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    channel_names : list, optional
        List of channel names to return. If this parameter is specified,
        it takes precedence over `channels`.
    warn_empty : bool, optional
        Whether to display a warning if the specified channel indices
        or names are not contained in the record, and no signal is
        returned.
    return_res : int, optional
        The numpy array dtype of the returned signals. Options are: 64,
        32, 16, and 8, where the value represents the numpy int or float
        dtype. Note that the value cannot be 8 when physical is True
        since there is no float8 format.

    Returns
    -------
    signals : ndarray
        A 2d numpy array storing the physical signals from the record.
    fields : dict
        A dictionary containing several key attributes of the read
        record:
          - fs: The sampling frequency of the record.
          - units: The units for each channel.
          - sig_name: The signal name for each channel.
          - comments: Any comments written in the header.

    Notes
    -----
    If a signal range or channel selection is specified when calling
    this function, the resulting attributes of the returned object will
    be set to reflect the section of the record that is actually read,
    rather than necessarily the entire record. For example, if
    `channels=[0, 1, 2]` is specified when reading a 12 channel record,
    the 'n_sig' attribute will be 3, not 12.

    The `rdrecord` function is the base function upon which this one is
    built. It returns all attributes present, along with the signals, as
    attributes in a `Record` object. The function, along with the
    returned data type, has more options than `rdsamp` for users who
    wish to more directly manipulate WFDB content.

    Examples
    --------
    >>> signals, fields = wfdb.rdsamp('sample-data/test01_00s',
                                      sampfrom=800,
                                      channel =[1,3])

    """
    if (pn_dir is not None) and ('.' not in pn_dir):
        dir_list = pn_dir.split('/')
        pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]), *dir_list[1:])

    record = rdrecord(record_name=record_name, sampfrom=sampfrom,
                      sampto=sampto, channels=channels, physical=True,
                      pn_dir=pn_dir, m2s=True, return_res=return_res,
                      channel_names=channel_names, warn_empty=warn_empty)

    signals = record.p_signal
    fields = {}
    for field in ['fs','sig_len', 'n_sig', 'base_date', 'base_time',
                  'units','sig_name', 'comments']:
        fields[field] = getattr(record, field)

    return signals, fields


def sampfreq(record_name, pn_dir=None):
    """
    Read a WFDB header file and return the sampling frequency of
    each of the signals in the record.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.

    Returns
    -------
    N/A

    Examples
    --------
    >>> wfdb.sampfreq('sample-data/test01_00s')
    >>> ECG 1    500
    >>> ECG 2    500
    >>> ECG 3    500
    >>> ECG 4    500

    """
    if (pn_dir is not None) and ('.' not in pn_dir):
        dir_list = pn_dir.split('/')
        pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]),
                                *dir_list[1:])

    record = rdheader(record_name, pn_dir=pn_dir)
    samps_per_frame = [record.fs*samp for samp in record.samps_per_frame]
    sig_name = record.sig_name

    for sig,samp in zip(sig_name, samps_per_frame):
        print('{}\t{}'.format(sig,samp))


def signame(record_name, pn_dir=None, sig_nums=[]):
    """
    Read a WFDB record file and return the signal names.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    sig_nums : list, optional
        A list of the signal numbers to be outputted.

    Returns
    -------
    N/A

    Examples
    --------
    >>> wfdb.signame('sample-data/test01_00s')
    >>> ECG 1
    >>> ECG 2
    >>> ECG 3
    >>> ECG 4

    >>> wfdb.signame('sample-data/test01_00s', sig_nums=[1,3])
    >>> ECG 2
    >>> ECG 4

    """
    if (pn_dir is not None) and ('.' not in pn_dir):
        dir_list = pn_dir.split('/')
        pn_dir = posixpath.join(dir_list[0], get_version(dir_list[0]),
                                 *dir_list[1:])

    record = rdheader(record_name, pn_dir=pn_dir)
    if len(sig_nums) > 0:
        for n in sig_nums:
            try:
                print(record.sig_name[n])
            except IndexError:
                raise Exception('sig_nums value {} out of range'.format(n))
    else:
        print(*record.sig_name, sep='\n')


def _get_wanted_channels(wanted_sig_names, record_sig_names, pad=False):
    """
    Given some wanted signal names, and the signal names contained in a
    record, return the indices of the record channels that intersect.

    Parameters
    ----------
    wanted_sig_names : list
        List of desired signal name strings
    record_sig_names : list
        List of signal names for a single record
    pad : bool, optional
        Whether the output channels is to always have the same number
        of elements and the wanted channels. If True, pads missing
        signals with None.

    Returns
    -------
    list
        The indices of the wanted record channel names.

    """
    if pad:
        return [record_sig_names.index(s) if s in record_sig_names else None for s in wanted_sig_names]
    else:
        return [record_sig_names.index(s) for s in wanted_sig_names if s in record_sig_names]


#------------------- /Reading Records -------------------#


def wrsamp(record_name, fs, units, sig_name, p_signal=None, d_signal=None,
           fmt=None, adc_gain=None, baseline=None, comments=None,
           base_time=None, base_date=None, write_dir=''):
    """
    Write a single segment WFDB record, creating a WFDB header file and any
    associated dat files.

    Parameters
    ----------
    record_name : str
        The string name of the WFDB record to be written (without any file
        extensions). Must not contain any "." since this would indicate an
        EDF file which is not compatible at this point.
    fs : int, float
        The sampling frequency of the record.
    units : list
        A list of strings giving the units of each signal channel.
    sig_name : list, str
        A list of strings giving the signal name of each signal channel.
    p_signal : ndarray, optional
        An (MxN) 2d numpy array, where M is the signal length. Gives the
        physical signal values intended to be written. Either p_signal or
        d_signal must be set, but not both. If p_signal is set, this method will
        use it to perform analogue-digital conversion, writing the resultant
        digital values to the dat file(s). If fmt is set, gain and baseline must
        be set or unset together. If fmt is unset, gain and baseline must both
        be unset.
    d_signal : ndarray, optional
        An (MxN) 2d numpy array, where M is the signal length. Gives the
        digital signal values intended to be directly written to the dat
        file(s). The dtype must be an integer type. Either p_signal or d_signal
        must be set, but not both. In addition, if d_signal is set, fmt, gain
        and baseline must also all be set.
    fmt : list, optional
        A list of strings giving the WFDB format of each file used to store each
        channel. Accepted formats are: '80','212','16','24', and '32'. There are
        other WFDB formats as specified by:
        https://www.physionet.org/physiotools/wag/signal-5.htm
        but this library will not write (though it will read) those file types.
    adc_gain : list, optional
        A list of numbers specifying the ADC gain.
    baseline : list, optional
        A list of integers specifying the digital baseline.
    comments : list, optional
        A list of string comments to be written to the header file.
    base_time : str, optional
        A string of the record's start time in 24h 'HH:MM:SS(.ms)' format.
    base_date : str, optional
        A string of the record's start date in 'DD/MM/YYYY' format.
    write_dir : str, optional
        The directory in which to write the files.

    Returns
    -------
    N/A

    Notes
    -----
    This is a gateway function, written as a simple method to write WFDB record
    files using the most common parameters. Therefore not all WFDB fields can be
    set via this function.

    For more control over attributes, create a `Record` object, manually set its
    attributes, and call its `wrsamp` instance method. If you choose this more
    advanced method, see also the `set_defaults`, `set_d_features`, and
    `set_p_features` instance methods to help populate attributes.

    Examples
    --------
    >>> # Read part of a record from Physionet
    >>> signals, fields = wfdb.rdsamp('a103l', sampfrom=50000, channels=[0,1],
                                      pn_dir='challenge-2015/training')
    >>> # Write a local WFDB record (manually inserting fields)
    >>> wfdb.wrsamp('ecgrecord', fs = 250, units=['mV', 'mV'],
                    sig_name=['I', 'II'], p_signal=signals, fmt=['16', '16'])

    """
    # Check for valid record name
    if '.' in record_name: 
        raise Exception("Record name must not contain '.'")
    # Check input field combinations
    if p_signal is not None and d_signal is not None:
        raise Exception('Must only give one of the inputs: p_signal or d_signal')
    if d_signal is not None:
        if fmt is None or adc_gain is None or baseline is None:
            raise Exception("When using d_signal, must also specify 'fmt', 'gain', and 'baseline' fields.")
    # Depending on whether d_signal or p_signal was used, set other
    # required features.
    if p_signal is not None:
        # Create the Record object
        record = Record(record_name=record_name, p_signal=p_signal, fs=fs,
                        fmt=fmt, units=units, sig_name=sig_name,
                        adc_gain=adc_gain, baseline=baseline,
                        comments=comments, base_time=base_time,
                        base_date=base_date)
        # Compute optimal fields to store the digital signal, carry out adc,
        # and set the fields.
        record.set_d_features(do_adc=1)
    else:
        # Create the Record object
        record = Record(record_name=record_name, d_signal=d_signal, fs=fs,
                        fmt=fmt, units=units, sig_name=sig_name,
                        adc_gain=adc_gain, baseline=baseline,
                        comments=comments, base_time=base_time,
                        base_date=base_date)
        # Use d_signal to set the fields directly
        record.set_d_features()

    # Set default values of any missing field dependencies
    record.set_defaults()
    # Write the record files - header and associated dat
    record.wrsamp(write_dir=write_dir)


def is_monotonic(full_list):
    """
    Determine whether elements in a list are monotonic. ie. unique
    elements are clustered together.

    ie. [5,5,3,4] is, [5,3,5] is not.

    Parameters
    ----------
    full_list : list
        The input elements used for the analysis.

    Returns
    -------
    bool
        Whether the elements are monotonic (True) or not (False).

    """
    prev_elements = set({full_list[0]})
    prev_item = full_list[0]

    for item in full_list:
        if item != prev_item:
            if item in prev_elements:
                return False
            prev_item = item
            prev_elements.add(item)

    return True


def dl_database(db_dir, dl_dir, records='all', annotators='all',
                keep_subdirs=True, overwrite=False):
    """
    Download WFDB record (and optionally annotation) files from a
    PhysioNet database. The database must contain a 'RECORDS' file in
    its base directory which lists its WFDB records.

    Parameters
    ----------
    db_dir : str
        The PhysioNet database directory to download. eg. For database:
        'http://physionet.org/content/mitdb/', db_dir='mitdb'.
    dl_dir : str
        The full local directory path in which to download the files.
    records : list, 'all', optional
        A list of strings specifying the WFDB records to download. Leave
        as 'all' to download all records listed in the database's
        RECORDS file.
        eg. records=['test01_00s', test02_45s] for database:
        https://physionet.org/content/macecgdb/
    annotators : list, 'all', None, optional
        A list of strings specifying the WFDB annotation file types to
        download along with the record files. Is either None to skip
        downloading any annotations, 'all' to download all annotation
        types as specified by the ANNOTATORS file, or a list of strings
        which each specify an annotation extension.
        eg. annotators = ['anI'] for database:
        https://physionet.org/content/prcp/
    keep_subdirs : bool, optional
        Whether to keep the relative subdirectories of downloaded files
        as they are organized in PhysioNet (True), or to download all
        files into the same base directory (False).
    overwrite : bool, optional
        If True, all files will be redownloaded regardless. If False,
        existing files with the same name and relative subdirectory will
        be checked. If the local file is the same size as the online
        file, the download is skipped. If the local file is larger, it
        will be deleted and the file will be redownloaded. If the local
        file is smaller, the file will be assumed to be partially
        downloaded and the remaining bytes will be downloaded and
        appended.

    Returns
    -------
    N/A

    Examples
    --------
    >>> wfdb.dl_database('ahadb', os.getcwd())

    """
    # Full url PhysioNet database
    if '/' in db_dir:
        dir_list = db_dir.split('/')
        db_dir = posixpath.join(dir_list[0], get_version(dir_list[0]), *dir_list[1:])
    else:
        db_dir = posixpath.join(db_dir, get_version(db_dir))
    db_url = posixpath.join(download.PN_CONTENT_URL, db_dir) + '/'
    # Check if the database is valid
    r = requests.get(db_url)
    r.raise_for_status()

    # Get the list of records
    record_list = download.get_record_list(db_dir, records)
    # Get the annotator extensions
    annotators = download.get_annotators(db_dir, annotators)

    # All files to download (relative to the database's home directory)
    all_files = []
    nested_records = []

    for rec in record_list:
        print('Generating record list for: ' + rec)
        # Check out whether each record is in MIT or EDF format
        if rec.endswith('.edf'):
            all_files.append(rec)
        else:
            # May be pointing to directory
            if rec.endswith(os.sep):
                nested_records += [posixpath.join(rec, sr) for sr in download.get_record_list(posixpath.join(db_dir, rec))]
            else:
                nested_records.append(rec)

    for rec in nested_records:
        print('Generating list of all files for: ' + rec)
        # If MIT format, have to figure out all associated files
        all_files.append(rec+'.hea')
        dir_name, base_rec_name = os.path.split(rec)
        record = rdheader(base_rec_name, pn_dir=posixpath.join(db_dir, dir_name))

        # Single segment record
        if isinstance(record, Record):
            # Add all dat files of the segment
            for file in (record.file_name if record.file_name else []):
                all_files.append(posixpath.join(dir_name, file))

        # Multi segment record
        else:
            for seg in record.seg_name:
                # Skip empty segments
                if seg == '~':
                    continue
                # Add the header
                all_files.append(posixpath.join(dir_name, seg+'.hea'))
                # Layout specifier has no dat files
                if seg.endswith('_layout'):
                    continue
                # Add all dat files of the segment
                rec_seg = rdheader(seg, pn_dir=posixpath.join(db_dir, dir_name))
                for file in rec_seg.file_name:
                    all_files.append(posixpath.join(dir_name, file))

        # Check whether the record has any requested annotation files
        if annotators is not None:
            for a in annotators:
                ann_file = rec+'.'+a
                url = posixpath.join(download.config.db_index_url, db_dir, ann_file)
                rh = requests.head(url)

                if rh.status_code != 404:
                    all_files.append(ann_file)

    dl_inputs = [(os.path.split(file)[1], os.path.split(file)[0], db_dir, dl_dir, keep_subdirs, overwrite) for file in all_files]

    # Make any required local directories
    download.make_local_dirs(dl_dir, dl_inputs, keep_subdirs)

    print('Downloading files...')
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.Pool(processes=2)
    pool.map(download.dl_pn_file, dl_inputs)
    print('Finished downloading files')

    return
