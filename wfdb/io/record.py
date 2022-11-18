import datetime
import multiprocessing.dummy
import posixpath
import os
import re

import numpy as np
import pandas as pd

from wfdb.io import _header
from wfdb.io import _signal
from wfdb.io import _url
from wfdb.io import download
from wfdb.io import header
from wfdb.io import util


# -------------- WFDB Signal Calibration and Classification ---------- #


# Unit scales used for default display scales. The unit scale that the
# class should measure. 'No Unit' will also be allowed in all cases.
# * Will it always be 1?
unit_scale = {
    "voltage": ["pV", "nV", "uV", "mV", "V", "kV"],
    "temperature": ["C", "F"],
    "pressure": ["mmHg"],
    "no_unit": ["NU"],
    "percentage": ["%"],
    "heart_rate": ["bpm"],
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
    index=[
        "bp",
        "co2",
        "co",
        "ecg",
        "eeg",
        "emg",
        "eog",
        "hr",
        "mmg",
        "o2",
        "pleth",
        "resp",
        "scg",
        "stat",
        "st",
        "temp",
        "unknown",
    ],
    columns=["description", "unit_scale", "signal_names"],
    data=[
        ["Blood Pressure", "pressure", ["bp", "abp", "pap", "cvp"]],  # bp
        ["Carbon Dioxide", "percentage", ["co2", "pco2"]],  # co2
        ["Carbon Monoxide", "percentage", ["co"]],  # co
        [
            "Electrocardiogram",
            "voltage",
            ["i", "ii", "iii", "iv", "v", "avr"],
        ],  # ecg
        ["Electroencephalogram", "voltage", ["eeg"]],  # eeg
        ["Electromyograph", "voltage", ["emg"]],  # emg
        ["Electrooculograph", "voltage", ["eog"]],  # eog
        ["Heart Rate", "heart_rate", ["hr"]],  # hr
        ["Magnetomyograph", "voltage", ["mmg"]],  # mmg
        ["Oxygen", "percentage", ["o2", "spo2"]],  # o2
        ["Plethysmograph", "pressure", ["pleth"]],  # pleth
        ["Respiration", "no_unit", ["resp"]],  # resp
        ["Seismocardiogram", "no_unit", ["scg"]],  # scg
        ["Status", "no_unit", ["stat", "status"]],  # stat
        ["ST Segment", "", ["st"]],  # st. This is not a signal?
        ["Temperature", "temperature", ["temp"]],  # temp
        ["Unknown Class", "no_unit", []],  # unknown. special class.
    ],
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
    "a": "uV",
    "abdomen": "uV",
    "abdo": "V",
    "abp": "mmHg",
    "airflow": "V",
    "ann": "units",
    "art": "mmHg",
    "atip": "mV",
    "av": "mV",
    "bp": "mmHg",
    "c": "uV",
    "c.o.": "lpm",
    "co": "Lpm",
    "cs": "mV",
    "cvp": "mmHg",
    "direct": "uV",
    "ecg": "mV",
    "edr": "units",
    "eeg": "mV",
    "emg": "mV",
    "eog": "mV",
    "event": "mV",
    "f": "uV",
    "fecg": "mV",
    "fhr": "bpm",
    "foobar": "mmHg",
    "hr": "bpm",
    "hva": "mV",
    "i": "mV",
    "ibp": "mmHg",
    "mcl": "mV",
    "nbp": "mmHg",
    "o": "uV",
    "p": "mmHg",
    "pap": "mmHg",
    "pawp": "mmHg",
    "pcg": "mV",
    "pleth": "mV",
    "pr": "bpm",
    "pulse": "bpm",
    "record": "mV",
    "resp": "l",
    "sao2": "%",
    "so2": "%",
    "spo2": "%",
    "sv": "ml",
    "t": "uV",
    "tblood": "degC",
    "temp": "degC",
    "thorax": "mV",
    "thor": "V",
    "v": "mV",
    "uc": "nd",
    "vtip": "mV",
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
    base_time : datetime.time, optional
        The time of day at the beginning of the record.
    base_date : datetime.date, optional
        The date at the beginning of the record.
    base_datetime : datetime.datetime, optional
        The date and time at the beginning of the record, equivalent to
        `datetime.combine(base_date, base_time)`.
    comments : list, optional
        A list of string comments to be written to the header file.
    sig_name : str, optional
        A list of strings giving the signal name of each signal channel.

    """

    # The base WFDB class extended by the Record and MultiRecord classes.
    def __init__(
        self,
        record_name=None,
        n_sig=None,
        fs=None,
        counter_freq=None,
        base_counter=None,
        sig_len=None,
        base_time=None,
        base_date=None,
        base_datetime=None,
        comments=None,
        sig_name=None,
    ):
        self.record_name = record_name
        self.n_sig = n_sig
        self.fs = fs
        self.counter_freq = counter_freq
        self.base_counter = base_counter
        self.sig_len = sig_len
        if base_datetime is not None:
            if base_time is not None:
                raise TypeError(
                    "cannot specify both base_time and base_datetime"
                )
            if base_date is not None:
                raise TypeError(
                    "cannot specify both base_date and base_datetime"
                )
            self.base_datetime = base_datetime
        else:
            self.base_time = base_time
            self.base_date = base_date
        self.comments = comments
        self.sig_name = sig_name

    @property
    def base_datetime(self):
        if self.base_date is None or self.base_time is None:
            return None
        else:
            return datetime.datetime.combine(
                date=self.base_date, time=self.base_time
            )

    @base_datetime.setter
    def base_datetime(self, value):
        if value is None:
            self.base_date = None
            self.base_time = None
        elif isinstance(value, datetime.datetime) and value.tzinfo is None:
            self.base_date = value.date()
            self.base_time = value.time()
        else:
            raise TypeError(f"invalid base_datetime value: {value!r}")

    def get_frame_number(self, time_value):
        """
        Convert a time value to a frame number.

        A time value may be specified as:
        - An integer or floating-point number, representing the number of
          WFDB frames elapsed from the start of the record.
        - A `datetime.timedelta` object, representing elapsed time from the
          start of the record.
        - A `datetime.datetime` object, representing an absolute date and
          time (if the record starting time is known.)

        Note that this function may return a value that is less than zero
        or greater than the actual length of the record.

        Parameters
        ----------
        time_value : number or timedelta or datetime
            A time value.

        Returns
        -------
        frame_number : float
            Frame number (possibly a fractional frame number).

        """
        if hasattr(time_value, "__float__"):
            return float(time_value)

        if isinstance(time_value, datetime.datetime):
            if not self.base_datetime:
                raise ValueError(
                    "base_datetime is unknown; cannot convert absolute "
                    "date/time to a frame number"
                )
            time_value -= self.base_datetime

        if isinstance(time_value, datetime.timedelta):
            return time_value.total_seconds() * self.fs

        raise TypeError(f"invalid time value: {time_value!r}")

    def get_elapsed_time(self, time_value):
        """
        Convert a time value to an elapsed time in seconds.

        A time value may be specified as:
        - An integer or floating-point number, representing the number of
          WFDB frames elapsed from the start of the record.
        - A `datetime.timedelta` object, representing elapsed time from the
          start of the record.
        - A `datetime.datetime` object, representing an absolute date and
          time (if the record starting time is known.)

        Parameters
        ----------
        time_value : number or timedelta or datetime
            A time value.

        Returns
        -------
        elapsed_time : timedelta
            Elapsed time from the start of the record.

        """
        time_value = self.get_frame_number(time_value)
        return datetime.timedelta(seconds=time_value / self.fs)

    def get_absolute_time(self, time_value):
        """
        Convert a time value to an absolute date and time.

        A time value may be specified as:
        - An integer or floating-point number, representing the number of
          WFDB frames elapsed from the start of the record.
        - A `datetime.timedelta` object, representing elapsed time from the
          start of the record.
        - A `datetime.datetime` object, representing an absolute date and
          time (if the record starting time is known.)

        Parameters
        ----------
        time_value : number or timedelta or datetime
            A time value.

        Returns
        -------
        absolute_time : datetime
            Absolute date and time.

        """
        time_value = self.get_elapsed_time(time_value)
        if not self.base_datetime:
            raise ValueError(
                "base_datetime is unknown; cannot convert frame number "
                "to an absolute date/time"
            )
        return time_value + self.base_datetime

    def check_field(self, field, required_channels="all"):
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
            raise Exception("Missing field required: %s" % field)

        # We should have a list specifying these automatically.

        # Whether the item should be a list. Watch out for required_channels for `segments`
        expect_list = True if field in LIST_FIELDS else False

        # Check the type of the field (and of its elements if it should
        # be a list)
        _check_item_type(
            item,
            field_name=field,
            allowed_types=ALLOWED_TYPES[field],
            expect_list=expect_list,
            required_channels=required_channels,
        )

        # Individual specific field checks
        if field in ["d_signal", "p_signal"]:
            check_np_array(
                item=item,
                field_name=field,
                ndim=2,
                parent_class=(
                    lambda f: np.integer if f == "d_signal" else np.floating
                )(field),
            )
        elif field in ["e_d_signal", "e_p_signal"]:
            for ch in range(len(item)):
                check_np_array(
                    item=item[ch],
                    field_name=field,
                    ndim=1,
                    parent_class=(
                        lambda f: np.integer
                        if f == "e_d_signal"
                        else np.floating
                    )(field),
                    channel_num=ch,
                )

        # Record specification fields
        elif field == "record_name":
            # Allow letters, digits, hyphens, and underscores.
            accepted_string = re.match(r"[-\w]+", self.record_name)
            if (
                not accepted_string
                or accepted_string.string != self.record_name
            ):
                raise ValueError(
                    "record_name must only comprise of letters, digits, hyphens, and underscores."
                )
        elif field == "n_seg":
            if self.n_seg <= 0:
                raise ValueError("n_seg must be a positive integer")
        elif field == "n_sig":
            if self.n_sig <= 0:
                raise ValueError("n_sig must be a positive integer")
        elif field == "fs":
            if self.fs <= 0:
                raise ValueError("fs must be a positive number")
        elif field == "counter_freq":
            if self.counter_freq <= 0:
                raise ValueError("counter_freq must be a positive number")
        elif field == "base_counter":
            if self.base_counter <= 0:
                raise ValueError("base_counter must be a positive number")
        elif field == "sig_len":
            if self.sig_len < 0:
                raise ValueError("sig_len must be a non-negative integer")

        # Signal specification fields
        elif field in _header.SIGNAL_SPECS.index:
            if required_channels == "all":
                required_channels = range(len(item))

            for ch in range(len(item)):
                # If the element is allowed to be None
                if ch not in required_channels:
                    if item[ch] is None:
                        continue

                if field == "file_name":
                    # Check for file_name characters
                    accepted_string = re.match(r"[-\w]+\.?[\w]+", item[ch])
                    if (
                        not accepted_string
                        or accepted_string.string != item[ch]
                    ):
                        raise ValueError(
                            "File names should only contain alphanumerics, hyphens, and an extension. eg. record-100.dat"
                        )
                    # Check that dat files are grouped together
                    if not util.is_monotonic(self.file_name):
                        raise ValueError(
                            "Signals in a record that share a given file must be consecutive."
                        )
                elif field == "fmt":
                    if item[ch] not in _signal.DAT_FMTS:
                        raise ValueError(
                            "File formats must be valid WFDB dat formats:",
                            _signal.DAT_FMTS,
                        )
                elif field == "samps_per_frame":
                    if item[ch] < 1:
                        raise ValueError(
                            "samps_per_frame values must be positive integers"
                        )
                elif field == "skew":
                    if item[ch] < 0:
                        raise ValueError(
                            "skew values must be non-negative integers"
                        )
                elif field == "byte_offset":
                    if item[ch] < 0:
                        raise ValueError(
                            "byte_offset values must be non-negative integers"
                        )
                elif field == "adc_gain":
                    if item[ch] <= 0:
                        raise ValueError("adc_gain values must be positive")
                elif field == "baseline":
                    # Original WFDB library 10.5.24 only has 4 bytes for baseline.
                    if item[ch] < -2147483648 or item[ch] > 2147483648:
                        raise ValueError(
                            "baseline values must be between -2147483648 (-2^31) and 2147483647 (2^31 -1)"
                        )
                elif field == "units":
                    if re.search(r"\s", item[ch]):
                        raise ValueError(
                            "units strings may not contain whitespaces."
                        )
                elif field == "adc_res":
                    if item[ch] < 0:
                        raise ValueError(
                            "adc_res values must be non-negative integers"
                        )
                elif field == "block_size":
                    if item[ch] < 0:
                        raise ValueError(
                            "block_size values must be non-negative integers"
                        )
                elif field == "sig_name":
                    if item[ch][:1].isspace() or item[ch][-1:].isspace():
                        raise ValueError(
                            "sig_name strings may not begin or end with "
                            "whitespace."
                        )
                    if re.search(r"[\x00-\x1f\x7f-\x9f]", item[ch]):
                        raise ValueError(
                            "sig_name strings may not contain "
                            "control characters."
                        )
                    if len(set(item)) != len(item):
                        raise ValueError("sig_name strings must be unique.")

        # Segment specification fields and comments
        elif field in _header.SEGMENT_SPECS.index:
            for ch in range(len(item)):
                if field == "seg_name":
                    # Segment names must be alphanumerics or just a single '~'
                    if item[ch] == "~":
                        continue
                    accepted_string = re.match(r"[-\w]+", item[ch])
                    if (
                        not accepted_string
                        or accepted_string.string != item[ch]
                    ):
                        raise ValueError(
                            "Non-null segment names may only contain alphanumerics and dashes. Null segment names must be set to '~'"
                        )
                elif field == "seg_len":
                    # For records with more than 1 segment, the first
                    # segment may be the layout specification segment
                    # with a length of 0
                    min_len = 0 if ch == 0 else 1
                    if item[ch] < min_len:
                        raise ValueError(
                            "seg_len values must be positive integers. Only seg_len[0] may be 0 to indicate a layout segment"
                        )
                # Comment field
                elif field == "comments":
                    if item[ch].startswith("#"):
                        print(
                            "Note: comment strings do not need to begin with '#'. This library adds them automatically."
                        )
                    if re.search("[\t\n\r\f\v]", item[ch]):
                        raise ValueError(
                            "comments may not contain tabs or newlines (they may contain spaces and underscores)."
                        )

    def check_read_inputs(
        self, sampfrom, sampto, channels, physical, smooth_frames, return_res
    ):
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
        if not hasattr(sampfrom, "__index__"):
            raise TypeError("sampfrom must be an integer")
        if not hasattr(sampto, "__index__"):
            raise TypeError("sampto must be an integer")
        if not isinstance(channels, list):
            raise TypeError("channels must be a list of integers")

        # Duration Ranges
        if sampfrom < 0:
            raise ValueError("sampfrom must be a non-negative integer")
        if sampfrom > self.sig_len:
            raise ValueError("sampfrom must be shorter than the signal length")
        if sampto < 0:
            raise ValueError("sampto must be a non-negative integer")
        if sampto > self.sig_len:
            raise ValueError("sampto must be shorter than the signal length")
        if sampto <= sampfrom:
            raise ValueError("sampto must be greater than sampfrom")

        # Channel Ranges
        if len(channels):
            if min(channels) < 0:
                raise ValueError(
                    "Input channels must all be non-negative integers"
                )
            if max(channels) > self.n_sig - 1:
                raise ValueError(
                    "Input channels must all be lower than the total number of channels"
                )

        if return_res not in [64, 32, 16, 8]:
            raise ValueError(
                "return_res must be one of the following: 64, 32, 16, 8"
            )
        if physical and return_res == 8:
            raise ValueError(
                "return_res must be one of the following when physical is True: 64, 32, 16"
            )

    def _adjust_datetime(self, sampfrom: int):
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
                self.base_datetime += datetime.timedelta(seconds=dt_seconds)
            # We can calculate the time even if there is no date
            elif self.base_time:
                tmp_datetime = datetime.datetime.combine(
                    datetime.datetime.today().date(), self.base_time
                )
                self.base_time = (
                    tmp_datetime + datetime.timedelta(seconds=dt_seconds)
                ).time()
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
    base_time : datetime.time, optional
        The time of day at the beginning of the record.
    base_date : datetime.date, optional
        The date at the beginning of the record.
    base_datetime : datetime.datetime, optional
        The date and time at the beginning of the record, equivalent to
        `datetime.combine(base_date, base_time)`.
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

    def __init__(
        self,
        p_signal=None,
        d_signal=None,
        e_p_signal=None,
        e_d_signal=None,
        record_name=None,
        n_sig=None,
        fs=None,
        counter_freq=None,
        base_counter=None,
        sig_len=None,
        base_time=None,
        base_date=None,
        base_datetime=None,
        file_name=None,
        fmt=None,
        samps_per_frame=None,
        skew=None,
        byte_offset=None,
        adc_gain=None,
        baseline=None,
        units=None,
        adc_res=None,
        adc_zero=None,
        init_value=None,
        checksum=None,
        block_size=None,
        sig_name=None,
        comments=None,
    ):

        # Note the lack of the 'n_seg' field. Single segment records cannot
        # have this field. Even n_seg = 1 makes the header a multi-segment
        # header.
        super(Record, self).__init__(
            record_name=record_name,
            n_sig=n_sig,
            fs=fs,
            counter_freq=counter_freq,
            base_counter=base_counter,
            sig_len=sig_len,
            base_time=base_time,
            base_date=base_date,
            base_datetime=base_datetime,
            comments=comments,
            sig_name=sig_name,
        )

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
                print("Attributes members mismatch.")
            return False

        for k in att1.keys():

            v1 = att1[k]
            v2 = att2[k]

            if type(v1) != type(v2):
                if verbose:
                    print("Mismatch in attribute: %s" % k, v1, v2)
                return False

            if isinstance(v1, np.ndarray):
                # Necessary for nans
                np.testing.assert_array_equal(v1, v2)
            elif (
                isinstance(v1, list)
                and len(v1) == len(v2)
                and all(isinstance(e, np.ndarray) for e in v1)
            ):
                for (e1, e2) in zip(v1, v2):
                    np.testing.assert_array_equal(e1, e2)
            else:
                if v1 != v2:
                    if verbose:
                        print("Mismatch in attribute: %s" % k, v1, v2)
                    return False

        return True

    def wrsamp(self, expanded=False, write_dir=""):
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
        # Update the checksum field (except for channels that did not have
        # a checksum to begin with, or where the checksum was already
        # valid.)
        if self.checksum is not None:
            checksums = self.calc_checksum(expanded=expanded)
            for ch, old_val in enumerate(self.checksum):
                if old_val is None or (checksums[ch] - old_val) % 65536 == 0:
                    checksums[ch] = old_val
            self.checksum = checksums

        # Perform field validity and cohesion checks, and write the
        # header file.
        self.wrheader(write_dir=write_dir, expanded=expanded)
        if self.n_sig > 0:
            # Perform signal validity and cohesion checks, and write the
            # associated dat files.
            self.wr_dats(expanded=expanded, write_dir=write_dir)

    def _arrange_fields(self, channels, sampfrom, smooth_frames):
        """
        Arrange/edit object fields to reflect user channel and/or signal
        range input.

        Parameters
        ----------
        channels : list
            List of channel numbers specified.
        sampfrom : int
            Starting sample number read.
        smooth_frames : bool
            Whether to convert the expanded signal array (e_d_signal) into
            a smooth signal array (d_signal).

        Returns
        -------
        N/A

        """
        # Rearrange signal specification fields
        for field in _header.SIGNAL_SPECS.index:
            item = getattr(self, field)
            setattr(self, field, [item[c] for c in channels])

        # Expanded signals - multiple samples per frame.
        if not smooth_frames:
            # Checksum and init_value to be updated if present
            # unless the whole signal length was input
            if self.sig_len != int(
                len(self.e_d_signal[0]) / self.samps_per_frame[0]
            ):
                self.checksum = self.calc_checksum(True)
                self.init_value = [s[0] for s in self.e_d_signal]

            self.n_sig = len(channels)
            self.sig_len = int(
                len(self.e_d_signal[0]) / self.samps_per_frame[0]
            )

        # MxN numpy array d_signal
        else:
            self.d_signal = self.smooth_frames("digital")
            self.e_d_signal = None

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a dataframe containing the data from this record.


        Returns
        -------
        A dataframe, with sig_name in the columns. The index is a DatetimeIndex
        if both base_date and base_time were set, otherwise a TimedeltaIndex.
        """
        if self.base_datetime is not None:
            index = pd.date_range(
                start=self.base_datetime,
                periods=self.sig_len,
                freq=pd.Timedelta(seconds=1 / self.fs),
            )
        else:
            index = pd.timedelta_range(
                start=pd.Timedelta(0),
                periods=self.sig_len,
                freq=pd.Timedelta(seconds=1 / self.fs),
            )

        if self.p_signal is not None:
            data = self.p_signal
        elif self.d_signal is not None:
            data = self.d_signal
        elif self.e_p_signal is not None:
            data = np.array(self.e_p_signal).T
        elif self.e_d_signal is not None:
            data = np.array(self.e_d_signal).T
        else:
            raise ValueError("No signal in record.")

        return pd.DataFrame(data=data, index=index, columns=self.sig_name)


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
    base_time : datetime.time, optional
        The time of day at the beginning of the record.
    base_date : datetime.date, optional
        The date at the beginning of the record.
    base_datetime : datetime.datetime, optional
        The date and time at the beginning of the record, equivalent to
        `datetime.combine(base_date, base_time)`.
    seg_name : str, optional
        The name of the segment.
    seg_len : List[int], optional
        The length of each segment.
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

    def __init__(
        self,
        segments=None,
        layout=None,
        record_name=None,
        n_sig=None,
        fs=None,
        counter_freq=None,
        base_counter=None,
        sig_len=None,
        base_time=None,
        base_date=None,
        base_datetime=None,
        seg_name=None,
        seg_len=None,
        comments=None,
        sig_name=None,
        sig_segments=None,
    ):

        super(MultiRecord, self).__init__(
            record_name=record_name,
            n_sig=n_sig,
            fs=fs,
            counter_freq=counter_freq,
            base_counter=base_counter,
            sig_len=sig_len,
            base_time=base_time,
            base_date=base_date,
            base_datetime=base_datetime,
            comments=comments,
            sig_name=sig_name,
        )

        self.layout = layout
        self.segments = segments
        self.seg_name = seg_name
        self.seg_len = seg_len
        self.sig_segments = sig_segments

        if segments:
            self.n_seg = len(segments)
            if not seg_len:
                self.seg_len = [segment.sig_len for segment in segments]

    def wrsamp(self, write_dir=""):
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

        for seg_num, segment in enumerate(self.segments):

            # If segment 0 is a layout specification record, check that its file names are all == '~''
            if seg_num == 0 and self.seg_len[0] == 0:
                for file_name in segment.file_name:
                    if file_name != "~":
                        raise ValueError(
                            "Layout specification records must have all file_names named '~'"
                        )

            # Sampling frequencies must all match the one in the master header
            if segment.fs != self.fs:
                raise ValueError(
                    "The 'fs' in each segment must match the overall record's 'fs'"
                )

            # Check the signal length of the segment against the corresponding seg_len field
            if segment.sig_len != self.seg_len[seg_num]:
                raise ValueError(
                    f"The signal length of segment {seg_num} does not match the corresponding segment length"
                )

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
        if self.layout == "fixed":
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
            seg_numbers.append(
                [sampto <= cs for cs in cumsumlengths].index(True)
            )

        # Add 1 for variable layout records
        seg_numbers = list(np.add(seg_numbers, startseg))

        # Obtain the sampfrom and sampto to read for each segment
        if seg_numbers[1] == seg_numbers[0]:
            # Only one segment to read
            seg_numbers = [seg_numbers[0]]
            # The segment's first sample number relative to the entire record
            segstartsamp = sum(self.seg_len[0 : seg_numbers[0]])
            readsamps = [[sampfrom - segstartsamp, sampto - segstartsamp]]

        else:
            # More than one segment to read
            seg_numbers = list(range(seg_numbers[0], seg_numbers[1] + 1))
            readsamps = [[0, self.seg_len[s]] for s in seg_numbers]

            # Starting sample for first segment.
            readsamps[0][0] = (
                sampfrom - ([0] + cumsumlengths)[seg_numbers[0] - startseg]
            )

            # End sample for last segment
            readsamps[-1][1] = (
                sampto - ([0] + cumsumlengths)[seg_numbers[-1] - startseg]
            )

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
        if self.layout == "fixed":
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
                if self.seg_name[seg_numbers[i]] == "~":
                    required_channels.append([])
                else:
                    # Get the signal names of the current segment
                    s_sig_names = rdheader(
                        os.path.join(dir_name, self.seg_name[seg_numbers[i]]),
                        pn_dir=pn_dir,
                    ).sig_name
                    required_channels.append(
                        _get_wanted_channels(w_sig_names, s_sig_names)
                    )

        return required_channels

    def _arrange_fields(
        self, seg_numbers, seg_ranges, channels, sampfrom=0, force_channels=True
    ):
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
        if self.layout == "fixed":
            self.n_sig = len(channels)
            self.segments = self.segments[seg_numbers[0] : seg_numbers[-1] + 1]
            self.seg_name = self.seg_name[seg_numbers[0] : seg_numbers[-1] + 1]
            self.seg_len = self.seg_len[seg_numbers[0] : seg_numbers[-1] + 1]
        else:
            self.segments = [self.segments[0]] + self.segments[
                seg_numbers[0] : seg_numbers[-1] + 1
            ]
            self.seg_name = [self.seg_name[0]] + self.seg_name[
                seg_numbers[0] : seg_numbers[-1] + 1
            ]
            self.seg_len = [self.seg_len[0]] + self.seg_len[
                seg_numbers[0] : seg_numbers[-1] + 1
            ]

            # Update the layout specification segment. At this point it
            # should match the full original header

            # Have to inspect existing channels of segments; requested
            # input channels will not be enough on its own because not
            # all signals may be present, depending on which section of
            # the signal was read.
            if not force_channels:
                # The desired signal names.
                desired_sig_names = [
                    self.segments[0].sig_name[ch] for ch in channels
                ]
                # Actual contained signal names of individual segments
                # contained_sig_names = [seg.sig_name for seg in self.segments[1:]]
                contained_sig_names = set(
                    [
                        name
                        for seg in self.segments[1:]
                        if seg is not None
                        for name in seg.sig_name
                    ]
                )
                # Remove non-present names. Keep the order.
                sig_name = [
                    name
                    for name in desired_sig_names
                    if name in contained_sig_names
                ]
                # Channel indices to keep for signal specification fields
                channels = [
                    self.segments[0].sig_name.index(name) for name in sig_name
                ]

            # Rearrange signal specification fields
            for field in _header.SIGNAL_SPECS.index:
                item = getattr(self.segments[0], field)
                setattr(self.segments[0], field, [item[c] for c in channels])

            self.segments[0].n_sig = self.n_sig = len(channels)
            if self.n_sig == 0:
                print(
                    "No signals of the desired channels are contained in the specified sample range."
                )

        # Update record specification parameters
        self.sig_len = sum([sr[1] - sr[0] for sr in seg_ranges])
        self.n_seg = len(self.segments)
        self._adjust_datetime(sampfrom=sampfrom)

    def multi_to_single(self, physical, return_res=64, expanded=False):
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
        expanded : bool, optional
            If false, combine the sample data from `p_signal` or `d_signal`
            into a single two-dimensional array. If true, combine the
            sample data from `e_p_signal` or `e_d_signal` into a list of
            one-dimensional arrays.

        Returns
        -------
        record : WFDB Record
            The single segment record created.

        """
        # The fields to transfer to the new object
        fields = self.__dict__.copy()

        # Remove multirecord fields
        for attr in ["segments", "seg_name", "seg_len", "n_seg"]:
            del fields[attr]

        # Figure out single segment fields to set for the new Record
        if self.layout == "fixed":
            # Get the fields from the first segment
            for attr in [
                "fmt",
                "adc_gain",
                "baseline",
                "units",
                "sig_name",
                "samps_per_frame",
            ]:
                fields[attr] = getattr(self.segments[0], attr)
        else:
            # For variable layout records, inspect the segments for the
            # attribute values.

            # Coincidentally, if physical=False, figure out if this
            # conversion can be performed. All signals of the same name
            # must have the same fmt, gain, baseline, and units for all
            # segments.

            # For either physical or digital conversion, all signals
            # of the same name must have the same samps_per_frame,
            # which must match the value in the layout header.

            # The layout header should be updated at this point to
            # reflect channels. We can depend on it for sig_name and
            # samps_per_frame, but not for fmt, adc_gain, units, and
            # baseline.

            # These signal names will be the key
            signal_names = self.segments[0].sig_name
            n_sig = len(signal_names)

            # This will be the field dictionary to copy over.
            reference_fields = {
                "fmt": n_sig * [None],
                "adc_gain": n_sig * [None],
                "baseline": n_sig * [None],
                "units": n_sig * [None],
                "samps_per_frame": self.segments[0].samps_per_frame,
            }

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
                            if field == "samps_per_frame":
                                expected = reference_fields[field][ch]
                                raise ValueError(
                                    f"Incorrect samples per frame "
                                    f"({item_ch} != {expected}) "
                                    f"for signal {signal_names[ch]} "
                                    f"in segment {seg.record_name} "
                                    f"of {self.record_name}"
                                )
                            elif physical:
                                mismatched_fields.append(field)
                            else:
                                raise Exception(
                                    "This variable layout multi-segment record cannot be converted to single segment, in digital format."
                                )
            # Remove mismatched signal fields for physical signals
            for field in set(mismatched_fields):
                del reference_fields[field]
            # At this point, the fields should be set for all channels
            fields.update(reference_fields)
            fields["sig_name"] = signal_names

        # Figure out signal attribute to set, and its dtype.
        if physical:
            if expanded:
                sig_attr = "e_p_signal"
            else:
                sig_attr = "p_signal"
            # Figure out the largest required dtype
            dtype = _signal._np_dtype(return_res, discrete=False)
            nan_vals = np.array([self.n_sig * [np.nan]], dtype=dtype)
        else:
            if expanded:
                sig_attr = "e_d_signal"
            else:
                sig_attr = "d_signal"
            # Figure out the largest required dtype
            dtype = _signal._np_dtype(return_res, discrete=True)
            nan_vals = np.array([_signal._digi_nan(fields["fmt"])], dtype=dtype)

        samps_per_frame = fields["samps_per_frame"]

        # Initialize the full signal array
        if expanded:
            combined_signal = []
            for nan_val, spf in zip(nan_vals[0], samps_per_frame):
                combined_signal.append(np.repeat(nan_val, spf * self.sig_len))
        else:
            combined_signal = np.repeat(nan_vals, self.sig_len, axis=0)

        # Start and end samples in the overall array to place the
        # segment samples into
        start_samps = [0] + list(np.cumsum(self.seg_len)[0:-1])
        end_samps = list(np.cumsum(self.seg_len))

        if self.layout == "fixed":
            # Copy over the signals directly. Recall there are no
            # empty segments in fixed layout records.
            for i in range(self.n_seg):
                signals = getattr(self.segments[i], sig_attr)
                if expanded:
                    for ch in range(self.n_sig):
                        start = start_samps[i] * samps_per_frame[ch]
                        end = end_samps[i] * samps_per_frame[ch]
                        combined_signal[ch][start:end] = signals[ch]
                else:
                    start = start_samps[i]
                    end = end_samps[i]
                    combined_signal[start:end, :] = signals
        else:
            # Copy over the signals into the matching channels
            for i in range(1, self.n_seg):
                seg = self.segments[i]
                if seg is not None:
                    # Get the segment channels to copy over for each
                    # overall channel
                    segment_channels = _get_wanted_channels(
                        fields["sig_name"], seg.sig_name, pad=True
                    )
                    signals = getattr(seg, sig_attr)
                    for ch in range(self.n_sig):
                        # Copy over relevant signal
                        if segment_channels[ch] is not None:
                            if expanded:
                                signal = signals[segment_channels[ch]]
                                start = start_samps[i] * samps_per_frame[ch]
                                end = end_samps[i] * samps_per_frame[ch]
                                combined_signal[ch][start:end] = signal
                            else:
                                signal = signals[:, segment_channels[ch]]
                                start = start_samps[i]
                                end = end_samps[i]
                                combined_signal[start:end, ch] = signal

        # Create the single segment Record object and set attributes
        record = Record()
        for field in fields:
            setattr(record, field, fields[field])
        setattr(record, sig_attr, combined_signal)

        # Use the signal to set record features
        if physical:
            record.set_p_features(expanded=expanded)
        else:
            record.set_d_features(expanded=expanded)

        return record


# ---------------------- Type Specifications ------------------------- #


# Allowed types of WFDB header fields, and also attributes defined in
# this library
ALLOWED_TYPES = dict(
    [
        [index, _header.FIELD_SPECS.loc[index, "allowed_types"]]
        for index in _header.FIELD_SPECS.index
    ]
)
ALLOWED_TYPES.update(
    {
        "comments": (str,),
        "p_signal": (np.ndarray,),
        "d_signal": (np.ndarray,),
        "e_p_signal": (np.ndarray,),
        "e_d_signal": (np.ndarray,),
        "segments": (Record, type(None)),
    }
)

# Fields that must be lists
LIST_FIELDS = tuple(_header.SIGNAL_SPECS.index) + (
    "comments",
    "e_p_signal",
    "e_d_signal",
    "segments",
)


def _check_item_type(
    item, field_name, allowed_types, expect_list=False, required_channels="all"
):
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
            raise TypeError("Field `%s` must be a list." % field_name)

        # All channels of the field must be present.
        if required_channels == "all":
            required_channels = list(range(len(item)))

        for ch in range(len(item)):
            # Check whether the field may be None
            if ch in required_channels:
                allowed_types_ch = allowed_types
            else:
                allowed_types_ch = allowed_types + (type(None),)

            if not isinstance(item[ch], allowed_types_ch):
                raise TypeError(
                    "Channel %d of field `%s` must be one of the following types:"
                    % (ch, field_name),
                    allowed_types_ch,
                )
    else:
        if not isinstance(item, allowed_types):
            raise TypeError(
                "Field `%s` must be one of the following types:", allowed_types
            )


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
        error_msg = "Field `%s` must have ndim == %d" % (field_name, ndim)
        if channel_num is not None:
            error_msg = ("Channel %d of f" % channel_num) + error_msg[1:]
        raise TypeError(error_msg)

    # Check dtype
    if not np.issubdtype(item.dtype, parent_class):
        error_msg = "Field `%s` must have a dtype that subclasses %s" % (
            field_name,
            parent_class,
        )
        if channel_num is not None:
            error_msg = ("Channel %d of f" % channel_num) + error_msg[1:]
        raise TypeError(error_msg)


# ------------------------- Reading Records --------------------------- #


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
    >>> ecg_record = wfdb.rdheader('100', pn_dir='mitdb')

    """
    dir_name, base_record_name = os.path.split(record_name)
    dir_name = os.path.abspath(dir_name)

    # Construct the download path using the database version
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    # Read the local or remote header file.
    file_name = f"{base_record_name}.hea"
    if pn_dir is None:
        with open(
            os.path.join(dir_name, file_name),
            "r",
            encoding="ascii",
            errors="ignore",
        ) as f:
            header_content = f.read()
    else:
        header_content = download._stream_header(file_name, pn_dir)

    # Separate comment and non-comment lines
    header_lines, comment_lines = header.parse_header_content(header_content)

    # Get fields from record line
    record_fields = _header._parse_record_line(header_lines[0])

    # Single segment header - Process signal specification lines
    if record_fields["n_seg"] is None:
        # Create a single-segment WFDB record object
        record = Record()

        # There are signals
        if len(header_lines) > 1:
            # Read the fields from the signal lines
            signal_fields = _header._parse_signal_lines(header_lines[1:])
            # Set the object's signal fields
            for field in signal_fields:
                setattr(record, field, signal_fields[field])

        # Set the object's record line fields
        for field in record_fields:
            if field == "n_seg":
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
            record.layout = "variable"
        else:
            record.layout = "fixed"

        # If specified, read the segment headers
        if rd_segments:
            record.segments = []
            # Get the base record name (could be empty)
            for s in record.seg_name:
                if s == "~":
                    record.segments.append(None)
                else:
                    record.segments.append(
                        rdheader(os.path.join(dir_name, s), pn_dir)
                    )
            # Fill in the sig_name attribute
            record.sig_name = record.get_sig_name()
            # Fill in the sig_segments attribute
            record.sig_segments = record.get_sig_segments()

    # Set the comments field
    record.comments = [line.strip(" \t#") for line in comment_lines]

    return record


def rdrecord(
    record_name,
    sampfrom=0,
    sampto=None,
    channels=None,
    physical=True,
    pn_dir=None,
    m2s=True,
    smooth_frames=True,
    ignore_skew=False,
    return_res=64,
    force_channels=True,
    channel_names=None,
    warn_empty=False,
):
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
        Specifies whether to smooth the samples in signals with more
        than one sample per frame and return an (MxN) uniform numpy
        array as the `d_signal` or `p_signal` field (True), or to
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
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

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
                # Calculate total number of samples per frame in the
                # first dat file.
                tsamps_per_frame = 0
                for fname, spf in zip(record.file_name, record.samps_per_frame):
                    if fname == record.file_name[0]:
                        tsamps_per_frame += spf

                # Calculate length from size of the dat file.
                record.sig_len = _signal._infer_sig_len(
                    file_name=record.file_name[0],
                    fmt=record.fmt[0],
                    tsamps_per_frame=tsamps_per_frame,
                    byte_offset=record.byte_offset[0],
                    dir_name=dir_name,
                    pn_dir=pn_dir,
                )
        sampto = record.sig_len

    # channel_names takes precedence over channels
    if channel_names is not None:
        # Figure out the channel indices matching the record, if any.
        if isinstance(record, Record):
            reference_record = record
        else:
            if record.layout == "fixed":
                # Find the first non-empty segment to get the signal
                # names
                first_seg_name = [n for n in record.seg_name if n != "~"][0]
                reference_record = rdheader(
                    os.path.join(dir_name, record.seg_name[0]), pn_dir=pn_dir
                )
            else:
                # Use the layout specification header to get the signal
                # names
                reference_record = rdheader(
                    os.path.join(dir_name, record.seg_name[0]), pn_dir=pn_dir
                )

        channels = _get_wanted_channels(
            wanted_sig_names=channel_names,
            record_sig_names=reference_record.sig_name,
        )

    elif channels is None:
        channels = list(range(record.n_sig))

    # Ensure that input fields are valid for the record
    record.check_read_inputs(
        sampfrom, sampto, channels, physical, smooth_frames, return_res
    )

    # If the signal doesn't have the specified channels, there will be
    # no signal. Recall that `rdsamp` is not called on segments of multi
    # segment records if the channels are not present, so this won't
    # break anything.
    if not len(channels):
        old_record = record
        record = Record()
        for attr in _header.RECORD_SPECS.index:
            if attr == "n_seg":
                continue
            elif attr in ["n_sig", "sig_len"]:
                setattr(record, attr, 0)
            else:
                setattr(record, attr, getattr(old_record, attr))
        if warn_empty:
            print("None of the specified signals were contained in the record")

    # A single segment record
    elif isinstance(record, Record):
        no_file = False
        sig_data = None

        record.e_d_signal = _signal._rd_segment(
            file_name=record.file_name,
            dir_name=dir_name,
            pn_dir=pn_dir,
            fmt=record.fmt,
            n_sig=record.n_sig,
            sig_len=record.sig_len,
            byte_offset=record.byte_offset,
            samps_per_frame=record.samps_per_frame,
            skew=record.skew,
            init_value=record.init_value,
            sampfrom=sampfrom,
            sampto=sampto,
            channels=channels,
            ignore_skew=ignore_skew,
            no_file=no_file,
            sig_data=sig_data,
            return_res=return_res,
        )

        # Only 1 sample/frame, or frames are smoothed. Return uniform numpy array
        if smooth_frames:
            # Arrange/edit the object fields to reflect user channel
            # and/or signal range input
            record._arrange_fields(
                channels=channels, sampfrom=sampfrom, smooth_frames=True
            )

            if physical:
                # Perform inplace dac to get physical signal
                record.dac(expanded=False, return_res=return_res, inplace=True)

        # Return each sample of the signals with multiple samples per frame
        else:
            # Arrange/edit the object fields to reflect user channel
            # and/or signal range input
            record._arrange_fields(
                channels=channels, sampfrom=sampfrom, smooth_frames=False
            )

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
        if record.layout == "variable":
            record.segments[0] = rdheader(
                os.path.join(dir_name, record.seg_name[0]), pn_dir=pn_dir
            )

        # The segment numbers and samples within each segment to read.
        seg_numbers, seg_ranges = record._required_segments(sampfrom, sampto)
        # The channels within each segment to read
        seg_channels = record._required_channels(
            seg_numbers, channels, dir_name, pn_dir
        )

        # Read the desired samples in the relevant segments
        for i in range(len(seg_numbers)):
            seg_num = seg_numbers[i]
            # Empty segment or segment with no relevant channels
            if record.seg_name[seg_num] == "~" or len(seg_channels[i]) == 0:
                record.segments[seg_num] = None
            else:
                record.segments[seg_num] = rdrecord(
                    os.path.join(dir_name, record.seg_name[seg_num]),
                    sampfrom=seg_ranges[i][0],
                    sampto=seg_ranges[i][1],
                    channels=seg_channels[i],
                    physical=physical,
                    pn_dir=pn_dir,
                    smooth_frames=smooth_frames,
                    return_res=return_res,
                )

        # Arrange the fields of the layout specification segment, and
        # the overall object, to reflect user input.
        record._arrange_fields(
            seg_numbers=seg_numbers,
            seg_ranges=seg_ranges,
            channels=channels,
            sampfrom=sampfrom,
            force_channels=force_channels,
        )

        # Convert object into a single segment Record object
        if m2s:
            record = record.multi_to_single(
                physical=physical,
                expanded=(not smooth_frames),
                return_res=return_res,
            )

    # Perform dtype conversion if necessary
    if isinstance(record, Record) and record.n_sig > 0:
        record.convert_dtype(physical, return_res, smooth_frames)

    return record


def rdsamp(
    record_name,
    sampfrom=0,
    sampto=None,
    channels=None,
    pn_dir=None,
    channel_names=None,
    warn_empty=False,
    return_res=64,
):
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
                                      channels=[1,3])

    """
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    record = rdrecord(
        record_name=record_name,
        sampfrom=sampfrom,
        sampto=sampto,
        channels=channels,
        physical=True,
        pn_dir=pn_dir,
        m2s=True,
        return_res=return_res,
        channel_names=channel_names,
        warn_empty=warn_empty,
    )

    signals = record.p_signal
    fields = {}
    for field in [
        "fs",
        "sig_len",
        "n_sig",
        "base_date",
        "base_time",
        "units",
        "sig_name",
        "comments",
    ]:
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
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    record = rdheader(record_name, pn_dir=pn_dir)
    samps_per_frame = [record.fs * samp for samp in record.samps_per_frame]
    sig_name = record.sig_name

    for sig, samp in zip(sig_name, samps_per_frame):
        print("{}\t{}".format(sig, samp))


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
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    record = rdheader(record_name, pn_dir=pn_dir)
    if len(sig_nums) > 0:
        for n in sig_nums:
            try:
                print(record.sig_name[n])
            except IndexError:
                raise Exception("sig_nums value {} out of range".format(n))
    else:
        print(*record.sig_name, sep="\n")


def wfdbdesc(record_name, pn_dir=None):
    """
    Reads specifications for the signals described in the header file for
    a record.

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
    >>> wfdb.wfdbdesc('100', pn_dir='mitdb')
    Record 100
    Notes
    =====
     69 M 1085 1629 x1
     Aldomet, Inderal
    =====

    Starting time: not specified
    Length: 0:30:05.555556 (650000 sample intervals)
    Sampling frequency: 360 Hz
    2 signals
    Group 0, Signal 0:
     File: 100.dat
     Description: MLII
     Gain: 200.0 mV
     Initial value: 995
     Storage format: 212 (1 sample per frame)
     I/O: can be unbuffered
     ADC resolution: 11 bits
     ADC zero: 1024
     Baseline: 1024
     Checksum: -22131
    Group 0, Signal 1:
     File: 100.dat
     Description: V5
     Gain: 200.0 mV
     Initial value: 1011
     Storage format: 212 (1 sample per frame)
     I/O: can be unbuffered
     ADC resolution: 11 bits
     ADC zero: 1024
     Baseline: 1024
     Checksum: 20052

    """
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    record = rdheader(record_name, pn_dir=pn_dir)
    if type(record) is MultiRecord:
        sub_string = "fixed" if record.segments else "variable"
        seg_string = f" (a {sub_string}-layout multi-segment record)"
    else:
        seg_string = ""

    print(f"Record {record_name}{seg_string}")

    if record.comments:
        print("Notes")
        print("=====")
        for comment in record.comments:
            print(f" {comment}")
        print("=====")
    print()

    try:
        start_time = (
            f"[{datetime.datetime.combine(record.base_date, record.base_time)}]"
        )
    except TypeError:
        start_time = "not specified"
    print(f"Starting time: {start_time}")

    record_length = str(datetime.timedelta(seconds=record.sig_len / record.fs))
    print(f"Length: {record_length} ({record.sig_len} sample intervals)")

    print(f"Sampling frequency: {record.fs} Hz")
    print(f'{record.n_sig} signal{"" if record.n_sig==1 else "s"}')

    if type(record) is not MultiRecord:
        for i in range(record.n_sig):
            print(f"Group 0, Signal {i}:")
            print(f" File: {record.file_name[i]}")
            print(f" Description: {record.sig_name[i]}")
            print(f" Gain: {record.adc_gain[i]} {record.units[i]}")
            print(f" Initial value: {record.init_value[i]}")
            sample_string = "" if record.samps_per_frame[i] == 1 else "s"
            print(
                f" Storage format: {record.fmt[i]} ({record.samps_per_frame[i]} sample{sample_string} per frame)"
            )
            if record.byte_offset[i]:
                byte_string = f"{record.byte_offset[i]}-byte blocks"
            else:
                byte_string = "can be unbuffered"
            print(f" I/O: {byte_string}")
            print(f" ADC resolution: {record.adc_res[i]} bits")
            print(f" ADC zero: {record.adc_zero[i]}")
            print(f" Baseline: {record.baseline[i]}")
            print(f" Checksum: {record.checksum[i]}")


def wfdbtime(record_name, input_times, pn_dir=None):
    """
    Use the specified record as a reference for determining the length of a
    sample interval and the absolute time represented by sample number 0. This
    program accepts one or more time arguments (in WFDB standard time format)
    and produces one line on the standard output for each such argument. In
    each output line, the corresponding time is written as a sample number (in
    the form sNNN), as an elapsed time interval in hours, minutes, and seconds
    from the beginning of the record (in the form hh:mm:ss.sss), and as an
    absolute time and date (in the form [hh:mm:ss.sss DD/MM/YYYY]). If the
    base time for the record is undefined, the absolute time cannot be
    calculated, and in this case the elapsed time appears twice instead.

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
    input_times : str, list
        The desired times (or samples) to be converted and displayed for the
        chosen record. This can either be a single string, or a list of
        strings depending on how many results the user would like. The string
        must either be an integer (in which case '1' will be interpreted as 1
        second), a datetime format (hh:mm:ss.sss DD/MM/YYYY), or prefixed with
        the letter 's' if the time at a selected sample is desired (i.e. 's10'
        for sample time). For the datetime format, 0's do not have be filled
        for all values (i.e. '5::' will be interpreted as '05:00:00.000',
        ':5:' will be '00:05:00.000', etc.). The string 'e' can be used to
        represent the end of the record.
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
    * Note, the use of a single string instead of a list.
    >>> wfdb.wfdbtime('sample-data/100', 's250')
        s250            00:00:00.694                    00:00:00.694

    * Note, the elapsed time and date fields are the same since no start time
      or date was provided.
    >>> wfdb.wfdbtime('sample-data/100', ['s10',':5:','2'])
         s10            00:00:00.028                    00:00:00.028
     s108000            00:05:00.000                    00:05:00.000
        s720            00:00:02.000                    00:00:02.000

    * Note, '01/01/0001' represents a start time was provided but not a date.
    >>> wfdb.wfdbtime('sample-data/3000003_0003', ['s1','0:0:05','e'])
          s1            00:00:00.008    [19:46:25.765000 01/01/0001]
        s625            00:00:05.000    [19:46:30.757000 01/01/0001]
       s1028            00:00:08.224    [19:46:33.981000 01/01/0001]

    * Note, the final argument results in the same date field as the argument
      but a different value for the elapsed time.
    >>> wfdb.wfdbtime('sample-data/3000003_0003', ['s1','::5','e','19:46:34.981 01/01/0001'])
          s1            00:00:00.008    [19:46:25.765000 01/01/0001]
        s625            00:00:05.000    [19:46:30.757000 01/01/0001]
       s1028            00:00:08.224    [19:46:33.981000 01/01/0001]
       s1153            00:00:09.224    [19:46:34.981000 01/01/0001]

    """
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    record = rdheader(record_name, pn_dir=pn_dir)
    try:
        start_time = datetime.datetime.combine(
            record.base_date, record.base_time
        )
    except TypeError:
        try:
            start_time = record.base_time
        except AttributeError:
            start_time = None

    if type(input_times) is str:
        input_times = [input_times]

    for times in input_times:
        if times == "e":
            sample_num = record.sig_len
            sample_time = float(sample_num / record.fs)
            seconds = float(sample_time) % 60
            minutes = int(float(sample_time) // 60)
            hours = int(float(sample_time) // 60 // 60)
        else:
            if times.startswith("s"):
                sample_num = int(times[1:])
                new_times = float(sample_num / record.fs)
                times_split = [f"{new_times}"]
            elif "/" in times:
                out_date = datetime.datetime.strptime(
                    times, "%H:%M:%S.%f %d/%m/%Y"
                )
                try:
                    start_time = datetime.datetime.combine(
                        datetime.date.min, start_time
                    )
                except TypeError:
                    start_time = start_time
                try:
                    elapsed_time = out_date - start_time
                except TypeError:
                    raise Exception(
                        "No start date or time provided in the record."
                    )
                total_seconds = elapsed_time.total_seconds()
                sample_num = "s" + str(
                    int(elapsed_time.total_seconds() * record.fs)
                )
                hours = int(total_seconds // 60 // 60)
                minutes = int(total_seconds // 60)
                out_time = f"{hours:02}:{minutes:02}:{total_seconds:06.3f}"
                out_date = f'[{out_date.strftime("%H:%M:%S.%f %d/%m/%Y")}]'
                print(f"{sample_num:>12}{out_time:>24}{out_date:>32}")
                break
            else:
                new_times = times
                times_split = [t if t != "" else "0" for t in times.split(":")]
            if len(times_split) == 1:
                seconds = float(new_times) % 60
                minutes = int(float(new_times) // 60)
                hours = int(float(new_times) // 60 // 60)
            elif len(times_split) == 2:
                seconds = float(times_split[1])
                minutes = int(times_split[0])
                hours = 0
            elif len(times_split) == 3:
                seconds = float(times_split[2])
                minutes = int(times_split[1])
                hours = int(times_split[0])
            if seconds >= 60:
                raise Exception("Seconds not in correct format")
            if minutes >= 60:
                raise Exception("Minutes not in correct format")
        out_time = f"{hours:02}:{minutes:02}:{seconds:06.3f}"
        out_date = _get_date_from_time(
            start_time, hours, minutes, seconds, out_time
        )
        if not times.startswith("s"):
            sample_num = int(
                sum(
                    x * 60**i for i, x in enumerate([seconds, minutes, hours])
                )
                * record.fs
            )
        sample_num = "s" + str(sample_num)
        print(f"{sample_num:>12}{out_time:>24}{out_date:>32}")


def _get_date_from_time(start_time, hours, minutes, seconds, out_time):
    """
    Convert a starting time to a date using the elapsed time.

    Parameters
    ----------
    start_time : datetime object
        The time the record start if available.
    hours : int
        The number of hours elapsed.
    minutes : int
        The number of minutes elapsed.
    seconds : int
        The number of seconds elapsed.
    out_time : str
        The string formatted time elapsed for a desired sample, if available.

    Returns
    -------
    out_date : datetime object
        The time the record ends after the caculcated elapsed time.

    """
    if start_time:
        try:
            start_time = (
                datetime.datetime.combine(datetime.date.min, start_time)
                - datetime.datetime.min
            )
        except TypeError:
            start_time = start_time - datetime.datetime.min
        microseconds = int(1000000 * (seconds % 1))
        elapsed_time = datetime.timedelta(
            hours=hours,
            minutes=minutes,
            seconds=int(seconds),
            microseconds=microseconds,
        )
        out_date = start_time + elapsed_time
        out_date = f'[{(datetime.datetime.min+out_date).strftime("%H:%M:%S.%f %d/%m/%Y")}]'
    else:
        out_date = out_time
    return out_date


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
        return [
            record_sig_names.index(s) if s in record_sig_names else None
            for s in wanted_sig_names
        ]
    else:
        return [
            record_sig_names.index(s)
            for s in wanted_sig_names
            if s in record_sig_names
        ]


# ------------------- /Reading Records -------------------#


def wrsamp(
    record_name,
    fs,
    units,
    sig_name,
    p_signal=None,
    d_signal=None,
    fmt=None,
    adc_gain=None,
    baseline=None,
    comments=None,
    base_time=None,
    base_date=None,
    base_datetime=None,
    write_dir="",
):
    """
    Write a single segment WFDB record, creating a WFDB header file and any
    associated dat files.

    Parameters
    ----------
    record_name : str
        The string name of the WFDB record to be written (without any file
        extensions). Must not contain any "."
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
    base_time : datetime.time, optional
        The time of day at the beginning of the record.
    base_date : datetime.date, optional
        The date at the beginning of the record.
    base_datetime : datetime.datetime, optional
        The date and time at the beginning of the record, equivalent to
        setting both `base_date` and `base_time`.
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
    if "." in record_name:
        raise Exception("Record name must not contain '.'")
    # Check input field combinations
    if p_signal is not None and d_signal is not None:
        raise Exception(
            "Must only give one of the inputs: p_signal or d_signal"
        )
    if d_signal is not None:
        if fmt is None or adc_gain is None or baseline is None:
            raise Exception(
                "When using d_signal, must also specify 'fmt', 'gain', and 'baseline' fields."
            )
    # Depending on whether d_signal or p_signal was used, set other
    # required features.
    if p_signal is not None:
        # Create the Record object
        record = Record(
            record_name=record_name,
            p_signal=p_signal,
            fs=fs,
            fmt=fmt,
            units=units,
            sig_name=sig_name,
            adc_gain=adc_gain,
            baseline=baseline,
            comments=comments,
            base_time=base_time,
            base_date=base_date,
            base_datetime=base_datetime,
        )
        # Compute optimal fields to store the digital signal, carry out adc,
        # and set the fields.
        record.set_d_features(do_adc=1)
    else:
        # Create the Record object
        record = Record(
            record_name=record_name,
            d_signal=d_signal,
            fs=fs,
            fmt=fmt,
            units=units,
            sig_name=sig_name,
            adc_gain=adc_gain,
            baseline=baseline,
            comments=comments,
            base_time=base_time,
            base_date=base_date,
            base_datetime=base_datetime,
        )
        # Use d_signal to set the fields directly
        record.set_d_features()

    # Set default values of any missing field dependencies
    record.set_defaults()
    # Write the record files - header and associated dat
    record.wrsamp(write_dir=write_dir)


def dl_database(
    db_dir,
    dl_dir,
    records="all",
    annotators="all",
    keep_subdirs=True,
    overwrite=False,
):
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
    if "/" in db_dir:
        dir_list = db_dir.split("/")
        db_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )
    else:
        db_dir = posixpath.join(db_dir, download.get_version(db_dir))
    db_url = posixpath.join(download.PN_CONTENT_URL, db_dir) + "/"
    # Check if the database is valid
    _url.openurl(db_url, check_access=True)

    # Get the list of records
    record_list = download.get_record_list(db_dir, records)
    # Get the annotator extensions
    annotators = download.get_annotators(db_dir, annotators)

    # All files to download (relative to the database's home directory)
    all_files = []
    nested_records = []

    for rec in record_list:
        print("Generating record list for: " + rec)
        # May be pointing to directory
        if rec.endswith(os.sep):
            nested_records += [
                posixpath.join(rec, sr)
                for sr in download.get_record_list(posixpath.join(db_dir, rec))
            ]
        else:
            nested_records.append(rec)

    for rec in nested_records:
        print("Generating list of all files for: " + rec)
        # If MIT format, have to figure out all associated files
        all_files.append(rec + ".hea")
        dir_name, base_rec_name = os.path.split(rec)
        record = rdheader(
            base_rec_name, pn_dir=posixpath.join(db_dir, dir_name)
        )

        # Single segment record
        if isinstance(record, Record):
            # Add all dat files of the segment
            for file in record.file_name if record.file_name else []:
                all_files.append(posixpath.join(dir_name, file))

        # Multi segment record
        else:
            for seg in record.seg_name:
                # Skip empty segments
                if seg == "~":
                    continue
                # Add the header
                all_files.append(posixpath.join(dir_name, seg + ".hea"))
                # Layout specifier has no dat files
                if seg.endswith("_layout"):
                    continue
                # Add all dat files of the segment
                rec_seg = rdheader(seg, pn_dir=posixpath.join(db_dir, dir_name))
                for file in rec_seg.file_name:
                    all_files.append(posixpath.join(dir_name, file))

        # Check whether the record has any requested annotation files
        if annotators is not None:
            for a in annotators:
                ann_file = rec + "." + a
                url = posixpath.join(
                    download.config.db_index_url, db_dir, ann_file
                )
                try:
                    _url.openurl(url, check_access=True)
                    all_files.append(ann_file)
                except FileNotFoundError:
                    pass

    dl_inputs = [
        (
            os.path.split(file)[1],
            os.path.split(file)[0],
            db_dir,
            dl_dir,
            keep_subdirs,
            overwrite,
        )
        for file in all_files
    ]

    # Make any required local directories
    download.make_local_dirs(dl_dir, dl_inputs, keep_subdirs)

    print("Downloading files...")
    # Create multiple processes to download files.
    # Limit to 2 connections to avoid overloading the server
    pool = multiprocessing.dummy.Pool(processes=2)
    pool.map(download.dl_pn_file, dl_inputs)
    print("Finished downloading files")

    return
