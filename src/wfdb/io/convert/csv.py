import numbers
import os
import warnings

import numpy as np
import pandas as pd

from wfdb.io.annotation import Annotation, format_ann_from_df, wrann
from wfdb.io.record import Record


def csv_to_wfdb(
    file_name,
    fs,
    units,
    fmt=None,
    adc_gain=None,
    baseline=None,
    samps_per_frame=None,
    counter_freq=None,
    base_counter=None,
    base_time=None,
    base_date=None,
    comments=None,
    sig_name=None,
    dat_file_name=None,
    skew=None,
    byte_offset=None,
    adc_res=None,
    adc_zero=None,
    init_value=None,
    checksum=None,
    block_size=None,
    record_only=False,
    header=True,
    delimiter=",",
    verbose=False,
    write_dir="",
):
    """Read a CSV file and write a WFDB record or return a `Record` object.

    By default, a header file ('.hea') and a signal file ('.dat') are written to
    `write_dir`, which is the current working directory unless set otherwise. If
    `record_only` is True, no files are written and the `Record` is returned instead.

    Parameters
    ----------
    file_name : str
        The path to the CSV file, including the '.csv' extension. Both relative
        and absolute paths are accepted. The file name without its directory
        and extension is used as the record name.
    fs : float
        The sampling frequency of the signals in Hz (one CSV row per sample).
    units : str or list of str
        The physical units of each signal. A single string is used for all
        signals.
    fmt : str or list of str, optional
        The WFDB storage format of each signal, such as '16', '212', '80', '24'
        or '32'. A single string is used for all signals. The default is '16'.
    adc_gain : float or list of float, optional
        The number of digital units per physical unit for each signal. A single
        number is used for all signals. The default is 200.
    baseline : int or list of int, optional
        The digital value corresponding to 0 physical units for each signal. A
        single number is used for all signals. The default is `adc_zero`.
    samps_per_frame : optional
        Deprecated and ignored. Each CSV row is one frame with one sample per
        signal.
    counter_freq : float, optional
        The counter frequency in ticks per second, written to the header.
    base_counter : float, optional
        The counter value corresponding to sample 0, written to the header.
        Only used if `counter_freq` is set.
    base_time : datetime.time, optional
        The time of day at the beginning of the record.
    base_date : datetime.date, optional
        The date at the beginning of the record. Only used if `base_time` is
        set.
    comments : list of str, optional
        Comments to write to the header file, one line per string.
    sig_name : list of str, optional
        The name of each signal. If given, these are used instead of the names
        in the CSV header.
    dat_file_name : str or list of str, optional
        The name of the signal file to write (without a directory). The default
        is the record name with the '.dat' extension.
    skew : int or list of int, optional
        The number of samples of each signal that precede sample 0, written to
        the header. A single number is used for all signals. The default is 0.
    byte_offset : int or list of int, optional
        The number of bytes before the first sample in the signal file. The
        signal file is written with this many empty leading bytes. A single
        number is used for all signals. The default is 0.
    adc_res : int or list of int, optional
        The resolution of the ADC in bits for each signal. A single number is
        used for all signals. The default is the resolution of `fmt`.
    adc_zero : int or list of int, optional
        The digital value in the middle of the ADC input range for each signal.
        A single number is used for all signals. The default is 0.
    init_value : optional
        Deprecated and ignored. The initial value of each signal is computed
        from the samples.
    checksum : optional
        Deprecated and ignored. The checksum of each signal is computed from
        the samples.
    block_size : int or list of int, optional
        The block size of each signal file in bytes, written to the header. A
        single number is used for all signals. The default is 0.
    record_only : bool, optional
        Whether to only return the record information (True) or not (False).
        If False, this function will write both a .dat and .hea file and
        return None.
    header : bool, optional
        Whether to assume the CSV has a first line header (True) or not (False)
        which defines the signal names. If false, this function will generate
        either the signal names provided by `sig_name` or set `[ch_0, ch_1, ...]`
        as the default.
    delimiter : str, optional
        What to use as the delimiter for the file to separate data. The default
        if a comma (','). Other common delimiters are tabs ('\t'), spaces (' '),
        pipes ('|'), and colons (':').
    verbose : bool, optional
        Whether to print all the information read about the file (True) or
        not (False).
    write_dir : str, optional
        The directory where the output files will be saved. If `write_dir` is
        empty (the default), the output files will be saved in the current
        working directory, not the directory of the input file. This is ignored
        if `record_only` is True.

    Returns
    -------
    record : Record, optional
        The WFDB Record object representing the contents of the CSV file read.
        This is only returned if `record_only` is True, otherwise None.

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
    >>> csv_to_wfdb('sample-data/100.csv', fs=360, units='mV')

    Create the header ('.hea') and record ('.dat') files, change units for
    each signal
    >>> csv_to_wfdb('sample-data/100.csv', fs=360, units=['mV','kV'])

    Return just the record, note the use of lists to specify which values should
    be applied to each signal
    >>> csv_record = csv_to_wfdb('sample-data/100.csv', fs=360,
    ...                          units=['mV', 'mV'], fmt=['80', '212'],
    ...                          adc_gain=[100, 200], baseline=[1024, 512],
    ...                          record_only=True)

    Return just the record, note the use of single strings and ints to specify
    when fields can be applied to all signals
    >>> csv_record = csv_to_wfdb('sample-data/100.csv', fs=360, units='mV',
    ...                          fmt=['80', '212'], adc_gain=200, baseline=1024,
    ...                          record_only=True)

    """
    # NOTE: No need to write input checks here since the Record class should
    # handle them (except verifying the CSV input format which is for Pandas)
    if header:
        df_CSV = pd.read_csv(file_name, delimiter=delimiter)
    else:
        df_CSV = pd.read_csv(file_name, delimiter=delimiter, header=None)
    if verbose:
        print("Successfully read CSV")

    # Extract the entire signal from the dataframe
    p_signal = df_CSV.values
    # The dataframe should be in (`sig_len`, `n_sig`) dimensions
    sig_len = p_signal.shape[0]
    if sig_len == 0:
        raise ValueError(f"The CSV file {file_name} contains no samples")
    if verbose:
        print(f"Signal length: {sig_len}")
    n_sig = p_signal.shape[1]
    if verbose:
        print(f"Number of signals: {n_sig}")

    # Check if signal names are valid and set defaults
    if not sig_name:
        if header:
            sig_name = df_CSV.columns.tolist()
            if any(map(str.isdigit, sig_name)):
                print(
                    "WARNING: One or more of your signal names are numbers, this "
                    "is not recommended:\n- Does your CSV have a header line "
                    "which defines the signal names?\n- If not, please set the "
                    f"parameter 'header' to False.\nSignal names: {sig_name}"
                )
        else:
            sig_name = ["ch_" + str(i) for i in range(n_sig)]
            if verbose:
                print(f"Signal names: {sig_name}")

    record_name = os.path.splitext(os.path.basename(file_name))[0]
    if verbose:
        print(f"Record name: {record_name}.hea")

    if dat_file_name is None:
        dat_file_name = record_name + ".dat"
    dat_file_name = _per_channel(dat_file_name, n_sig, "dat_file_name")
    if verbose:
        print(f"Output record: {dat_file_name[0]}")

    # These follow from the CSV samples, so they can't be chosen freely
    ignored = {
        "samps_per_frame": samps_per_frame,
        "init_value": init_value,
        "checksum": checksum,
    }
    ignored = [name for name, value in ignored.items() if value is not None]
    if ignored:
        warnings.warn(
            f"The {', '.join(ignored)} argument(s) of csv_to_wfdb are deprecated "
            "and ignored. Each CSV row is one frame with one sample per signal, "
            "and the initial value and checksum are computed from the samples.",
            FutureWarning,
            stacklevel=2,
        )

    # Fill in defaults and use one value per channel. Only None means missing,
    # so explicit zeros are kept.
    if fmt is None:
        fmt = "16"
    if adc_gain is None:
        adc_gain = 200
    if adc_zero is None:
        adc_zero = 0
    adc_zero = _per_channel(adc_zero, n_sig, "adc_zero")
    if baseline is None:
        baseline = adc_zero
    if block_size is None:
        block_size = 0

    record = Record(
        record_name=record_name,
        n_sig=n_sig,
        fs=fs,
        counter_freq=counter_freq,
        base_counter=base_counter,
        sig_len=sig_len,
        base_time=base_time,
        base_date=base_date,
        comments=comments,
        sig_name=sig_name,
        p_signal=p_signal.astype("float64"),
        file_name=dat_file_name,
        fmt=_per_channel(fmt, n_sig, "fmt"),
        skew=_per_channel(skew, n_sig, "skew"),
        byte_offset=_per_channel(byte_offset, n_sig, "byte_offset"),
        adc_gain=_per_channel(adc_gain, n_sig, "adc_gain"),
        baseline=_per_channel(baseline, n_sig, "baseline"),
        units=_per_channel(units, n_sig, "units"),
        adc_res=_per_channel(adc_res, n_sig, "adc_res"),
        adc_zero=adc_zero,
        block_size=_per_channel(block_size, n_sig, "block_size"),
    )

    # The initial value and checksum describe the digital samples
    record.d_signal = record.adc()
    record.init_value = [int(v) for v in record.d_signal[0, :]]
    record.checksum = record.calc_checksum()
    # Fill in the remaining fields as wrsamp does (e.g. adc_res from fmt)
    record.set_defaults()
    if verbose:
        print(f"Signal format: {record.fmt}")
        print(f"Signal ADC gain: {record.adc_gain}")
        print(f"Signal baseline: {record.baseline}")
        print(f"Signal ADC resolution: {record.adc_res}")
        print(f"Signal ADC zero: {record.adc_zero}")
        print(f"Signal initial value: {record.init_value}")
        print(f"Signal checksum: {record.checksum}")
        print(f"Signal block size: {record.block_size}")

    if record_only:
        record.d_signal = None
        if verbose:
            print("Record generated successfully")
        return record

    record.wrsamp(write_dir=write_dir)
    if verbose:
        print("File generated successfully")


def _per_channel(value, n_sig, name):
    """Return `value` as a list with one entry per channel.

    A single value (str, int, float or NumPy scalar) is repeated for all
    channels, None is returned unchanged and anything else is converted to a list.
    `name` is the argument name used in the error for a bool.

    """
    if value is None:
        return None
    # bool is a Number, but never a valid header field
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must not be a bool")
    if isinstance(value, (str, numbers.Number, np.generic)):
        return [value] * n_sig
    return list(value)


def csv2ann(
    file_name,
    extension="atr",
    fs=None,
    record_only=False,
    time_onset=True,
    header=True,
    delimiter=",",
    verbose=False,
    write_dir="",
):
    """Read a CSV file and return an `Annotation` object or write a file.

    The annotation descriptors are the attributes of the `Annotation` object.

    Parameters
    ----------
    file_name : str
        The path to the CSV file to be read, including the '.csv' file
        extension. Both relative and absolute paths are accepted. The file
        name without its directory and extension is used as the record name
        of the annotation file.
    extension : str, optional
        The string annotation file extension.
    fs : float, optional
        This will be used if annotation onsets are given in the format of time
        (`time_onset` = True) instead of sample since onsets must be sample
        numbers in order for `wrann` to work. This number can be expressed in
        any format legal for a Python input of floating point numbers (thus
        '360', '360.', '360.0', and '3.6e2' are all legal and equivalent). The
        sampling frequency must be greater than 0; if it is missing, a value
        of 250 is assumed.
    record_only : bool, optional
        Whether to only return the record information (True) or not (False).
        If false, this function will generate the annotation file.
    time_onset : bool, optional
        Whether to assume the values provided in the 'onset' column are in
        units of time (True) or samples (False). If True, convert the onset
        times to samples by using the, now required, `fs` input.
    header : bool, optional
        Whether to assume the CSV has a first line header (True) or not
        (False) which defines the signal names.
    delimiter : str, optional
        What to use as the delimiter for the file to separate data. The default
        if a comma (','). Other common delimiters are tabs ('\t'), spaces (' '),
        pipes ('|'), and colons (':').
    verbose : bool, optional
        Whether to print all the information read about the file (True) or
        not (False).
    write_dir : str, optional
        The directory in which to write the annotation file. The default is
        the current working directory. Ignored if `record_only` is True.

    Returns
    -------
    N/A : Annotation, optional
        The WFDB Annotation object representing the contents of the CSV file
        read.

    Notes
    -----
    CSVs should be in one of the two possible following format:

    1) All events are single time events (no duration).

    onset,description
    onset_1,description_1
    onset_2,description_2
    ...,...

    Or this format if `header=False` is defined:

    onset_1,description_1
    onset_2,description_2
    ...,...

    2) A duration is specified for some events.

    onset,duration,description
    onset_1,duration_1,description_1
    onset_2,duration_2,description_2
    ...,...,...

    Or this format if `header=False` is defined:

    onset_1,duration_1,description_1
    onset_2,duration_2,description_2
    ...,...,...

    By default, the 'onset' will be interpreted as a sample number if it is
    strictly in integer format and as a time otherwise. By default, the
    'duration' will be interpreted as time values and not elapsed samples. By
    default, the 'description' will be interpreted as the `aux_note` for the
    annotation and the `symbol` will automatically be set to " which defines a
    comment. Future additions will allow the user to customize such
    attributes.

    Examples
    --------
    1) Write WFDB annotation file from CSV with time onsets:
       ======= start example.csv =======
       onset,description
       0.2,p-wave
       0.8,qrs
       ======== end example.csv ========
       >>> wfdb.csv2ann('example.csv', fs=360)
       * Creates a WFDB annotation file called: 'example.atr'

    2) Write WFDB annotation file from CSV with sample onsets:
       ======= start example.csv =======
       onset,description
       5,p-wave
       13,qrs
       ======== end example.csv ========
       >>> wfdb.csv2ann('example.csv', fs=10, time_onset=False)
       * Creates a WFDB annotation file called: 'example.atr'
       * 5,13 samples -> 0.5,1.3 seconds for onset

    3) Write WFDB annotation file from CSV with time onsets, durations, and no
       header:
       ======= start example.csv =======
       0.2,0.1,qrs
       0.8,0.4,qrs
       ======== end example.csv ========
       >>> wfdb.csv2ann('example.csv', extension='qrs', fs=360, header=False)
       * Creates a WFDB annotation file called: 'example.qrs'

    """
    # NOTE: No need to write input checks here since the Annotation class
    # should handle them (except verifying the CSV input format which is for
    # Pandas)
    if header:
        df_CSV = pd.read_csv(file_name, delimiter=delimiter)
    else:
        df_CSV = pd.read_csv(file_name, delimiter=delimiter, header=None)
    if verbose:
        print("Successfully read CSV")

    if verbose:
        print("Creating Pandas dataframe from CSV")
    if df_CSV.shape[1] == 2:
        if verbose:
            print("onset,description format detected")
        if not header:
            df_CSV.columns = ["onset", "description"]
        df_out = df_CSV
    elif df_CSV.shape[1] == 3:
        if verbose:
            print("onset,duration,description format detected")
            print("Converting durations to single time-point events")
        if not header:
            df_CSV.columns = ["onset", "duration", "description"]
        df_out = format_ann_from_df(df_CSV)
    else:
        raise Exception("""The number of columns in the CSV was not
                        recognized.""")

    record_name = os.path.splitext(os.path.basename(file_name))[0]
    if time_onset:
        if not fs:
            raise Exception("""`fs` must be provided if `time_onset` is True
                            since it is required to convert time onsets to
                            samples""")
        sample = (df_out["onset"].to_numpy() * fs).astype(np.int64)
    else:
        sample = df_out["onset"].to_numpy()
    # Assume each annotation is a comment
    symbol = ['"'] * len(df_out.index)
    subtype = np.array([22] * len(df_out.index))
    # Assume each annotation belongs with the 1st channel
    chan = np.array([0] * len(df_out.index))
    num = np.array([0] * len(df_out.index))
    aux_note = df_out["description"].tolist()

    if verbose:
        print("Finished CSV parsing... writing to Annotation object")

    if record_only:
        if verbose:
            print("Finished creating Annotation object")
        return Annotation(
            record_name=record_name,
            extension=extension,
            sample=sample,
            symbol=symbol,
            subtype=subtype,
            chan=chan,
            num=num,
            aux_note=aux_note,
            fs=fs,
        )
    else:
        wrann(
            record_name,
            extension,
            sample=sample,
            symbol=symbol,
            subtype=subtype,
            chan=chan,
            num=num,
            aux_note=aux_note,
            fs=fs,
            write_dir=write_dir,
        )
        if verbose:
            print("Finished writing Annotation file")
