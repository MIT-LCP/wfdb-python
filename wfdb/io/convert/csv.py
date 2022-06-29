import os

import numpy as np
import pandas as pd

from wfdb.io.annotation import format_ann_from_df, Annotation, wrann
from wfdb.io.record import Record, wrsamp


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
):
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
    base_time : datetime.time, optional
        This field can be present only if the number of samples is also present.
        It gives the time of day that corresponds to the beginning of the
        record.
    base_date : datetime.date, optional
        This field can be present only if the base time is also present. It contains
        the date that corresponds to the beginning of the record.
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
    >>> csv_to_wfdb('sample-data/100.csv', fs=360, units='mV')

    Create the header ('.hea') and record ('.dat') files, change units for
    each signal
    >>> csv_to_wfdb('sample-data/100.csv', fs=360, units=['mV','kV'])

    Return just the record, note the use of lists to specify which values should
    be applied to each signal
    >>> csv_record = csv_to_wfdb('sample-data/100.csv', fs=360, units=['mV','mV'],
                                 fmt=['80',212'], adc_gain=[100,200],
                                 baseline=[1024,512], record_only=True)

    Return just the record, note the use of single strings and ints to specify
    when fields can be applied to all signals
    >>> csv_record = csv_to_wfdb('sample-data/100.csv', fs=360, units='mV',
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
        print("Successfully read CSV")
    # Extract the entire signal from the dataframe
    p_signal = df_CSV.values
    # The dataframe should be in (`sig_len`, `n_sig`) dimensions
    sig_len = p_signal.shape[0]
    if verbose:
        print("Signal length: {}".format(sig_len))
    n_sig = p_signal.shape[1]
    if verbose:
        print("Number of signals: {}".format(n_sig))
    # Check if signal names are valid and set defaults
    if not sig_name:
        if header:
            sig_name = df_CSV.columns.to_list()
            if any(map(str.isdigit, sig_name)):
                print(
                    "WARNING: One or more of your signal names are numbers, this "
                    "is not recommended:\n- Does your CSV have a header line "
                    "which defines the signal names?\n- If not, please set the "
                    "parameter 'header' to False.\nSignal names: {}".format(
                        sig_name
                    )
                )
        else:
            sig_name = ["ch_" + str(i) for i in range(n_sig)]
            if verbose:
                print("Signal names: {}".format(sig_name))

    # Set the output header file name to be the same, remove path
    if os.sep in file_name:
        file_name = file_name.split(os.sep)[-1]
    record_name = file_name.replace(".csv", "")
    if verbose:
        print("Output header: {}.hea".format(record_name))

    # Replace the CSV file tag with DAT
    dat_file_name = file_name.replace(".csv", ".dat")
    dat_file_name = [dat_file_name] * n_sig
    if verbose:
        print("Output record: {}".format(dat_file_name[0]))

    # Convert `units` from string to list if necessary
    units = [units] * n_sig if type(units) is str else units

    # Set the default `fmt` if none exists
    if not fmt:
        fmt = ["16"] * n_sig
    fmt = [fmt] * n_sig if type(fmt) is str else fmt
    if verbose:
        print("Signal format: {}".format(fmt))

    # Set the default `adc_gain` if none exists
    if not adc_gain:
        adc_gain = [200] * n_sig
    adc_gain = [adc_gain] * n_sig if type(adc_gain) is int else adc_gain
    if verbose:
        print("Signal ADC gain: {}".format(adc_gain))

    # Set the default `baseline` if none exists
    if not baseline:
        if adc_zero:
            baseline = [adc_zero] * n_sig
        else:
            baseline = [0] * n_sig
    baseline = [baseline] * n_sig if type(baseline) is int else baseline
    if verbose:
        print("Signal baseline: {}".format(baseline))

    # Convert `samps_per_frame` from int to list if necessary
    samps_per_frame = (
        [samps_per_frame] * n_sig
        if type(samps_per_frame) is int
        else samps_per_frame
    )

    # Convert `skew` from int to list if necessary
    skew = [skew] * n_sig if type(skew) is int else skew

    # Convert `byte_offset` from int to list if necessary
    byte_offset = (
        [byte_offset] * n_sig if type(byte_offset) is int else byte_offset
    )

    # Set the default `adc_res` if none exists
    if not adc_res:
        adc_res = [12] * n_sig
    adc_res = [adc_res] * n_sig if type(adc_res) is int else adc_res
    if verbose:
        print("Signal ADC resolution: {}".format(adc_res))

    # Set the default `adc_zero` if none exists
    if not adc_zero:
        adc_zero = [0] * n_sig
    adc_zero = [adc_zero] * n_sig if type(adc_zero) is int else adc_zero
    if verbose:
        print("Signal ADC zero: {}".format(adc_zero))

    # Set the default `init_value`
    # NOTE: Initial value (and subsequently the digital signal) won't be correct
    # unless the correct `baseline` and `adc_gain` are provided... this is just
    # the best approximation
    if not init_value:
        init_value = p_signal[0, :]
        init_value = baseline + (np.array(adc_gain) * init_value)
        init_value = [int(i) for i in init_value.tolist()]
    if verbose:
        print("Signal initial value: {}".format(init_value))

    # Set the default `checksum`
    if not checksum:
        checksum = [int(np.sum(v) % 65536) for v in np.transpose(p_signal)]
    if verbose:
        print("Signal checksum: {}".format(checksum))

    # Set the default `block_size`
    if not block_size:
        block_size = [0] * n_sig
    block_size = [block_size] * n_sig if type(block_size) is int else block_size
    if verbose:
        print("Signal block size: {}".format(block_size))

    # Convert array to floating point
    p_signal = p_signal.astype("float64")

    # Either return the record or generate the record and header files
    # if requested
    if record_only:
        # Create the record from the input and generated values
        record = Record(
            record_name=record_name,
            n_sig=n_sig,
            fs=fs,
            samps_per_frame=samps_per_frame,
            counter_freq=counter_freq,
            base_counter=base_counter,
            sig_len=sig_len,
            base_time=base_time,
            base_date=base_date,
            comments=comments,
            sig_name=sig_name,
            p_signal=p_signal,
            d_signal=None,
            e_p_signal=None,
            e_d_signal=None,
            file_name=dat_file_name,
            fmt=fmt,
            skew=skew,
            byte_offset=byte_offset,
            adc_gain=adc_gain,
            baseline=baseline,
            units=units,
            adc_res=adc_res,
            adc_zero=adc_zero,
            init_value=init_value,
            checksum=checksum,
            block_size=block_size,
        )
        if verbose:
            print("Record generated successfully")
        return record

    else:
        # Write the information to a record and header file
        wrsamp(
            record_name=record_name,
            fs=fs,
            units=units,
            sig_name=sig_name,
            p_signal=p_signal,
            fmt=fmt,
            adc_gain=adc_gain,
            baseline=baseline,
            comments=comments,
            base_time=base_time,
            base_date=base_date,
        )
        if verbose:
            print("File generated successfully")


def csv2ann(
    file_name,
    extension="atr",
    fs=None,
    record_only=False,
    time_onset=True,
    header=True,
    delimiter=",",
    verbose=False,
):
    """
    Read a CSV/TSV/etc. file and return either an `Annotation` object with the
    annotation descriptors as attributes or write an annotation file.

    Parameters
    ----------
    file_name : str
        The name of the CSV file to be read, including the '.csv' file
        extension. If the argument contains any path delimiter characters, the
        argument will be interpreted as PATH/BASE_RECORD. Both relative and
        absolute paths are accepted. The BASE_RECORD file name will be used to
        name the annotation file with the desired extension.
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
        raise Exception(
            """The number of columns in the CSV was not
                        recognized."""
        )

    # Remove extension from input file name
    file_name = file_name.split(".")[0]
    if time_onset:
        if not fs:
            raise Exception(
                """`fs` must be provided if `time_onset` is True
                            since it is required to convert time onsets to
                            samples"""
            )
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
            record_name=file_name,
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
            file_name,
            extension,
            sample=sample,
            symbol=symbol,
            subtype=subtype,
            chan=chan,
            num=num,
            aux_note=aux_note,
            fs=fs,
        )
        if verbose:
            print("Finished writing Annotation file")
