import datetime
import os
import struct

import numpy as np

from wfdb.io.record import rdrecord


def wfdb_to_mat(
    record_name, pn_dir=None, sampfrom=0, sampto=None, channels=None
):
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
        The name of the input WFDB record to be read.
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
    >>> wfdb2mat('100', pn_dir='pwave')

    The output file name is 100m.mat and 100m.hea

    """
    record = rdrecord(
        record_name, pn_dir=pn_dir, sampfrom=sampfrom, sampto=sampto
    )
    record_name_out = record_name.split(os.sep)[-1].replace("-", "_") + "m"

    # Some variables describing the format of the .mat file
    field_version = 256  # 0x0100 or 256
    endian_indicator = b"IM"  # little endian
    master_type = 14  # matrix
    sub1_type = 6  # UINT32
    sub2_type = 5  # INT32
    sub3_type = 1  # INT8
    sub1_class = 6  # double precision array

    # Determine if we can write 8-bit unsigned samples, or if 16 or 32 bits
    # are needed per sample
    bytes_per_element = 1
    for i in range(record.n_sig):
        if record.adc_res[i] > 0:
            if record.adc_res[i] > 16:
                bytes_per_element = 4
            elif (record.adc_res[i] > 8) and (bytes_per_element < 2):
                bytes_per_element = 2
        else:
            # adc_res not specified.. try to guess from format
            if (record.fmt[i] == "24") or (record.fmt[i] == "32"):
                bytes_per_element = 4
            elif (record.fmt[i] != "80") and (bytes_per_element < 2):
                bytes_per_element = 2

    if bytes_per_element == 1:
        sub4_type = 2  # MAT8
        out_type = "<u1"  # np.uint8
        wfdb_type = "80"  # Offset binary form (80)
        offset = 128  # Offset between sample values and the raw
        # byte/word values as interpreted by Matlab/Octave
    elif bytes_per_element == 2:
        sub4_type = 3  # MAT16
        out_type = "<i2"  # np.int16
        wfdb_type = "16"  # Align with byte boundary (16)
        offset = 0  # Offset between sample values and the raw
        # byte/word values as interpreted by Matlab/Octave
    else:
        sub4_type = 5  # MAT32
        out_type = "<i4"  # np.int32
        wfdb_type = "32"  # Align with byte boundary (32)
        offset = 0  # Offset between sample values and the raw
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
    output_file = record_name_out + ".mat"
    with open(output_file, "wb") as f:
        # Descriptive text (124 bytes)
        f.write(struct.pack("<124s", b"MATLAB 5.0"))
        # Version (2 bytes)
        f.write(struct.pack("<H", field_version))
        # Endian indicator (2 bytes)
        f.write(struct.pack("<2s", endian_indicator))

        # Master tag data type (4 bytes)
        f.write(struct.pack("<I", master_type))
        # Master tag number of bytes (4 bytes)
        # Number of bytes of data element
        #     = (8 + 8) + (8 + 8) + (8 + 8) + (8 + bytes_of_data)
        #     = 56 + bytes_of_data
        f.write(struct.pack("<I", master_bytes))

        # Matrix data has 4 subelements (5 if imaginary):
        #     Array flags, dimensions array, array name, real part
        # Each subelement has its own subtag, and subdata

        # Subelement 1: Array flags
        # Subtag 1: data type (4 bytes)
        f.write(struct.pack("<I", sub1_type))
        # Subtag 1: number of bytes (4 bytes)
        f.write(struct.pack("<I", 8))
        # Value class indication the MATLAB data type (8 bytes)
        f.write(struct.pack("<Q", sub1_class))

        # Subelement 2: Rows and columns
        # Subtag 2: data type (4 bytes)
        f.write(struct.pack("<I", sub2_type))
        # Subtag 2: number of bytes (4 bytes)
        f.write(struct.pack("<I", 8))
        # Number of signals (4 bytes)
        f.write(struct.pack("<I", record.n_sig))
        # Number of rows (4 bytes)
        f.write(struct.pack("<I", desired_length))

        # Subelement 3: Array name
        # Subtag 3: data type (4 bytes)
        f.write(struct.pack("<I", sub3_type))
        # Subtag 3: number of bytes (4 bytes)
        f.write(struct.pack("<I", 3))
        # Subtag 3: name of the array (8 bytes)
        f.write(struct.pack("<8s", b"val"))

        # Subelement 4: Signal data
        # Subtag 4: data type (4 bytes)
        f.write(struct.pack("<I", sub4_type))
        # Subtag 4: number of bytes (4 bytes)
        f.write(struct.pack("<I", bytes_of_data))

        # Total size of everything before actual data:
        #     128 byte header
        #     + 8 byte master tag
        #     + 56 byte subelements (48 byte default + 8 byte name)
        #     = 192

        # Copy the selected data into the .mat file
        out_data = (
            record.p_signal * record.adc_gain
            + record.baseline
            - record.adc_zero
        )
        # Cast the data to the correct type base on the bytes_per_element
        out_data = np.around(out_data).astype(out_type)
        # out_data should be [r1c1, r1c2, r2c1, r2c2, etc.]
        out_data = out_data.flatten()
        out_fmt = "<%sh" % len(out_data)
        f.write(struct.pack(out_fmt, *out_data))

    # Display some useful information
    if record.base_time is None:
        if record.base_date is None:
            datetime_string = "[None]"
        else:
            datetime_string = "[{}]".format(
                record.base_date.strftime("%d/%m/%Y")
            )
    else:
        if record.base_date is None:
            datetime_string = "[{}]".format(
                record.base_time.strftime("%H:%M:%S.%f")
            )
        else:
            datetime_string = "[{} {}]".format(
                record.base_time.strftime("%H:%M:%S.%f"),
                record.base_date.strftime("%d/%m/%Y"),
            )

    print("Source: record {}\t\tStart: {}".format(record_name, datetime_string))
    print(
        "val has {} rows (signals) and {} columns (samples/signal)".format(
            record.n_sig, desired_length
        )
    )
    duration_string = str(
        datetime.timedelta(seconds=desired_length / record.fs)
    )
    print("Duration: {}".format(duration_string))
    print(
        "Sampling frequency: {} Hz\tSampling interval: {} sec".format(
            record.fs, 1 / record.fs
        )
    )
    print(
        "{:<7}{:<20}{:<17}{:<10}{:<10}".format(
            "Row", "Signal", "Gain", "Base", "Units"
        )
    )
    record.sig_name = [s.replace(" ", "_") for s in record.sig_name]
    for i in range(record.n_sig):
        print(
            "{:<7}{:<20}{:<17}{:<10}{:<10}".format(
                i,
                record.sig_name[i],
                record.adc_gain[i],
                record.baseline[i] - record.adc_zero[i] + offset,
                record.units[i],
            )
        )

    # Modify the record file to reflect the new data
    num_channels = record.n_sig if (channels is None) else len(channels)
    record.record_name = record_name_out
    record.n_sig = num_channels
    record.samps_per_frame = num_channels * [1]
    record.file_name = num_channels * [output_file]
    record.fmt = num_channels * [wfdb_type]
    record.byte_offset = num_channels * [192]
    record.baseline = [
        b - record.adc_zero[i] for i, b in enumerate(record.baseline)
    ]
    record.adc_zero = num_channels * [0]
    record.init_value = out_data[: record.n_sig].tolist()

    # Write the header file RECm.hea
    record.wrheader()
    # Append the following lines to create a signature
    with open(record_name_out + ".hea", "a") as f:
        f.write("#Creator: wfdb2mat\n")
        f.write("#Source: record {}\n".format(record_name))
