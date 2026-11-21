import datetime
import os
import struct

import numpy as np

from wfdb.io._signal import _checksum, _digi_nan
from wfdb.io.record import rdrecord


def wfdb_to_mat(record_name, pn_dir=None, sampfrom=0, sampto=None, channels=None):
    """Convert a PhysioNet record to a Matlab .mat file.

    This program converts the signals of any PhysioNet record (or one in any compatible
    format) into a .mat file that can be read directly using any version of Matlab, and
    a short text file containing information about the signals (names, gains, baselines,
    units, sampling frequency, and start time/date if known). If the input record name
    is REC, the output files are RECm.mat and RECm.hea. The output files can also be
    read by any WFDB application as record RECm.

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
    otherwise. A wider type is used when shifted samples do not fit the type
    suggested by the header. Conversion fails if they cannot fit in 32 bits.

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
    The entire file is composed of::

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
        record_name,
        pn_dir=pn_dir,
        sampfrom=sampfrom,
        sampto=sampto,
        channels=channels,
        physical=False,
    )
    record_name_out = os.path.basename(os.fspath(record_name)).replace("-", "_") + "m"

    adc_res = [r if r is not None else 0 for r in record.adc_res]
    adc_zero = [z if z is not None else 0 for z in record.adc_zero]
    baseline = [
        b if b is not None else adc_zero[i] for i, b in enumerate(record.baseline)
    ]

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
        if adc_res[i] > 0:
            if adc_res[i] > 16:
                bytes_per_element = 4
            elif (adc_res[i] > 8) and (bytes_per_element < 2):
                bytes_per_element = 2
        else:
            # adc_res not specified.. try to guess from format
            if record.fmt[i] in ("24", "32", "524"):
                bytes_per_element = 4
            elif (record.fmt[i] not in ("80", "508")) and (bytes_per_element < 2):
                bytes_per_element = 2

    shifted_data = record.d_signal.astype(np.int64) - np.asarray(
        adc_zero, dtype=np.int64
    )
    valid_min = valid_max = None
    for i, fmt in enumerate(record.fmt):
        channel = shifted_data[:, i]
        orig_nan = _digi_nan(fmt)
        if orig_nan is not None:
            channel = channel[record.d_signal[:, i] != orig_nan]
        if channel.size:
            channel_min = int(channel.min())
            channel_max = int(channel.max())
            valid_min = (
                channel_min if valid_min is None else min(valid_min, channel_min)
            )
            valid_max = (
                channel_max if valid_max is None else max(valid_max, channel_max)
            )

    # The lowest digital value is reserved for missing samples in each format.
    # Promote the output type if the shifted valid samples do not fit.
    for width, max_valid in ((1, 127), (2, 32767), (4, 2147483647)):
        if width >= bytes_per_element and (
            valid_min is None or (valid_min >= -max_valid and valid_max <= max_valid)
        ):
            bytes_per_element = width
            break
    else:
        raise ValueError("Shifted samples exceed the 32-bit WFDB range")

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
    # Snip record
    if record.sig_len > max_length:
        raise Exception("Can't write .mat file: data size exceeds 2GB limit")

    # Bytes of actual data
    bytes_of_data = bytes_per_element * record.n_sig * record.sig_len
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
        f.write(struct.pack("<I", record.sig_len))

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

        # Copy the selected data into the .mat file, preserving missing samples
        out_nan = _digi_nan(wfdb_type)
        for i, fmt in enumerate(record.fmt):
            orig_nan = _digi_nan(fmt)
            if orig_nan is not None:
                shifted_data[record.d_signal[:, i] == orig_nan, i] = out_nan
        # Cast the data to the correct type based on bytes_per_element
        # out_data is C-contiguous: [r1c1, r1c2, r2c1, r2c2, etc.]
        out_data = np.ascontiguousarray(shifted_data + offset, dtype=out_type)
        f.write(out_data.tobytes())
        if bytes_remain:
            f.write(b"\x00" * (8 - bytes_remain))

    # Display some useful information
    if record.base_time is None:
        if record.base_date is None:
            datetime_string = "[None]"
        else:
            datetime_string = "[{}]".format(record.base_date.strftime("%d/%m/%Y"))
    else:
        if record.base_date is None:
            datetime_string = "[{}]".format(record.base_time.strftime("%H:%M:%S.%f"))
        else:
            datetime_string = "[{} {}]".format(
                record.base_time.strftime("%H:%M:%S.%f"),
                record.base_date.strftime("%d/%m/%Y"),
            )

    print(f"Source: record {record_name}\t\tStart: {datetime_string}")
    print(
        f"val has {record.n_sig} rows (signals) and {record.sig_len} columns (samples/signal)"
    )
    duration_string = str(datetime.timedelta(seconds=record.sig_len / record.fs))
    print(f"Duration: {duration_string}")
    print(f"Sampling frequency: {record.fs} Hz\tSampling interval: {1 / record.fs} sec")
    print(
        "{:<7}{:<20}{:<17}{:<10}{:<10}".format("Row", "Signal", "Gain", "Base", "Units")
    )
    record.sig_name = [
        s.replace(" ", "_") if s is not None else None for s in record.sig_name
    ]
    for i in range(record.n_sig):
        sig_name = record.sig_name[i] if record.sig_name[i] is not None else ""
        unit = record.units[i] if record.units[i] is not None else ""
        print(
            f"{i:<7}{sig_name:<20}{record.adc_gain[i]:<17}{baseline[i] - adc_zero[i] + offset:<10}{unit:<10}"
        )

    # Modify the record file to reflect the new data
    record.record_name = record_name_out
    record.samps_per_frame = record.n_sig * [1]
    record.skew = None
    record.file_name = record.n_sig * [output_file]
    record.fmt = record.n_sig * [wfdb_type]
    record.byte_offset = record.n_sig * [192]
    record.adc_res = adc_res
    record.baseline = [baseline[i] - adc_zero[i] for i in range(record.n_sig)]
    record.adc_zero = record.n_sig * [0]
    record.block_size = record.n_sig * [0]
    record.init_value = shifted_data[0].tolist()
    record.checksum = [_checksum(s) for s in shifted_data.T]

    # Write the header file RECm.hea
    record.wrheader()
    # Append the following lines to create a signature
    with open(record_name_out + ".hea", "a") as f:
        f.write("#Creator: wfdb2mat\n")
        f.write(f"#Source: record {record_name}\n")
