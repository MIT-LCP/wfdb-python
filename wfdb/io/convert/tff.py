"""
Module for reading ME6000 .tff format files.

http://www.biomation.com/kin/me6000.htm

"""
import datetime
import os
import struct

import numpy as np


def rdtff(file_name, cut_end=False):
    """
    Read values from a tff file.

    Parameters
    ----------
    file_name : str
        Name of the .tff file to read.
    cut_end : bool, optional
        If True, cuts out the last sample for all channels. This is for
        reading files which appear to terminate with the incorrect
        number of samples (ie. sample not present for all channels).

    Returns
    -------
    signal : ndarray
        A 2d numpy array storing the physical signals from the record.
    fields : dict
        A dictionary containing several key attributes of the read record.
    markers : ndarray
        A 1d numpy array storing the marker locations.
    triggers : ndarray
        A 1d numpy array storing the trigger locations.

    Notes
    -----
    This function is slow because tff files may contain any number of
    escape sequences interspersed with the signals. There is no way to
    know the number of samples/escape sequences beforehand, so the file
    is inefficiently parsed a small chunk at a time.

    It is recommended that you convert your tff files to WFDB format.

    """
    file_size = os.path.getsize(file_name)
    with open(file_name, "rb") as fp:
        fields, file_fields = _rdheader(fp)
        signal, markers, triggers = _rdsignal(
            fp,
            file_size=file_size,
            header_size=file_fields["header_size"],
            n_sig=file_fields["n_sig"],
            bit_width=file_fields["bit_width"],
            is_signed=file_fields["is_signed"],
            cut_end=cut_end,
        )
    return signal, fields, markers, triggers


def _rdheader(fp):
    """
    Read header info of the windaq file.

    Parameters
    ----------
    fp : file IO object
        The input header file to be read.

    Returns
    -------
    fields : dict
        For interpreting the waveforms.
    file_fields : dict
        For reading the signal samples.

    """
    tag = None
    # The '2' tag indicates the end of tags.
    while tag != 2:
        # For each header element, there is a tag indicating data type,
        # followed by the data size, followed by the data itself. 0's
        # pad the content to the nearest 4 bytes. If data_len=0, no pad.
        tag = struct.unpack(">H", fp.read(2))[0]
        data_size = struct.unpack(">H", fp.read(2))[0]
        pad_len = (4 - (data_size % 4)) % 4
        pos = fp.tell()
        # Currently, most tags will be ignored...
        # storage method
        if tag == 1001:
            storage_method = fs = struct.unpack("B", fp.read(1))[0]
            storage_method = {0: "recording", 1: "manual", 2: "online"}[
                storage_method
            ]
        # fs, unit16
        elif tag == 1003:
            fs = struct.unpack(">H", fp.read(2))[0]
        # sensor type
        elif tag == 1007:
            # Each byte contains information for one channel
            n_sig = data_size
            channel_data = struct.unpack(">%dB" % data_size, fp.read(data_size))
            # The documentation states: "0 : Channel is not used"
            # This means the samples are NOT saved.
            channel_map = (
                (1, 1, "emg"),
                (15, 30, "goniometer"),
                (31, 46, "accelerometer"),
                (47, 62, "inclinometer"),
                (63, 78, "polar_interface"),
                (79, 94, "ecg"),
                (95, 110, "torque"),
                (111, 126, "gyrometer"),
                (127, 142, "sensor"),
            )
            sig_name = []
            # The number range that the data lies between gives the
            # channel
            for data in channel_data:
                # Default case if byte value falls outside of channel map
                base_name = "unknown"
                # Unused channel
                if data == 0:
                    n_sig -= 1
                    break
                for item in channel_map:
                    if item[0] <= data <= item[1]:
                        base_name = item[2]
                        break
                existing_count = [base_name in name for name in sig_name].count(
                    True
                )
                sig_name.append("%s_%d" % (base_name, existing_count))
        # Display scale. Probably not useful.
        elif tag == 1009:
            # 100, 500, 1000, 2500, or 8500uV
            display_scale = struct.unpack(">I", fp.read(4))[0]
        # sample format, uint8
        elif tag == 3:
            sample_fmt = struct.unpack("B", fp.read(1))[0]
            is_signed = bool(sample_fmt >> 7)
            # ie. 8 or 16 bits
            bit_width = sample_fmt & 127
        # Measurement start time - seconds from 1.1.1970 UTC
        elif tag == 101:
            n_seconds = struct.unpack(">I", fp.read(4))[0]
            base_datetime = datetime.datetime.utcfromtimestamp(n_seconds)
            base_date = base_datetime.date()
            base_time = base_datetime.time()
        # Measurement start time - minutes from UTC
        elif tag == 102:
            n_minutes = struct.unpack(">h", fp.read(2))[0]
        # Go to the next tag
        fp.seek(pos + data_size + pad_len)
    header_size = fp.tell()
    # For interpreting the waveforms
    fields = {
        "fs": fs,
        "n_sig": n_sig,
        "sig_name": sig_name,
        "base_time": base_time,
        "base_date": base_date,
    }
    # For reading the signal samples
    file_fields = {
        "header_size": header_size,
        "n_sig": n_sig,
        "bit_width": bit_width,
        "is_signed": is_signed,
    }
    return fields, file_fields


def _rdsignal(fp, file_size, header_size, n_sig, bit_width, is_signed, cut_end):
    """
    Read the signal.

    Parameters
    ----------
    fp : file IO object
        The input header file to be read.
    file_size : int
        Size of the file in bytes.
    header_size : int
        Size of the header file in bytes.
    n_sig : int
        The number of signals contained in the dat file.
    bit_width : int
        The number of bits necessary to represent the number in binary.
    is_signed : bool
        Whether the number is signed (True) or not (False).
    cut_end : bool, optional
        If True, enables reading the end of files which appear to terminate
        with the incorrect number of samples (ie. sample not present for all channels),
        by checking and skipping the reading the end of such files.
        Checking this option makes reading slower.

    Returns
    -------
    signal : ndarray
        Tranformed expanded signal into uniform signal.
    markers : ndarray
        A 1d numpy array storing the marker locations.
    triggers : ndarray
        A 1d numpy array storing the trigger locations.

    """
    # Cannot initially figure out signal length because there
    # are escape sequences.
    fp.seek(header_size)
    signal_size = file_size - header_size
    byte_width = int(bit_width / 8)
    # numpy dtype
    dtype = str(byte_width)
    if is_signed:
        dtype = "i" + dtype
    else:
        dtype = "u" + dtype
    # big endian
    dtype = ">" + dtype
    # The maximum possible samples given the file size
    # All channels must be present
    max_samples = int(signal_size / byte_width)
    max_samples = max_samples - max_samples % n_sig
    # Output information
    signal = np.empty(max_samples, dtype=dtype)
    markers = []
    triggers = []
    # Number of (total) samples read
    sample_num = 0

    # Read one sample for all channels at a time
    if cut_end:
        stop_byte = file_size - n_sig * byte_width + 1
        while fp.tell() < stop_byte:
            chunk = fp.read(2)
            sample_num = _get_sample(
                fp, chunk, n_sig, dtype, signal, markers, triggers, sample_num
            )
    else:
        while True:
            chunk = fp.read(2)
            if not chunk:
                break
            sample_num = _get_sample(
                fp, chunk, n_sig, dtype, signal, markers, triggers, sample_num
            )

    # No more bytes to read. Reshape output arguments.
    signal = signal[:sample_num]
    signal = signal.reshape((-1, n_sig))
    markers = np.array(markers, dtype="int")
    triggers = np.array(triggers, dtype="int")
    return signal, markers, triggers


def _get_sample(fp, chunk, n_sig, dtype, signal, markers, triggers, sample_num):
    """
    Get the total number of samples in the signal.

    Parameters
    ----------
    fp : file IO object
        The input header file to be read.
    chunk : str
        The data currently being processed.
    n_sig : int
        The number of signals contained in the dat file.
    dtype : str
        String numpy dtype used to store the signal of the given
        resolution.
    signal : ndarray
        Tranformed expanded signal into uniform signal.
    markers : ndarray
        A 1d numpy array storing the marker locations.
    triggers : ndarray
        A 1d numpy array storing the trigger locations.
    sample_num : int
        The total number of samples in the signal.

    Returns
    -------
    sample_num : int
        The total number of samples in the signal.

    """
    tag = struct.unpack(">h", chunk)[0]
    # Escape sequence
    if tag == -32768:
        # Escape sequence structure: int16 marker, uint8 type,
        # uint8 length, uint8 * length data, padding % 2
        escape_type = struct.unpack("B", fp.read(1))[0]
        data_len = struct.unpack("B", fp.read(1))[0]
        # Marker*
        if escape_type == 1:
            # *In manual mode, this could be block start/top time.
            # But we are it is just a single time marker.
            markers.append(sample_num / n_sig)
        # Trigger
        elif escape_type == 2:
            triggers.append(sample_num / n_sig)
        fp.seek(data_len + data_len % 2, 1)
    # Regular samples
    else:
        fp.seek(-2, 1)
        signal[sample_num : sample_num + n_sig] = np.fromfile(
            fp, dtype=dtype, count=n_sig
        )
        sample_num += n_sig
    return sample_num
