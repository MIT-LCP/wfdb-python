import datetime
import functools
import math
import os
import posixpath
import struct

import numpy as np
import pandas as pd

from wfdb.io.annotation import Annotation, format_ann_from_df, wrann
from wfdb.io.record import Record, rdrecord, SIG_UNITS
from wfdb.io import _url
from wfdb.io import download


def read_edf(
    record_name,
    pn_dir=None,
    header_only=False,
    verbose=False,
    rdedfann_flag=False,
    encoding="iso8859-1",
):
    """
    Read a EDF format file into a WFDB Record.

    Many EDF files contain signals at widely varying sampling frequencies.
    `read_edf` handles these properly, but the default behavior of most WFDB
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
    performed by `read_edf` is no longer necessary. However, one can still use
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
    header_only : bool, optional
        Whether to only return the header information (True) or not (False).
        If true, this function will only return `['fs', 'sig_len', 'n_sig',
        'base_date', 'base_time', 'units', 'sig_name', 'comments']`.
    verbose : bool, optional
        Whether to print all the information read about the file (True) or
        not (False).
    rdedfann_flag : bool, optional
        Whether the function is being called by `rdedfann` or the user. If it
        is being called by the user and the file has annotations, then warn
        them that the EDF file has annotations and that they should use
        `rdedfann` instead.
    encoding : str, optional
        The encoding to use for strings in the header. Although the edf
        specification requires ascii strings, some files do not adhere to it.

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
    >>> record = read_edf('x001_FAROS.edf',
                           pn_dir='simultaneous-measurements/raw_data')

    """
    if pn_dir is not None:

        if "." not in pn_dir:
            dir_list = pn_dir.split("/")
            pn_dir = posixpath.join(
                dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
            )

        file_url = posixpath.join(download.PN_INDEX_URL, pn_dir, record_name)
        # Currently must download file for MNE to read it though can give the
        # user the option to delete it immediately afterwards
        with _url.openurl(file_url, "rb") as f:
            open(record_name, "wb").write(f.read())

    # Open the desired file
    edf_file = open(record_name, mode="rb")

    # Version of this data format (8 bytes)
    version = struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)

    # Check to see that the input is an EDF file. (This check will detect
    # most but not all other types of files.)
    if version != "0       ":
        raise Exception(
            "Input does not appear to be EDF -- no conversion attempted"
        )
    else:
        if verbose:
            print("EDF version number: {}".format(version.strip()))

    # Local patient identification (80 bytes)
    patient_id = struct.unpack("<80s", edf_file.read(80))[0].decode(encoding)
    if verbose:
        print("Patient ID: {}".format(patient_id))

    # Local recording identification (80 bytes)
    # Bob Kemp recommends using this field to encode the start date
    # including an abbreviated month name in English and a full (4-digit)
    # year, as is done here if this information is available in the input
    # record. EDF+ requires this.
    record_id = struct.unpack("<80s", edf_file.read(80))[0].decode(encoding)
    if verbose:
        print("Recording ID: {}".format(record_id))

    # Start date of recording (dd.mm.yy) (8 bytes)
    start_date = struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)
    if verbose:
        print("Recording Date: {}".format(start_date))
    start_day, start_month, start_year = [int(i) for i in start_date.split(".")]
    # This should work for a while
    if start_year < 1970:
        start_year += 1900
    if start_year < 1970:
        start_year += 100

    # Start time of recording (hh.mm.ss) (8 bytes)
    start_time = struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)
    if verbose:
        print("Recording Time: {}".format(start_time))
    start_hour, start_minute, start_second = [
        int(i) for i in start_time.split(".")
    ]

    # Number of bytes in header (8 bytes)
    header_bytes = int(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding))
    if verbose:
        print("Number of bytes in header record: {}".format(header_bytes))

    # Reserved (44 bytes)
    reserved_notes = (
        struct.unpack("<44s", edf_file.read(44))[0].decode(encoding).strip()
    )
    if reserved_notes[:5] == "EDF+C":
        # The file is EDF compatible and will work without issue
        # See: Bob Kemp, Jesus Olivan, European data format ‘plus’ (EDF+), an
        #      EDF alike standard format for the exchange of physiological
        #      data, Clinical Neurophysiology, Volume 114, Issue 9, 2003,
        #      Pages 1755-1761, ISSN 1388-2457
        pass
    elif reserved_notes[:5] == "EDF+D":
        raise Exception(
            "EDF+ File: interrupted data records (not currently supported)"
        )
    else:
        if verbose:
            print("Free Space: {}".format(reserved_notes))

    # Number of blocks (-1 if unknown) (8 bytes)
    num_blocks = int(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding))
    if verbose:
        print("Number of data records: {}".format(num_blocks))
    if num_blocks == -1:
        raise Exception(
            "Number of data records in unknown (not currently supported)"
        )

    # Duration of a block, in seconds (8 bytes)
    block_duration = float(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding))
    if verbose:
        print(
            "Duration of each data record in seconds: {}".format(block_duration)
        )
    if block_duration <= 0.0:
        block_duration = 1.0

    # Number of signals (4 bytes)
    n_sig = int(struct.unpack("<4s", edf_file.read(4))[0].decode(encoding))
    if verbose:
        print("Number of signals: {}".format(n_sig))
    if n_sig < 1:
        raise Exception("Done: not any signals left to read")

    # Label (e.g., EEG FpzCz or Body temp) (16 bytes each)
    sig_name = []
    for _ in range(n_sig):
        temp_sig = struct.unpack("<16s", edf_file.read(16))[0].decode(encoding).strip()
        if temp_sig == "EDF Annotations" and not rdedfann_flag:
            print(
                "*** This may be an EDF+ Annotation file instead, please see "
                "the `rdedfann` function. ***"
            )
        sig_name.append(temp_sig)
    if verbose:
        print("Signal Labels: {}".format(sig_name))

    # Transducer type (e.g., AgAgCl electrode) (80 bytes each)
    transducer_types = []
    for _ in range(n_sig):
        transducer_types.append(
            struct.unpack("<80s", edf_file.read(80))[0].decode(encoding).strip()
        )
    if verbose:
        print("Transducer Types: {}".format(transducer_types))

    # Physical dimension (e.g., uV or degreeC) (8 bytes each)
    physical_dims = []
    for _ in range(n_sig):
        physical_dims.append(
            struct.unpack("<8s", edf_file.read(8))[0].decode(encoding).strip()
        )
    if verbose:
        print("Physical Dimensions: {}".format(physical_dims))

    # Physical minimum (e.g., -500 or 34) (8 bytes each)
    physical_min = np.array([])
    for _ in range(n_sig):
        physical_min = np.append(
            physical_min,
            float(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)),
        )
    if verbose:
        print("Physical Minimums: {}".format(physical_min))

    # Physical maximum (e.g., 500 or 40) (8 bytes each)
    physical_max = np.array([])
    for _ in range(n_sig):
        physical_max = np.append(
            physical_max,
            float(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)),
        )
    if verbose:
        print("Physical Maximums: {}".format(physical_max))

    # Digital minimum (e.g., -2048) (8 bytes each)
    digital_min = np.array([])
    for _ in range(n_sig):
        digital_min = np.append(
            digital_min,
            float(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)),
        )
    if verbose:
        print("Digital Minimums: {}".format(digital_min))

    # Digital maximum (e.g., 2047) (8 bytes each)
    digital_max = np.array([])
    for _ in range(n_sig):
        digital_max = np.append(
            digital_max,
            float(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding)),
        )
    if verbose:
        print("Digital Maximums: {}".format(digital_max))

    # Prefiltering (e.g., HP:0.1Hz LP:75Hz) (80 bytes each)
    prefilter_info = []
    for _ in range(n_sig):
        prefilter_info.append(
            struct.unpack("<80s", edf_file.read(80))[0].decode(encoding).strip()
        )
    if verbose:
        print("Prefiltering Information: {}".format(prefilter_info))

    # Number of samples per block (8 bytes each)
    samps_per_block = []
    for _ in range(n_sig):
        samps_per_block.append(
            int(struct.unpack("<8s", edf_file.read(8))[0].decode(encoding))
        )
    if verbose:
        print("Number of Samples per Record: {}".format(samps_per_block))

    # The last 32*nsig bytes in the header are unused
    for _ in range(n_sig):
        struct.unpack("<32s", edf_file.read(32))[0].decode(encoding)

    # Pre-process the acquired data before creating the record
    record_name_out = (
        record_name.split(os.sep)[-1].replace("-", "_").replace(".edf", "")
    )
    sample_rate = [int(i / block_duration) for i in samps_per_block]
    fs = functools.reduce(math.gcd, sample_rate)
    samps_per_frame = [int(s / min(samps_per_block)) for s in samps_per_block]
    sig_len = int(fs * num_blocks * block_duration)
    base_time = datetime.time(start_hour, start_minute, start_second)
    base_date = datetime.date(start_year, start_month, start_day)
    file_name = n_sig * [record_name_out + ".dat"]
    fmt = n_sig * ["16"]
    skew = n_sig * [None]
    byte_offset = n_sig * [None]
    adc_gain_all = (digital_max - digital_min) / (physical_max - physical_min)
    adc_gain = [float(format(a, ".12g")) for a in adc_gain_all]
    baseline = (digital_max - (physical_max * adc_gain_all) + 1).astype("int64")

    units = n_sig * [""]
    for i, f in enumerate(physical_dims):
        if f == "n/a":
            label = sig_name[i].lower().split()[0]
            if label in list(SIG_UNITS.keys()):
                units[i] = SIG_UNITS[label]
            else:
                units[i] = "n/a"
        else:
            f = f.replace("µ", "u")  # Maybe more weird symbols to check for?
            if f == "":
                units[i] = "mV"
            else:
                units[i] = f

    adc_res = [int(math.log2(f)) for f in (digital_max - digital_min)]
    adc_zero = [int(f) for f in ((digital_max + 1 + digital_min) / 2)]
    block_size = n_sig * [0]
    base_datetime = datetime.datetime(
        start_year,
        start_month,
        start_day,
        start_hour,
        start_minute,
        start_second,
    )
    base_time = datetime.time(
        base_datetime.hour, base_datetime.minute, base_datetime.second
    )
    base_date = datetime.date(
        base_datetime.year, base_datetime.month, base_datetime.day
    )

    if header_only:
        return {
            "fs": fs,
            "sig_len": sig_len,
            "n_sig": n_sig,
            "base_date": base_date,
            "base_time": base_time,
            "units": units,
            "sig_name": sig_name,
            "comments": [],
        }

    sig_data = np.empty((sig_len, n_sig))
    temp_sig_data = np.fromfile(edf_file, dtype=np.int16)
    temp_sig_data = temp_sig_data.reshape((-1, sum(samps_per_block)))
    temp_all_sigs = np.hsplit(temp_sig_data, np.cumsum(samps_per_block)[:-1])
    for i in range(n_sig):
        # Check if `samps_per_frame` has all equal values
        if samps_per_frame.count(samps_per_frame[0]) == len(samps_per_frame):
            sig_data[:, i] = (
                temp_all_sigs[i].flatten() - baseline[i]
            ) / adc_gain_all[i]
        else:
            temp_sig_data = temp_all_sigs[i].flatten()
            if samps_per_frame[i] == 1:
                sig_data[:, i] = (temp_sig_data - baseline[i]) / adc_gain_all[i]
            else:
                for j in range(sig_len):
                    start_ind = j * samps_per_frame[i]
                    stop_ind = start_ind + samps_per_frame[i]
                    sig_data[j, i] = np.mean(
                        (temp_sig_data[start_ind:stop_ind] - baseline[i])
                        / adc_gain_all[i]
                    )

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

    init_value = [int(s[0, 0]) for s in temp_all_sigs]
    checksum = [
        int(np.sum(v) % 65536) for v in np.transpose(sig_data)
    ]  # not all values correct?

    record = Record(
        record_name=record_name_out,
        n_sig=n_sig,
        fs=fs,
        samps_per_frame=samps_per_frame,
        counter_freq=None,
        base_counter=None,
        sig_len=sig_len,
        base_time=base_time,
        base_date=base_date,
        comments=[],
        sig_name=sig_name,  # Remove whitespace to make compatible later?
        p_signal=sig_data,
        d_signal=None,
        e_p_signal=None,
        e_d_signal=None,
        file_name=n_sig * [record_name_out + ".dat"],
        fmt=n_sig * ["16"],
        skew=n_sig * [None],
        byte_offset=n_sig * [None],
        adc_gain=adc_gain,
        baseline=baseline,
        units=units,
        adc_res=[int(math.log2(f)) for f in (digital_max - digital_min)],
        adc_zero=[int(f) for f in ((digital_max + 1 + digital_min) / 2)],
        init_value=init_value,
        checksum=checksum,
        block_size=n_sig * [0],
    )

    record.base_datetime = base_datetime

    return record


def wfdb_to_edf(
    record_name,
    pn_dir=None,
    sampfrom=0,
    sampto=None,
    channels=None,
    output_filename="",
    edf_plus=False,
):
    """
    These programs convert EDF (European Data Format) files into
    WFDB-compatible files (as used in PhysioNet) and vice versa. European
    Data Format (EDF) was originally designed for storage of polysomnograms.

    Note that WFDB format does not include a standard way to specify the
    transducer type or the prefiltering specification; these parameters are
    not preserved by these conversion programs. Also note that use of the
    standard signal and unit names specified for EDF is permitted but not
    enforced by `wfdb_to_edf`.

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
    >>> wfdb.wfdb_to_edf('100', pn_dir='pwave')

    The output file name is '100.edf'

    """
    record = rdrecord(
        record_name,
        pn_dir=pn_dir,
        sampfrom=sampfrom,
        sampto=sampto,
        smooth_frames=False,
    )
    record_name_out = record_name.split(os.sep)[-1].replace("-", "_")

    # Maximum data block length, in bytes
    edf_max_block = 61440

    # Convert to the expected month name formatting
    month_names = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]

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
    while bytes_per_block > edf_max_block:
        frames_per_block /= 10
        bytes_per_block = samples_per_frame * 2 * frames_per_block

    seconds_per_block = int(frames_per_block / frames_per_second)

    if (frames_per_block < 1) and (bytes_per_block < edf_max_block / 60):
        # The number of frames/minute
        frames_per_block = frames_per_minute
        bytes_per_block = 2 * samples_per_frame * frames_per_block
        seconds_per_block = 60

    if bytes_per_block > edf_max_block:
        print(
            (
                "Can't convert record %s to EDF: EDF blocks can't be larger "
                "than {} bytes, but each input frame requires {} bytes.  Use "
                "'channels' to select a subset of the input signals or trim "
                "using 'sampfrom' and 'sampto'."
            ).format(edf_max_block, samples_per_frame * 2)
        )

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
        if record.adc_res[i] < 1:
            # Guess the ADC resolution based on format
            if record.fmt[i] == "24":
                temp_adc_res = 24
            elif record.fmt[i] == "32":
                temp_adc_res = 32
            elif record.fmt[i] == "80":
                temp_adc_res = 8
            elif record.fmt[i] == "212":
                temp_adc_res = 12
            elif (record.fmt[i] == "310") or (record.fmt[i] == "311"):
                temp_adc_res = 10
            else:
                temp_adc_res = 16
        else:
            temp_adc_res = record.adc_res[i]
        # Determine the physical and digital extrema
        digital_max.append(
            int(record.adc_zero[i] + (1 << (temp_adc_res - 1)) - 1)
        )
        digital_min.append(int(record.adc_zero[i] - (1 << (temp_adc_res - 1))))
        physical_max.append(
            (digital_max[i] - record.baseline[i]) / record.adc_gain[i]
        )
        physical_min.append(
            (digital_min[i] - record.baseline[i]) / record.adc_gain[i]
        )

    # The maximum record name length to write is 80 bytes
    if len(record_name_out) > 80:
        record_name_write = record_name_out[:79] + "\0"
    else:
        record_name_write = record_name_out

    # The maximum seconds per block length to write is 8 bytes
    if len(str(seconds_per_block)) > 8:
        seconds_per_block_write = seconds_per_block[:7] + "\0"
    else:
        seconds_per_block_write = seconds_per_block

    # The maximum signal name length to write is 16 bytes
    sig_name_write = len(record.sig_name) * []
    for s in record.sig_name:
        if len(s) > 16:
            sig_name_write.append(s[:15] + "\0")
        else:
            sig_name_write.append(s)

    # The maximum units length to write is 8 bytes
    units_write = len(record.units) * []
    for s in record.units:
        if len(s) > 8:
            units_write.append(s[:7] + "\0")
        else:
            units_write.append(s)

    # Configure the output datetime
    if hasattr("record", "base_datetime"):
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
        for j, sig in enumerate(temp_data):
            ind_start = i * samps_per_record[j]
            ind_stop = (i + 1) * samps_per_record[j]
            out_data.extend(sig[ind_start:ind_stop].tolist())
    out_data = np.array(out_data, dtype=np.int16)

    # Start writing the file
    if output_filename == "":
        output_filename = record_name_out + ".edf"

    with open(output_filename, "wb") as f:

        print(
            "Converting record {} to {} ({} mode)".format(
                record_name, output_filename, "EDF+" if edf_plus else "EDF"
            )
        )

        # Version of this data format (8 bytes)
        f.write(struct.pack("<8s", b"0").replace(b"\x00", b"\x20"))

        # Local patient identification (80 bytes)
        f.write(
            struct.pack(
                "<80s", "{}".format(record_name_write).encode("ascii")
            ).replace(b"\x00", b"\x20")
        )

        # Local recording identification (80 bytes)
        # Bob Kemp recommends using this field to encode the start date
        # including an abbreviated month name in English and a full (4-digit)
        # year, as is done here if this information is available in the input
        # record. EDF+ requires this.
        if hasattr("record", "base_datetime"):
            f.write(
                struct.pack(
                    "<80s",
                    "Startdate {}-{}-{}".format(
                        start_day, month_names[start_month - 1], start_year
                    ).encode("ascii"),
                ).replace(b"\x00", b"\x20")
            )
        else:
            f.write(
                struct.pack("<80s", b"Startdate not recorded").replace(
                    b"\x00", b"\x20"
                )
            )
        if edf_plus:
            print("WARNING: EDF+ requires start date (not specified)")

        # Start date of recording (dd.mm.yy) (8 bytes)
        f.write(
            struct.pack(
                "<8s",
                "{:02d}.{:02d}.{:02d}".format(
                    start_day, start_month, start_year % 100
                ).encode("ascii"),
            ).replace(b"\x00", b"\x20")
        )

        # Start time of recording (hh.mm.ss) (8 bytes)
        f.write(
            struct.pack(
                "<8s",
                "{:02d}.{:02d}.{:02d}".format(
                    start_hour, start_minute, start_second
                ).encode("ascii"),
            ).replace(b"\x00", b"\x20")
        )

        # Number of bytes in header (8 bytes)
        f.write(
            struct.pack(
                "<8s", "{:d}".format(header_bytes).encode("ascii")
            ).replace(b"\x00", b"\x20")
        )

        # Reserved (44 bytes)
        if edf_plus:
            f.write(struct.pack("<44s", b"EDF+C").replace(b"\x00", b"\x20"))
        else:
            f.write(struct.pack("<44s", b"").replace(b"\x00", b"\x20"))

        # Number of blocks (-1 if unknown) (8 bytes)
        f.write(
            struct.pack(
                "<8s", "{:d}".format(num_blocks).encode("ascii")
            ).replace(b"\x00", b"\x20")
        )

        # Duration of a block, in seconds (8 bytes)
        f.write(
            struct.pack(
                "<8s", "{:g}".format(seconds_per_block_write).encode("ascii")
            ).replace(b"\x00", b"\x20")
        )

        # Number of signals (4 bytes)
        f.write(
            struct.pack(
                "<4s", "{:d}".format(record.n_sig).encode("ascii")
            ).replace(b"\x00", b"\x20")
        )

        # Label (e.g., EEG FpzCz or Body temp) (16 bytes each)
        for i in sig_name_write:
            f.write(
                struct.pack("<16s", "{}".format(i).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # Transducer type (e.g., AgAgCl electrode) (80 bytes each)
        for _ in range(record.n_sig):
            f.write(
                struct.pack("<80s", b"transducer type not recorded").replace(
                    b"\x00", b"\x20"
                )
            )

        # Physical dimension (e.g., uV or degreeC) (8 bytes each)
        for i in units_write:
            f.write(
                struct.pack("<8s", "{}".format(i).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # Physical minimum (e.g., -500 or 34) (8 bytes each)
        for pmin in physical_min:
            f.write(
                struct.pack("<8s", "{:g}".format(pmin).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # Physical maximum (e.g., 500 or 40) (8 bytes each)
        for pmax in physical_max:
            f.write(
                struct.pack("<8s", "{:g}".format(pmax).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # Digital minimum (e.g., -2048) (8 bytes each)
        for dmin in digital_min:
            f.write(
                struct.pack("<8s", "{:d}".format(dmin).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # Digital maximum (e.g., 2047) (8 bytes each)
        for dmax in digital_max:
            f.write(
                struct.pack("<8s", "{:d}".format(dmax).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # Prefiltering (e.g., HP:0.1Hz LP:75Hz) (80 bytes each)
        for _ in range(record.n_sig):
            f.write(
                struct.pack("<80s", b"prefiltering not recorded").replace(
                    b"\x00", b"\x20"
                )
            )

        # Number of samples per block (8 bytes each)
        for spr in samps_per_record:
            f.write(
                struct.pack("<8s", "{:d}".format(spr).encode("ascii")).replace(
                    b"\x00", b"\x20"
                )
            )

        # The last 32*nsig bytes in the header are unused
        for _ in range(record.n_sig):
            f.write(struct.pack("<32s", b"").replace(b"\x00", b"\x20"))

        # Write the data blocks
        out_data.tofile(f, format="%d")

        # Add the buffer
        correct_bytes = num_blocks * sum(samps_per_record)
        current_bytes = len(out_data)
        num_to_write = correct_bytes - current_bytes
        for i in range(num_to_write):
            f.write(b"\x00\x80")

    print("Header block size: {:d} bytes".format((record.n_sig + 1) * 256))
    print(
        "Data block size: {:g} seconds ({:d} frames or {:d} bytes)".format(
            seconds_per_block, int(frames_per_block), int(bytes_per_block)
        )
    )
    print(
        "Recording length: {:d} ({:d} data blocks, {:d} frames, {:d} bytes)".format(
            sum(
                [
                    num_blocks,
                    num_blocks * int(frames_per_block),
                    num_blocks * bytes_per_block,
                ]
            ),
            num_blocks,
            num_blocks * int(frames_per_block),
            num_blocks * bytes_per_block,
        )
    )
    print(
        "Total length of file to be written: {:d} bytes".format(
            int((record.n_sig + 1) * 256 + num_blocks * bytes_per_block)
        )
    )

    if edf_plus:
        print(
            (
                "WARNING: EDF+ requires the subject's gender, birthdate, and name, as "
                "well as additional information about the recording that is not usually "
                "available. This information is not saved in the output file even if "
                "available. EDF+ also requires the use of standard names for signals and "
                "for physical units;  these requirements are not enforced by this program. "
                "To make the output file fully EDF+ compliant, its header must be edited "
                "manually."
            )
        )

        if "EDF-Annotations" not in record.sig_name:
            print(
                "WARNING: The output file does not include EDF annotations, which are required for EDF+."
            )

    # Check that all characters in the header are valid (printable ASCII
    # between 32 and 126 inclusive). Note that this test does not prevent
    # generation of files containing invalid characters; it merely warns
    # the user if this has happened.
    header_test = open(output_filename, "rb").read((record.n_sig + 1) * 256)
    for i, val in enumerate(header_test):
        if (val < 32) or (val > 126):
            print(
                "WARNING: output contains an invalid character, {}, at byte {}".format(
                    val, i
                )
            )


def rdedfann(
    record_name,
    pn_dir=None,
    delete_file=True,
    info_only=True,
    record_only=False,
    verbose=False,
    encoding="iso8859-1",
):
    """
    This program returns the annotation information from an EDF+ file
    containing annotations (with the signal name given as 'EDF Annotations').
    The information that is returned if `info_only` is set to True is:
        {
            'onset_time': list of %H:%M:%S.fff strings denoting the annotation
                          onset times,
            'sample_num': list of integers denoting the annotation onset
                          sample numbers,
            'comment': list of comments (`aux_note`) for the annotations,
            'duration': list of floats denoting the duration of the event
        }
    Else, this function will return either the WFDB Annotation format of the
    information of the file if `record_only` is set to True, or nothing if
    neither are set to True though a WFDB Annotation file will be created.

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
    info_only : bool, optional
        Return, strictly, the information contained in the file as formatted
        by the original WFDB package. Must not be True if `record_only` is
        True.
    record_only : bool, optional
        Whether to only return the annotation information (True) or not
        (False). If False, this function will generate a WFDB-formatted
        annotation file. If True, it will return the object returned if that
        file were read with `rdann`. Must not be True if `info_only` is True.
    verbose : bool, optional
        Whether to print all the information read about the file (True) or
        not (False).
    encoding : str, optional
        The encoding to use for strings in the header. Although the edf
        specification requires ascii strings, some files do not adhere to it.

    Returns
    -------
    N/A : dict, Annotation, optional
        If 'info_only' is set to True, return all of the annotation
        information needed to generate WFDB-formatted annotation files.
        If 'record_only' is set to True, return the WFDB-formatted annotation
        object generated by the `rdann` output. If none are set to True, write
        the WFDB-formatted annotation file.

    Notes
    -----
    The entire file is composed of (seen here:
    https://www.edfplus.info/specs/edfplus.html#edfplusannotations):

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
    ns * 16 ascii : ns * label (must be 'EDF Annotations')
    ns * 80 ascii : ns * transducer type (must be whitespace)
    ns * 8 ascii : ns * physical dimension (must be whitespace)
    ns * 8 ascii : ns * physical minimum (e.g. -500 or 34, different than physical maximum)
    ns * 8 ascii : ns * physical maximum (e.g. 500 or 40, different than physical minimum)
    ns * 8 ascii : ns * digital minimum (must be -32768)
    ns * 8 ascii : ns * digital maximum (must be 32767)
    ns * 80 ascii : ns * prefiltering (must be whitespace)
    ns * 8 ascii : ns * nr of samples in each data record
    ns * 32 ascii : ns * reserved

    ANNOTATION RECORD

    Examples
    --------
    >>> ann_info = wfdb.rdedfann('sample-data/test_edfann.edf')

    """
    # Some preliminary checks
    if info_only and record_only:
        raise Exception(
            "Both `info_only` and `record_only` are set. Only one "
            "can be set at a time."
        )

    # According to the EDF+ docs:
    #   "The coding is EDF compatible in the sense that old EDF software would
    #    simply treat this 'EDF Annotations' signal as if it were a (strange-
    #    looking) ordinary signal"
    rec = read_edf(
        record_name,
        pn_dir=pn_dir,
        delete_file=delete_file,
        record_only=True,
        rdedfann_flag=True,
    )

    # Convert from array of integers to ASCII strings
    annotation_string = ""
    for chunk in rec.p_signal.flatten().astype(np.int64):
        if chunk + 1 == 0:
            continue
        else:
            adjusted_hex = hex(
                struct.unpack("<H", struct.pack(">H", chunk + 1))[0]
            )
            annotation_string += bytes.fromhex(adjusted_hex[2:]).decode(encoding)
            # Remove all of the whitespace
            for rep in ["\x00", "\x14", "\x15"]:
                annotation_string = annotation_string.replace(rep, " ")

    # Parse the resulting annotation string
    onsets = []
    onset_times = []
    sample_nums = []
    comments = []
    durations = []
    all_anns = annotation_string.split("+")
    for ann in all_anns:
        if ann == "":
            continue
        try:
            ann_split = ann.strip().split(" ")
            onset = float(ann_split[0])
            hours, rem = divmod(onset, 3600)
            minutes, seconds = divmod(rem, 60)
            onset_time = f"{hours:02.0f}:{minutes:02.0f}:{seconds:06.3f}"
            sample_num = int(onset * rec.sig_len)
            duration = float(ann_split[1])
            comment = " ".join(ann_split[2:])
            if verbose:
                print(
                    f"{onset_time}\t{sample_num}\t{comment}\t\tduration: {duration}"
                )
            onsets.append(onset)
            onset_times.append(onset_time)
            sample_nums.append(sample_num)
            comments.append(comment)
            durations.append(duration)
        except IndexError:
            continue

    if info_only:
        return {
            "onset_time": onset_times,
            "sample_num": sample_nums,
            "comment": comments,
            "duration": durations,
        }
    else:
        df_in = pd.DataFrame(
            data={
                "onset": onsets,
                "duration": durations,
                "description": comments,
            }
        )
        df_out = format_ann_from_df(df_in)
        # Remove extension from input file name
        record_name = record_name.split(os.sep)[-1].split(".")[0]
        extension = "atr"
        fs = rec.fs
        sample = (df_out["onset"].to_numpy() * fs).astype(np.int64)
        # Assume each annotation is a comment
        symbol = ['"'] * len(df_out.index)
        subtype = np.array([22] * len(df_out.index))
        # Assume each annotation belongs with the 1st channel
        chan = np.array([0] * len(df_out.index))
        num = np.array([0] * len(df_out.index))
        aux_note = df_out["description"].tolist()

        if record_only:
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
            )
