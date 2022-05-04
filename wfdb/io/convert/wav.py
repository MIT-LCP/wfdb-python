import os
import posixpath
import struct

import numpy as np

from wfdb.io import Record
from wfdb.io import download
from wfdb.io import _url
from wfdb.io.record import rdrecord


def wfdb_to_wav(
    record_name,
    pn_dir=None,
    sampfrom=0,
    sampto=None,
    channels=None,
    output_filename="",
    write_header=False,
):
    """
    This program converts a WFDB record into .wav format (format 16, multiplexed
    signals, with embedded header information).

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

    Examples
    --------
    >>> wfdb_to_wav('100', pn_dir='pwave')

    The output file name is '100.wav'

    """
    record = rdrecord(
        record_name,
        pn_dir=pn_dir,
        sampfrom=sampfrom,
        sampto=sampto,
        smooth_frames=False,
    )
    record_name_out = record_name.split(os.sep)[-1].replace("-", "_")

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
    if output_filename != "":
        if not output_filename.endswith(".wav"):
            raise Exception("Name of output file must end in '.wav'")
    else:
        output_filename = record_name_out + ".wav"

    with open(output_filename, "wb") as f:
        # Write the WAV file identifier
        f.write(struct.pack(">4s", b"RIFF"))
        # Write the number of bytes to follow in the file
        # (num_samps*frame_length) sample bytes, and 36 more bytes of miscellaneous embedded header
        f.write(struct.pack("<I", file_bytes))
        # Descriptor for the format of the file
        f.write(struct.pack(">8s", b"WAVEfmt "))
        # Number of bytes to follow in the format chunk
        f.write(struct.pack("<I", 16))
        # The format tag
        f.write(struct.pack("<H", 1))
        # The number of signals
        f.write(struct.pack("<H", record.n_sig))
        # The samples per second
        f.write(struct.pack("<I", samps_per_second))
        # The number of bytes per second
        f.write(struct.pack("<I", samps_per_second * frame_length))
        # The length of each frame
        f.write(struct.pack("<H", frame_length))
        # The number of bits per samples
        f.write(struct.pack("<H", bits_per_sample))
        # The descriptor to indicate that the data information is next
        f.write(struct.pack(">4s", b"data"))
        # The number of bytes in the signal data chunk
        f.write(struct.pack("<I", chunk_bytes))
        # Write the signal data... the closest I can get to the original implementation
        # Mismatched elements: 723881 / 15400000 (4.7%)
        # Max absolute difference: 2
        # Max relative difference: 0.00444444
        #  x: array([ -322,  3852, -9246, ...,     0,     0,     0], dtype=int16)
        #  y: array([ -322,  3852, -9246, ...,     0,     0,     0], dtype=int16)
        sig_data = (
            np.left_shift(np.subtract(record.adc(), offset), shift)
            .reshape((1, -1))
            .astype(np.int16)
        )
        sig_data.tofile(f)

    # If asked to write the accompanying header file
    if write_header:
        record.adc_zero = record.n_sig * [0]
        record.adc_res = record.n_sig * [16]
        record.adc_gain = [
            (r * (1 << shift[i])) for i, r in enumerate(record.adc_gain)
        ]
        record.baseline = [
            (b - offset[i]) for i, b in enumerate(record.baseline)
        ]
        record.baseline = [
            (b * (1 << shift[i])) for i, b in enumerate(record.baseline)
        ]
        record.file_name = record.n_sig * [record_name_out + ".wav"]
        record.block_size = record.n_sig * [0]
        record.fmt = record.n_sig * ["16"]
        record.samps_per_fram = record.n_sig * [1]
        record.init_value = sig_data[0][: record.n_sig].tolist()
        record.byte_offset = record.n_sig * [44]
        # Write the header file
        record.wrheader()


def read_wav(record_name, pn_dir=None, delete_file=True, record_only=False):
    """
    Convert .wav (format 16, multiplexed signals, with embedded header
    information) formatted files to WFDB format. See here for more details about
    the formatting of a .wav file: http://soundfile.sapp.org/doc/WaveFormat/.

    This process may not work with some .wav files that are encoded using
    variants of the original .wav format that are not WFDB-compatible. In
    principle, this program should be able to recognize such files by their
    format codes, and it will produce an error message in such cases. If
    the format code is incorrect, however, `read_wav` may not recognize that
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

    Returns
    -------
    record : Record
        A WFDB Record object.

    Notes
    -----
    Files that can be processed successfully using `read_wav` always have exactly
    three chunks (a header chunk, a format chunk, and a data chunk).  In .wav
    files, binary data are always written in little-endian format (least
    significant byte first). The format of `read_wav`'s input files is as follows:

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
    >>> record = read_wav('sample-data/SC4001E0-PSG.wav')

    """
    if not record_name.endswith(".wav"):
        raise Exception("Name of the input file must end in .wav")

    if pn_dir is not None:

        if "." not in pn_dir:
            dir_list = pn_dir.split("/")
            pn_dir = posixpath.join(
                dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
            )

        file_url = posixpath.join(download.PN_INDEX_URL, pn_dir, record_name)
        # Currently must download file to read it though can give the
        # user the option to delete it immediately afterwards
        with _url.openurl(file_url, "rb") as f:
            open(record_name, "wb").write(f.read())

    wave_file = open(record_name, mode="rb")
    record_name_out = (
        record_name.split(os.sep)[-1].replace("-", "_").replace(".wav", "")
    )

    chunk_ID = "".join(
        [s.decode() for s in struct.unpack(">4s", wave_file.read(4))]
    )
    if chunk_ID != "RIFF":
        raise Exception("{} is not a .wav-format file".format(record_name))

    correct_chunk_size = os.path.getsize(record_name) - 8
    chunk_size = struct.unpack("<I", wave_file.read(4))[0]
    if chunk_size != correct_chunk_size:
        raise Exception(
            "Header chunk has incorrect length (is {} should be {})".format(
                chunk_size, correct_chunk_size
            )
        )

    fmt = struct.unpack(">4s", wave_file.read(4))[0].decode()
    if fmt != "WAVE":
        raise Exception("{} is not a .wav-format file".format(record_name))

    subchunk1_ID = struct.unpack(">4s", wave_file.read(4))[0].decode()
    if subchunk1_ID != "fmt ":
        raise Exception("Format chunk missing or corrupt")

    subchunk1_size = struct.unpack("<I", wave_file.read(4))[0]
    audio_format = struct.unpack("<H", wave_file.read(2))[0]
    if audio_format > 1:
        print("PCM has compression of {}".format(audio_format))

    if (subchunk1_size != 16) or (audio_format != 1):
        raise Exception("Unsupported format {}".format(audio_format))

    num_channels = struct.unpack("<H", wave_file.read(2))[0]
    if num_channels == 1:
        print("Reading Mono formatted .wav file...")
    elif num_channels == 2:
        print("Reading Stereo formatted .wav file...")
    else:
        print("Reading {}-channel formatted .wav file...".format(num_channels))

    sample_rate = struct.unpack("<I", wave_file.read(4))[0]
    print("Sample rate: {}".format(sample_rate))
    byte_rate = struct.unpack("<I", wave_file.read(4))[0]
    print("Byte rate: {}".format(byte_rate))
    block_align = struct.unpack("<H", wave_file.read(2))[0]
    print("Block align: {}".format(block_align))
    bits_per_sample = struct.unpack("<H", wave_file.read(2))[0]
    print("Bits per sample: {}".format(bits_per_sample))
    # I wish this were more precise but unfortunately some information
    # is lost in .wav files which is needed for these calculations
    if bits_per_sample <= 8:
        adc_res = 8
        adc_gain = 12.5
    elif bits_per_sample <= 16:
        adc_res = 16
        adc_gain = 6400
    else:
        raise Exception(
            "Unsupported resolution ({} bits/sample)".format(bits_per_sample)
        )

    if block_align != (num_channels * int(adc_res / 8)):
        raise Exception(
            "Format chunk of {} has incorrect frame length".format(block_align)
        )

    subchunk2_ID = struct.unpack(">4s", wave_file.read(4))[0].decode()
    if subchunk2_ID != "data":
        raise Exception("Format chunk missing or corrupt")

    correct_subchunk2_size = os.path.getsize(record_name) - 44
    subchunk2_size = struct.unpack("<I", wave_file.read(4))[0]
    if subchunk2_size != correct_subchunk2_size:
        raise Exception(
            "Data chunk has incorrect length.. (is {} should be {})".format(
                subchunk2_size, correct_subchunk2_size
            )
        )
    sig_len = int(subchunk2_size / block_align)

    sig_data = (
        np.fromfile(wave_file, dtype=np.int16).reshape((-1, num_channels))
        / (2 * adc_res)
    ).astype(np.int16)

    init_value = [int(s[0]) for s in np.transpose(sig_data)]
    checksum = [
        int(np.sum(v) % 65536) for v in np.transpose(sig_data)
    ]  # not all values correct?

    if pn_dir is not None and delete_file:
        os.remove(record_name)

    record = Record(
        record_name=record_name_out,
        n_sig=num_channels,
        fs=num_channels * [sample_rate],
        samps_per_frame=num_channels * [1],
        counter_freq=None,
        base_counter=None,
        sig_len=sig_len,
        base_time=None,
        base_date=None,
        comments=[],
        sig_name=num_channels * [None],
        p_signal=None,
        d_signal=sig_data,
        e_p_signal=None,
        e_d_signal=None,
        file_name=num_channels * [record_name_out + ".dat"],
        fmt=num_channels * ["16" if (adc_res == 16) else "80"],
        skew=num_channels * [None],
        byte_offset=num_channels * [None],
        adc_gain=num_channels * [adc_gain],
        baseline=num_channels * [0 if (adc_res == 16) else 128],
        units=num_channels * [None],
        adc_res=num_channels * [adc_res],
        adc_zero=num_channels * [0 if (adc_res == 16) else 128],
        init_value=init_value,
        checksum=checksum,
        block_size=num_channels * [0],
    )

    return record
