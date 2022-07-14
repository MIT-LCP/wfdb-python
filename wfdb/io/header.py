"""
Module for parsing header files.

This module will eventually replace _header.py

"""
import datetime
import re
from typing import List, Tuple


class HeaderSyntaxError(ValueError):
    """Invalid syntax found in a WFDB header file."""


# Record line pattern. Format:
# RECORD_NAME/NUM_SEG NUM_SIG SAMP_FREQ/COUNT_FREQ(BASE_COUNT_VAL) SAMPS_PER_SIG BASE_TIME BASE_DATE
rx_record = re.compile(
    r"""
    [ \t]* (?P<record_name>[-\w]+)
           /?(?P<n_seg>\d*)
    [ \t]+ (?P<n_sig>\d+)
    [ \t]* (?P<fs>\d*\.?\d*)
           /*(?P<counter_freq>-?\d*\.?\d*)
           \(?(?P<base_counter>-?\d*\.?\d*)\)?
    [ \t]* (?P<sig_len>\d*)
    [ \t]* (?P<base_time>\d{,2}:?\d{,2}:?\d{,2}\.?\d{,6})
    [ \t]* (?P<base_date>\d{,2}/?\d{,2}/?\d{,4})
    """,
    re.VERBOSE,
)

# Signal line pattern. Format:
# FILE_NAME FORMATxSAMP_PER_FRAME:SKEW+BYTE_OFFSET ADC_GAIN(BASELINE)/UNITS ADC_RES ADC_ZERO CHECKSUM BLOCK_SIZE DESCRIPTION
rx_signal = re.compile(
    r"""
    [ \t]* (?P<file_name>~?[-\w]*\.?[\w]*)
    [ \t]+ (?P<fmt>\d+)
           x?(?P<samps_per_frame>\d*)
           :?(?P<skew>\d*)
           \+?(?P<byte_offset>\d*)
    [ \t]* (?P<adc_gain>-?\d*\.?\d*e?[\+-]?\d*)
           \(?(?P<baseline>-?\d*)\)?
           /?(?P<units>[\w\^\-\?%\/]*)
    [ \t]* (?P<adc_res>\d*)
    [ \t]* (?P<adc_zero>-?\d*)
    [ \t]* (?P<init_value>-?\d*)
    [ \t]* (?P<checksum>-?\d*)
    [ \t]* (?P<block_size>\d*)
    [ \t]* (?P<sig_name>[\S]?[^\t\n\r\f\v]*)
    """,
    re.VERBOSE,
)

# Segment line
rx_segment = re.compile(
    r"""
    [ \t]* (?P<seg_name>[-\w]*~?)
    [ \t]+ (?P<seg_len>\d+)
    """,
    re.VERBOSE,
)


def wfdb_strptime(time_string: str) -> datetime.time:
    """
    Given a time string in an acceptable WFDB format, return
    a datetime.time object.

    Valid formats: SS, MM:SS, HH:MM:SS, all with and without microsec.

    Parameters
    ----------
    time_string : str
        The time to be converted to a datetime.time object.

    Returns
    -------
    datetime.time object
        The time converted from str format.

    """
    n_colons = time_string.count(":")

    if n_colons == 0:
        time_fmt = "%S"
    elif n_colons == 1:
        time_fmt = "%M:%S"
    elif n_colons == 2:
        time_fmt = "%H:%M:%S"

    if "." in time_string:
        time_fmt += ".%f"

    return datetime.datetime.strptime(time_string, time_fmt).time()


def parse_header_content(
    header_content: str,
) -> Tuple[List[str], List[str]]:
    """
    Parse the text of a header file.

    Parameters
    ----------
    header_content: str
        The string content of the full header file

    Returns
    -------
    header_lines : List[str]
        A list of all the non-comment lines
    comment_lines : List[str]
        A list of all the comment lines

    """
    header_lines, comment_lines = [], []
    for line in header_content.splitlines():
        line = line.strip()
        # Comment line
        if line.startswith("#"):
            comment_lines.append(line)
        # Non-empty non-comment line = header line.
        elif line:
            header_lines.append(line)

    return header_lines, comment_lines
