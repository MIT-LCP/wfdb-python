"""
A module for a potentially better designed Record and Header class.
"""
from dataclasses import dataclass, field
import datetime
import os
import re

from typing import List, Optional

# from wfdb.io import _header

# @dataclass
# class Data:
#     a: List[int] = field(default_factory=list)
#     b: List[int] = field(default_factory=list)


# @dataclass
# class EE:
#     a: Optional[List[int]] = None

@dataclass
class A:
    x :int
    y: int

@dataclass
class B(A):
    z: int


@dataclass
class RecordSpecFields:
    # Record specification fields
    name: Optional[str] = None
    # n_seg == 1 does NOT mean the same as it being missing.
    n_seg: Optional[int] = None
    n_sig: Optional[int] = None
    fs: Optional[float] = None
    counter_freq: Optional[float] = None
    base_counter: Optional[float] = None
    sig_len: Optional[int] = None
    base_time: Optional[datetime.time] = None
    base_date: Optional[datetime.date] = None



_rx_record = re.compile(
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


class HeaderSyntaxError(ValueError):
    """Invalid syntax found in a WFDB header file."""


def parse_record_fields(record_line: str):
    """
    Extract fields from a record line string into a dictionary.

    Parameters
    ----------
    record_line : str
        The record line contained in the header file

    Returns
    -------
    record_fields : dict
        The fields for the given record line.

    """
    # Dictionary for record fields
    record_fields = RecordSpecFields()

    # Read string fields from record line
    match = _rx_record.match(record_line)
    if match is None:
        raise HeaderSyntaxError("invalid syntax in record line")
    (
        record_fields.record_name,
        record_fields.n_seg,
        record_fields.n_sig,
        record_fields.fs,
        record_fields.counter_freq,
        record_fields.base_counter,
        record_fields.sig_len,
        record_fields.base_time,
        record_fields.base_date,
    ) = match.groups()

    # Replace missing optional values with their read defaults. Set empty string fields to None.
    record_fields.n_seg = (int(record_fields.n_seg) if record_fields.n_seg else None)

    if (record_fields.n_sig <= 0){
        raise ValueError()
    }

    record_fields.fs = record_fields.fs or 250

    record_fields.counter_freq = record_fields.counter_freq or record_fields.fs
    record_fields.base_counter = record_fields.base_counter or 0
    record_fields.sig_len = record_fields.sig_len or None
    record_fields.base_time = record_fields.base_time or



    for field in RECORD_SPECS.index:
        # Replace empty strings with their read defaults (which are
        # mostly None)
        if record_fields[field] == "":
            record_fields[field] = RECORD_SPECS.loc[field, "read_default"]
        # Typecast non-empty strings for non-string (numerical/datetime)
        # fields
        else:
            if RECORD_SPECS.loc[field, "allowed_types"] == int_types:
                record_fields[field] = int(record_fields[field])
            elif RECORD_SPECS.loc[field, "allowed_types"] == float_types:
                record_fields[field] = float(record_fields[field])
                # cast fs to an int if it is close
                if field == "fs":
                    fs = float(record_fields["fs"])
                    if round(fs, 8) == float(int(fs)):
                        fs = int(fs)
                    record_fields["fs"] = fs
            elif field == "base_time":
                record_fields["base_time"] = wfdb_strptime(
                    record_fields["base_time"]
                )
            elif field == "base_date":
                record_fields["base_date"] = datetime.datetime.strptime(
                    record_fields["base_date"], "%d/%m/%Y"
                ).date()

    # This is not a standard WFDB field, but is useful to set.
    if record_fields["base_date"] and record_fields["base_time"]:
        record_fields["base_datetime"] = datetime.datetime.combine(
            record_fields["base_date"], record_fields["base_time"]
        )

    return record_fields


@dataclass
class SignalSpecFields:
    # Signal specification fields. One value per channel.
    file_name: List[str] = field(default_factory=list)
    fmt: List[str] = field(default_factory=list)
    samps_per_frame: List[int] = field(default_factory=list)
    skew: List[int] = field(default_factory=list)
    byte_offset: List[int] = field(default_factory=list)
    adc_gain: List[float] = field(default_factory=list)
    baseline: List[int] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    adc_res: List[int] = field(default_factory=list)
    adc_zero: List[int] = field(default_factory=list)
    init_value: List[int] = field(default_factory=list)
    checksum: List[int] = field(default_factory=list)
    block_size: List[int] = field(default_factory=list)
    sig_name: List[str] = field(default_factory=list)


@dataclass
class SegmentSpecFields:
    # Segment specification fields. One value per segment.
    seg_name: List[str]
    seg_len: List[int]


@dataclass
class RecordInfo(RecordSpecFields, SignalSpecFields, SegmentSpecFields):
    """
    A data class for storing metadata about a Record.
    Contains all fields from a header file.
    """

    comments: List[str]

    @property
    def base_datetime(self) -> Optional[datetime.datetime]:
        if self.base_date and self.base_time:
            return datetime.datetime.combine(self.base_date, self.base_time)
        return None


# def read_header(record_name: str) -> RecordInfo:
#     """
#     Read a WFDB header
#     """
#     dir_name, base_record_name = os.path.split(record_name)
#     dir_name = os.path.abspath(dir_name)

#     # Read the header file. Separate comment and non-comment lines
#     header_lines, comment_lines = _header._read_header_lines(
#         base_record_name, dir_name
#     )

#     record_fields = _header._parse_record_fields(header_lines[0])

#     info = RecordInfo(comments=[line.strip(" \t#") for line in comment_lines])
#     return info


class Record:
    """
    A new WFDB Record.
    """

    def __init__(self, info: RecordInfo):
        self.info = info
