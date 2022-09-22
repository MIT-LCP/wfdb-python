import datetime
from typing import Collection, List, Tuple

import numpy as np
import pandas as pd

from wfdb.io import _signal
from wfdb.io import util
from wfdb.io.header import HeaderSyntaxError, rx_record, rx_segment, rx_signal

"""
Notes
-----
In the original WFDB package, certain fields have default values, but
not all of them. Some attributes need to be present for core
functionality, i.e. baseline, whereas others are not essential, yet have
defaults, i.e. base_time.

This inconsistency has likely resulted in the generation of incorrect
files, and general confusion. This library aims to make explicit,
whether certain fields are present in the file, by setting their values
to None if they are not written in, unless the fields are essential, in
which case an actual default value will be set.

The read vs write default values are different for 2 reasons:
1. We want to force the user to be explicit with certain important
   fields when writing WFDB records fields, without affecting
   existing WFDB headers when reading.
2. Certain unimportant fields may be dependencies of other
   important fields. When writing, we want to fill in defaults
   so that the user doesn't need to. But when reading, it should
   be clear that the fields are missing.

"""
int_types = (int, np.int64, np.int32, np.int16, np.int8)
float_types = (float, np.float64, np.float32) + int_types

_SPECIFICATION_COLUMNS = [
    "allowed_types",
    "delimiter",
    "dependency",
    "write_required",
    "read_default",
    "write_default",
]

RECORD_SPECS = pd.DataFrame(
    index=[
        "record_name",
        "n_seg",
        "n_sig",
        "fs",
        "counter_freq",
        "base_counter",
        "sig_len",
        "base_time",
        "base_date",
    ],
    columns=_SPECIFICATION_COLUMNS,
    dtype="object",
    data=[
        [(str,), "", None, True, None, None],  # record_name
        [int_types, "/", "record_name", True, None, None],  # n_seg
        [int_types, " ", "record_name", True, None, None],  # n_sig
        [float_types, " ", "n_sig", True, 250, None],  # fs
        [float_types, "/", "fs", False, None, None],  # counter_freq
        [float_types, "(", "counter_freq", False, None, None],  # base_counter
        [int_types, " ", "fs", True, None, None],  # sig_len
        [
            (datetime.time,),
            " ",
            "sig_len",
            False,
            None,
            "00:00:00",
        ],  # base_time
        [(datetime.date,), " ", "base_time", False, None, None],  # base_date
    ],
)

SIGNAL_SPECS = pd.DataFrame(
    index=[
        "file_name",
        "fmt",
        "samps_per_frame",
        "skew",
        "byte_offset",
        "adc_gain",
        "baseline",
        "units",
        "adc_res",
        "adc_zero",
        "init_value",
        "checksum",
        "block_size",
        "sig_name",
    ],
    columns=_SPECIFICATION_COLUMNS,
    dtype="object",
    data=[
        [(str,), "", None, True, None, None],  # file_name
        [(str,), " ", "file_name", True, None, None],  # fmt
        [int_types, "x", "fmt", False, 1, None],  # samps_per_frame
        [int_types, ":", "fmt", False, None, None],  # skew
        [int_types, "+", "fmt", False, None, None],  # byte_offset
        [float_types, " ", "fmt", True, 200.0, None],  # adc_gain
        [int_types, "(", "adc_gain", True, 0, None],  # baseline
        [(str,), "/", "adc_gain", True, "mV", None],  # units
        [int_types, " ", "adc_gain", False, None, 0],  # adc_res
        [int_types, " ", "adc_res", False, None, 0],  # adc_zero
        [int_types, " ", "adc_zero", False, None, None],  # init_value
        [int_types, " ", "init_value", False, None, None],  # checksum
        [int_types, " ", "checksum", False, None, 0],  # block_size
        [(str,), " ", "block_size", False, None, None],  # sig_name
    ],
)

SEGMENT_SPECS = pd.DataFrame(
    index=["seg_name", "seg_len"],
    columns=_SPECIFICATION_COLUMNS,
    dtype="object",
    data=[
        [(str), "", None, True, None, None],  # seg_name
        [int_types, " ", "seg_name", True, None, None],  # seg_len
    ],
)

# Specifications of all WFDB header fields, except for comments
FIELD_SPECS = pd.concat((RECORD_SPECS, SIGNAL_SPECS, SEGMENT_SPECS))


class BaseHeaderMixin(object):
    """
    Mixin class with multi-segment header methods. Inherited by Record and
    MultiRecord classes.

    Attributes
    ----------
    N/A

    """

    def get_write_subset(self, spec_type):
        """
        Get a set of fields used to write the header; either 'record'
        or 'signal' specification fields. Helper function for
        `get_write_fields`. Gets the default required fields, the user
        defined fields, and their dependencies.

        Parameters
        ----------
        spec_type : str
            The set of specification fields desired. Either 'record' or
            'signal'.

        Returns
        -------
        write_fields : list or dict
            For record fields,  returns a list of all fields needed. For
            signal fields, it returns a dictionary of all fields needed,
            with keys = field and value = list of channels that must be
            present for the field.

        """
        if spec_type == "record":
            write_fields = []
            record_specs = RECORD_SPECS.copy()

            # Remove the n_seg requirement for single segment items
            if not hasattr(self, "n_seg"):
                record_specs.drop("n_seg", inplace=True)

            for field in record_specs.index[-1::-1]:
                # Continue if the field has already been included
                if field in write_fields:
                    continue
                # If the field is required by default or has been
                # defined by the user
                if (
                    record_specs.loc[field, "write_required"]
                    or getattr(self, field) is not None
                ):
                    req_field = field
                    # Add the field and its recursive dependencies
                    while req_field is not None:
                        write_fields.append(req_field)
                        req_field = record_specs.loc[req_field, "dependency"]
            # Add comments if any
            if getattr(self, "comments") is not None:
                write_fields.append("comments")

        # signal spec field. Need to return a potentially different list for each channel.
        elif spec_type == "signal":
            # List of lists for each channel
            write_fields = []
            signal_specs = SIGNAL_SPECS.copy()

            for ch in range(self.n_sig):
                # The fields needed for this channel
                write_fields_ch = []
                for field in signal_specs.index[-1::-1]:
                    if field in write_fields_ch:
                        continue

                    item = getattr(self, field)
                    # If the field is required by default or has been defined by the user
                    if signal_specs.loc[field, "write_required"] or (
                        item is not None and item[ch] is not None
                    ):
                        req_field = field
                        # Add the field and its recursive dependencies
                        while req_field is not None:
                            write_fields_ch.append(req_field)
                            req_field = signal_specs.loc[
                                req_field, "dependency"
                            ]

                write_fields.append(write_fields_ch)

            # Convert the list of lists to a single dictionary.
            # keys = field and value = list of channels in which the
            # field is required.
            dict_write_fields = {}

            # For fields present in any channel:
            for field in set(
                [i for write_fields_ch in write_fields for i in write_fields_ch]
            ):
                dict_write_fields[field] = []

                for ch in range(self.n_sig):
                    if field in write_fields[ch]:
                        dict_write_fields[field].append(ch)

            write_fields = dict_write_fields

        return write_fields


class HeaderMixin(BaseHeaderMixin):
    """
    Mixin class with single-segment header methods. Inherited by Record class.

    Attributes
    ----------
    N/A

    """

    def set_defaults(self):
        """
        Set defaults for fields needed to write the header if they have
        defaults.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        Notes
        -----
        - This is NOT called by `rdheader`. It is only automatically
          called by the gateway `wrsamp` for convenience.
        - This is also not called by `wrheader` since it is supposed to
          be an explicit function.
        - This is not responsible for initializing the attributes. That
          is done by the constructor.

        See also `set_p_features` and `set_d_features`.

        """
        rfields, sfields = self.get_write_fields()
        for f in rfields:
            self.set_default(f)
        for f in sfields:
            self.set_default(f)

    def wrheader(self, write_dir="", expanded=True):
        """
        Write a WFDB header file. The signals are not used. Before
        writing:
        - Get the fields used to write the header for this instance.
        - Check each required field.
        - Check that the fields are cohesive with one another.

        Parameters
        ----------
        write_dir : str, optional
            The output directory in which the header is written.
        expanded : bool, optional
            Whether the header file should include `samps_per_frame` (this
            should only be true if the signal files are written using
            `expanded=True`).

        Returns
        -------
        N/A

        Notes
        -----
        This function does NOT call `set_defaults`. Essential fields
        must be set beforehand.

        """
        # Get all the fields used to write the header
        # sig_write_fields is a dictionary of
        # {field_name:required_channels}
        rec_write_fields, sig_write_fields = self.get_write_fields()
        if not expanded:
            sig_write_fields.pop("samps_per_frame", None)

        # Check the validity of individual fields used to write the header
        # Record specification fields (and comments)
        for field in rec_write_fields:
            self.check_field(field)

        # Signal specification fields.
        for field in sig_write_fields:
            self.check_field(field, required_channels=sig_write_fields[field])

        # Check the cohesion of fields used to write the header
        self.check_field_cohesion(rec_write_fields, list(sig_write_fields))

        # Write the header file using the specified fields
        self.wr_header_file(rec_write_fields, sig_write_fields, write_dir)

    def get_write_fields(self):
        """
        Get the list of fields used to write the header, separating
        record and signal specification fields. Returns the default
        required fields, the user defined fields, and their dependencies.

        Does NOT include `d_signal` or `e_d_signal`.

        Parameters
        ----------
        N/A

        Returns
        -------
        rec_write_fields : list
            Record specification fields to be written. Includes
            'comment' if present.
        sig_write_fields : dict
            Dictionary of signal specification fields to be written,
            with values equal to the channels that need to be present
            for each field.

        """
        # Record specification fields
        rec_write_fields = self.get_write_subset("record")

        # Add comments if any
        if self.comments != None:
            rec_write_fields.append("comments")

        # Get required signal fields if signals are present.
        self.check_field("n_sig")

        if self.n_sig > 0:
            sig_write_fields = self.get_write_subset("signal")
        else:
            sig_write_fields = None

        return rec_write_fields, sig_write_fields

    def _auto_signal_file_names(self):
        fmt = self.fmt or [None] * self.n_sig
        spf = self.samps_per_frame or [None] * self.n_sig
        num_groups = 0
        group_number = []
        prev_fmt = prev_spf = None
        channels_in_group = 0

        for ch_fmt, ch_spf in zip(fmt, spf):
            if ch_fmt != prev_fmt:
                num_groups += 1
                channels_in_group = 0
            elif ch_fmt in ("508", "516", "524"):
                if channels_in_group >= 8 or ch_spf != prev_spf:
                    num_groups += 1
                    channels_in_group = 0
            group_number.append(num_groups)
            prev_fmt = ch_fmt
            prev_spf = ch_spf

        if num_groups < 2:
            return [self.record_name + ".dat"] * self.n_sig
        else:
            digits = len(str(group_number[-1]))
            return [
                self.record_name + "_" + str(g).rjust(digits, "0") + ".dat"
                for g in group_number
            ]

    def set_default(self, field):
        """
        Set the object's attribute to its default value if it is missing
        and there is a default. Not responsible for initializing the
        attribute. That is done by the constructor.

        Parameters
        ----------
        field : str
            The desired attribute of the object.

        Returns
        -------
        N/A

        """
        # Record specification fields
        if field in RECORD_SPECS.index:
            # Return if no default to set, or if the field is already
            # present.
            if (
                RECORD_SPECS.loc[field, "write_default"] is None
                or getattr(self, field) is not None
            ):
                return
            setattr(self, field, RECORD_SPECS.loc[field, "write_default"])

        # Signal specification fields
        # Setting entire list default, not filling in blanks in lists.
        elif field in SIGNAL_SPECS.index:

            # Specific dynamic case
            if field == "file_name" and self.file_name is None:
                self.file_name = self._auto_signal_file_names()
                return

            item = getattr(self, field)

            # Return if no default to set, or if the field is already
            # present.
            if (
                SIGNAL_SPECS.loc[field, "write_default"] is None
                or item is not None
            ):
                return

            # Set more specific defaults if possible
            if field == "adc_res" and self.fmt is not None:
                self.adc_res = _signal._fmt_res(self.fmt)
                return

            setattr(
                self,
                field,
                [SIGNAL_SPECS.loc[field, "write_default"]] * self.n_sig,
            )

    def check_field_cohesion(self, rec_write_fields, sig_write_fields):
        """
        Check the cohesion of fields used to write the header.

        Parameters
        ----------
        rec_write_fields : list
            List of record specification fields to write.
        sig_write_fields : dict
            Dictionary of signal specification fields to write, values
            being equal to a list of channels to write for each field.

        Returns
        -------
        N/A

        """
        # If there are no signal specification fields, there is nothing to check.
        if self.n_sig > 0:

            # The length of all signal specification fields must match n_sig
            # even if some of its elements are None.
            for f in sig_write_fields:
                if len(getattr(self, f)) != self.n_sig:
                    raise ValueError(
                        "The length of field: " + f + " must match field n_sig."
                    )

            # Each file_name must correspond to only one fmt, (and only one byte offset if defined).
            datfmts = {}
            for ch in range(self.n_sig):
                if self.file_name[ch] not in datfmts:
                    datfmts[self.file_name[ch]] = self.fmt[ch]
                else:
                    if datfmts[self.file_name[ch]] != self.fmt[ch]:
                        raise ValueError(
                            "Each file_name (dat file) specified must have the same fmt"
                        )

            datoffsets = {}
            if self.byte_offset is not None:
                # At least one byte offset value exists
                for ch in range(self.n_sig):
                    if self.byte_offset[ch] is None:
                        continue
                    if self.file_name[ch] not in datoffsets:
                        datoffsets[self.file_name[ch]] = self.byte_offset[ch]
                    else:
                        if (
                            datoffsets[self.file_name[ch]]
                            != self.byte_offset[ch]
                        ):
                            raise ValueError(
                                "Each file_name (dat file) specified must have the same byte offset"
                            )

    def wr_header_file(self, rec_write_fields, sig_write_fields, write_dir):
        """
        Write a header file using the specified fields. Converts Record
        attributes into appropriate WFDB format strings.

        Parameters
        ----------
        rec_write_fields : list
            List of record specification fields to write.
        sig_write_fields : dict
            Dictionary of signal specification fields to write, values
            being equal to a list of channels to write for each field.
        write_dir : str
            The directory in which to write the header file.

        Returns
        -------
        N/A

        """
        # Create record specification line
        record_line = ""
        # Traverse the ordered dictionary
        for field in RECORD_SPECS.index:
            # If the field is being used, add it with its delimiter
            if field in rec_write_fields:
                string_field = str(getattr(self, field))

                # Certain fields need extra processing
                if field == "fs" and isinstance(self.fs, float):
                    if round(self.fs, 8) == float(int(self.fs)):
                        string_field = str(int(self.fs))
                elif field == "base_time" and "." in string_field:
                    string_field = string_field.rstrip("0")
                elif field == "base_date":
                    string_field = "/".join(
                        (string_field[8:], string_field[5:7], string_field[:4])
                    )

                record_line += (
                    RECORD_SPECS.loc[field, "delimiter"] + string_field
                )
                # The 'base_counter' field needs to be closed with ')'
                if field == "base_counter":
                    record_line += ")"

        header_lines = [record_line]

        # Create signal specification lines (if any) one channel at a time
        if self.n_sig > 0:
            signal_lines = self.n_sig * [""]
            for ch in range(self.n_sig):
                # Traverse the signal fields
                for field in SIGNAL_SPECS.index:
                    # If the field is being used, add each of its
                    # elements with the delimiter to the appropriate
                    # line
                    if (
                        field in sig_write_fields
                        and ch in sig_write_fields[field]
                    ):
                        signal_lines[ch] += SIGNAL_SPECS.loc[
                            field, "delimiter"
                        ] + str(getattr(self, field)[ch])
                    # The 'baseline' field needs to be closed with ')'
                    if field == "baseline":
                        signal_lines[ch] += ")"

            header_lines += signal_lines

        # Create comment lines (if any)
        if "comments" in rec_write_fields:
            comment_lines = ["# " + comment for comment in self.comments]
            header_lines += comment_lines

        util.lines_to_file(self.record_name + ".hea", write_dir, header_lines)


class MultiHeaderMixin(BaseHeaderMixin):
    """
    Mixin class with multi-segment header methods. Inherited by
    MultiRecord class.

    Attributes
    ----------
    N/A

    """

    def set_defaults(self):
        """
        Set defaults for fields needed to write the header if they have
        defaults. This is NOT called by rdheader. It is only called by the
        gateway wrsamp for convenience. It is also not called by wrheader since
        it is supposed to be an explicit function. Not responsible for
        initializing the attributes. That is done by the constructor.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        """
        for field in self.get_write_fields():
            self.set_default(field)

    def wrheader(self, write_dir=""):
        """
        Write a multi-segment WFDB header file. The signals or segments are
        not used. Before writing:
        - Get the fields used to write the header for this instance.
        - Check each required field.
        - Check that the fields are cohesive with one another.

        Parameters
        ----------
        write_dir : str, optional
            The output directory in which the header is written.

        Returns
        -------
        N/A

        Notes
        -----
        This function does NOT call `set_defaults`. Essential fields
        must be set beforehand.

        """
        # Get all the fields used to write the header
        write_fields = self.get_write_fields()

        # Check the validity of individual fields used to write the header
        for field in write_fields:
            self.check_field(field)

        # Check the cohesion of fields used to write the header
        self.check_field_cohesion()

        # Write the header file using the specified fields
        self.wr_header_file(write_fields, write_dir)

    def get_write_fields(self):
        """
        Get the list of fields used to write the multi-segment header.

        Parameters
        ----------
        N/A

        Returns
        -------
        write_fields : list
            All the default required fields, the user defined fields,
            and their dependencies.

        """
        # Record specification fields
        write_fields = self.get_write_subset("record")

        # Segment specification fields are all mandatory
        write_fields = write_fields + ["seg_name", "seg_len"]

        # Comments
        if self.comments != None:
            write_fields.append("comments")
        return write_fields

    def set_default(self, field):
        """
        Set a field to its default value if there is a default.

        Parameters
        ----------
        field : str
            The desired attribute of the object.

        Returns
        -------
        N/A

        """
        # Record specification fields
        if field in RECORD_SPECS:
            # Return if no default to set, or if the field is already present.
            if (
                RECORD_SPECS[field].write_def is None
                or getattr(self, field) is not None
            ):
                return
            setattr(self, field, RECORD_SPECS[field].write_def)

    def check_field_cohesion(self):
        """
        Check the cohesion of fields used to write the header.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        """
        # The length of seg_name and seg_len must match n_seg
        for f in ["seg_name", "seg_len"]:
            if len(getattr(self, f)) != self.n_seg:
                raise ValueError(
                    "The length of field: " + f + " does not match field n_seg."
                )

        # Check the sum of the 'seg_len' fields against 'sig_len'
        if np.sum(self.seg_len) != self.sig_len:
            raise ValueError(
                "The sum of the 'seg_len' fields do not match the 'sig_len' field"
            )

    def wr_header_file(self, write_fields, write_dir):
        """
        Write a header file using the specified fields.

        Parameters
        ----------
        write_fields : list
            All the default required fields, the user defined fields,
            and their dependencies.
        write_dir : str
            The output directory in which the header is written.

        Returns
        -------
        N/A

        """
        # Create record specification line
        record_line = ""
        # Traverse the ordered dictionary
        for field in RECORD_SPECS.index:
            # If the field is being used, add it with its delimiter
            if field in write_fields:
                record_line += RECORD_SPECS.loc[field, "delimiter"] + str(
                    getattr(self, field)
                )

        header_lines = [record_line]

        # Create segment specification lines
        segment_lines = self.n_seg * [""]
        # For both fields, add each of its elements with the delimiter
        # to the appropriate line
        for field in SEGMENT_SPECS.index:
            for seg_num in range(self.n_seg):
                segment_lines[seg_num] += SEGMENT_SPECS.loc[
                    field, "delimiter"
                ] + str(getattr(self, field)[seg_num])

        header_lines = header_lines + segment_lines

        # Create comment lines (if any)
        if "comments" in write_fields:
            comment_lines = ["# " + comment for comment in self.comments]
            header_lines += comment_lines

        util.lines_to_file(self.record_name + ".hea", header_lines, write_dir)

    def get_sig_segments(self, sig_name=None):
        """
        Get a list of the segment numbers that contain a particular signal
        (or a dictionary of segment numbers for a list of signals).
        Only works if information about the segments has been read in.

        Parameters
        ----------
        sig_name : str, list
            The name of the signals to be segmented.

        Returns
        -------
        sig_dict : dict
            Segments for each desired signal.
        sig_segs : list
            Segments for the desired signal.

        """
        if self.segments is None:
            raise Exception(
                "The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rsegment_fieldsments=True"
            )

        # Default value = all signal names.
        if sig_name is None:
            sig_name = self.get_sig_name()

        if isinstance(sig_name, list):
            sig_dict = {}
            for sig in sig_name:
                sig_dict[sig] = self.get_sig_segments(sig)
            return sig_dict
        elif isinstance(sig_name, str):
            sig_segs = []
            for i in range(self.n_seg):
                if (
                    self.seg_name[i] != "~"
                    and sig_name in self.segments[i].sig_name
                ):
                    sig_segs.append(i)
            return sig_segs
        else:
            raise TypeError("sig_name must be a string or a list of strings")

    def get_sig_name(self):
        """
        Get the signal names for the entire record.

        Parameters
        ----------
        N/A

        Returns
        -------
        sig_name : str, list
            The name of the signals to be segmented.

        """
        if self.segments is None:
            raise Exception(
                "The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rd_segments=True"
            )

        if self.layout == "fixed":
            for i in range(self.n_seg):
                if self.seg_name[i] != "~":
                    sig_name = self.segments[i].sig_name
                    break
        else:
            sig_name = self.segments[0].sig_name

        return sig_name

    def contained_ranges(self, sig_name: str) -> List[Tuple[int, int]]:
        """
        Given a signal name, return the sample ranges that contain signal values,
        relative to the start of the full record. Does not account for NaNs/missing
        values.

        This function is mainly useful for variable layout records, but can also be
        used for fixed-layout records. Only works if the headers from the individual
        segment records have already been read in.

        Parameters
        ----------
        sig_name : str
            The name of the signal to query.

        Returns
        -------
        ranges : List[Tuple[int, int]]
            Tuple pairs which specify thee sample ranges in which the signal is contained.
            The second value of each tuple pair will be one beyond the signal index.
            eg. A length 1000 signal would generate a tuple of: (0, 1000), allowing
            selection using signal[0:1000].

        """
        if self.segments is None:
            raise Exception(
                "The MultiRecord's segments must be read in before this method is called. ie. Call rdheader() with rd_segments=True"
            )
        ranges = []
        seg_start = 0

        range_start = None

        # TODO: Add shortcut for fixed-layout records

        # Cannot process segments only because missing segments are None
        # and do not contain length information.
        for seg_num in range(self.n_seg):
            seg_len = self.seg_len[seg_num]
            segment = self.segments[seg_num]

            if seg_len == 0:
                continue

            # Open signal range
            if (
                range_start is None
                and segment is not None
                and sig_name in segment.sig_name
            ):
                range_start = seg_start
            # Close signal range
            elif range_start is not None and (
                segment is None or sig_name not in segment.sig_name
            ):
                ranges.append((range_start, seg_start))
                range_start = None

            seg_start += seg_len

        # Account for final segment
        if range_start is not None:
            ranges.append((range_start, seg_start))

        return ranges

    def contained_combined_ranges(
        self,
        sig_names: Collection[str],
    ) -> List[Tuple[int, int]]:
        """
        Given a collection of signal name, return the sample ranges that
        contain all of the specified signals, relative to the start of the
        full record. Does not account for NaNs/missing values.

        This function is mainly useful for variable layout records, but can also be
        used for fixed-layout records. Only works if the headers from the individual
        segment records have already been read in.

        Parameters
        ----------
        sig_names : List[str]
            The names of the signals to query.

        Returns
        -------
        ranges : List[Tuple[int, int]]
            Tuple pairs which specify thee sample ranges in which the signal is contained.
            The second value of each tuple pair will be one beyond the signal index.
            eg. A length 1000 signal would generate a tuple of: (0, 1000), allowing
            selection using signal[0:1000].

        """
        # TODO: Add shortcut for fixed-layout records

        if len(sig_names) == 0:
            return []

        combined_ranges = self.contained_ranges(sig_names[0])

        if len(sig_names) > 1:
            for name in sig_names[1:]:
                combined_ranges = util.overlapping_ranges(
                    combined_ranges, self.contained_ranges(name)
                )

        return combined_ranges


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


def _parse_record_line(record_line: str) -> dict:
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
    record_fields = {}

    # Read string fields from record line
    match = rx_record.match(record_line)
    if match is None:
        raise HeaderSyntaxError("invalid syntax in record line")
    (
        record_fields["record_name"],
        record_fields["n_seg"],
        record_fields["n_sig"],
        record_fields["fs"],
        record_fields["counter_freq"],
        record_fields["base_counter"],
        record_fields["sig_len"],
        record_fields["base_time"],
        record_fields["base_date"],
    ) = match.groups()

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


def _parse_signal_lines(signal_lines):
    """
    Extract fields from a list of signal line strings into a dictionary.

    Parameters
    ----------
    signal_lines : list
        The name of the signal line that will be used to extact fields.

    Returns
    -------
    signal_fields : dict
        The fields for the given signal line.

    """
    n_sig = len(signal_lines)
    # Dictionary for signal fields
    signal_fields = {}

    # Each dictionary field is a list
    for field in SIGNAL_SPECS.index:
        signal_fields[field] = n_sig * [None]

    # Read string fields from signal line
    for ch in range(n_sig):
        match = rx_signal.match(signal_lines[ch])
        if match is None:
            raise HeaderSyntaxError("invalid syntax in signal line")
        (
            signal_fields["file_name"][ch],
            signal_fields["fmt"][ch],
            signal_fields["samps_per_frame"][ch],
            signal_fields["skew"][ch],
            signal_fields["byte_offset"][ch],
            signal_fields["adc_gain"][ch],
            signal_fields["baseline"][ch],
            signal_fields["units"][ch],
            signal_fields["adc_res"][ch],
            signal_fields["adc_zero"][ch],
            signal_fields["init_value"][ch],
            signal_fields["checksum"][ch],
            signal_fields["block_size"][ch],
            signal_fields["sig_name"][ch],
        ) = match.groups()

        for field in SIGNAL_SPECS.index:
            # Replace empty strings with their read defaults (which are mostly None)
            # Note: Never set a field to None. [None]* n_sig is accurate, indicating
            # that different channels can be present or missing.
            if signal_fields[field][ch] == "":
                signal_fields[field][ch] = SIGNAL_SPECS.loc[
                    field, "read_default"
                ]

                # Special case: missing baseline defaults to ADCzero if present
                if field == "baseline" and signal_fields["adc_zero"][ch] != "":
                    signal_fields["baseline"][ch] = int(
                        signal_fields["adc_zero"][ch]
                    )
            # Typecast non-empty strings for numerical fields
            else:
                if SIGNAL_SPECS.loc[field, "allowed_types"] is int_types:
                    signal_fields[field][ch] = int(signal_fields[field][ch])
                elif SIGNAL_SPECS.loc[field, "allowed_types"] is float_types:
                    signal_fields[field][ch] = float(signal_fields[field][ch])
                    # Special case: adc_gain of 0 means 200
                    if (
                        field == "adc_gain"
                        and signal_fields["adc_gain"][ch] == 0
                    ):
                        signal_fields["adc_gain"][ch] = 200.0

    return signal_fields


def _read_segment_lines(segment_lines):
    """
    Extract fields from segment line strings into a dictionary.

    Parameters
    ----------
    segment_line : list
        The name of the segment line that will be used to extact fields.

    Returns
    -------
    segment_fields : dict
        The fields for the given segment line.

    """
    # Dictionary for segment fields
    segment_fields = {}

    # Each dictionary field is a list
    for field in SEGMENT_SPECS.index:
        segment_fields[field] = [None] * len(segment_lines)

    # Read string fields from signal line
    for i in range(len(segment_lines)):
        match = rx_segment.match(segment_lines[i])
        if match is None:
            raise HeaderSyntaxError("invalid syntax in segment line")
        (
            segment_fields["seg_name"][i],
            segment_fields["seg_len"][i],
        ) = match.groups()

        # Typecast strings for numerical field
        if field == "seg_len":
            segment_fields["seg_len"][i] = int(segment_fields["seg_len"][i])

    return segment_fields
