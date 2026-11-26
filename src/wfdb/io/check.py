"""Check local WFDB records before sharing them."""

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from wfdb.io import _signal
from wfdb.io.archive import WFDBArchive
from wfdb.io.header import HeaderSyntaxError
from wfdb.io.record import (
    MultiRecord,
    _parse_header_content,
    _rdrecord,
    rdheader,
    rdrecord,
)


@dataclass(frozen=True)
class CheckIssue:
    """A problem found while checking a WFDB record.

    Attributes
    ----------
    path : pathlib.Path
        The header or signal file associated with the problem. For an archive,
        this is the path of the ``.wfdb`` file.
    code : str
        A stable identifier for the kind of problem.
    message : str
        A human-readable description of the problem.
    """

    path: Path
    code: str
    message: str


class _Source:
    """Read a record from a directory or an open .wfdb archive."""

    def __init__(self, path, archive=None, main_name=None):
        self.path = path.resolve()
        self.archive = archive
        self.main_name = main_name

    def key(self, name):
        return (self.path, name)

    def header_path(self, name):
        return self.path if self.archive else self.path / f"{name}.hea"

    def signal_path(self, name):
        return self.path if self.archive else self.path / name

    def header_exists(self, name):
        return self.exists(f"{name}.hea")

    def exists(self, name):
        if self.archive:
            return self.archive.exists(name)
        return (self.path / name).is_file()

    def is_symlink(self, name):
        if self.archive:
            return self.archive.is_symlink(name)
        return (self.path / name).is_symlink()

    def file_size(self, name):
        if self.archive:
            return self.archive.file_size(name)
        return (self.path / name).stat().st_size

    def read_header(self, name):
        if self.archive:
            archive_name = None if name == self.main_name else name
            return _parse_header_content(self.archive.read_header(archive_name))
        return rdheader(str(self.path / name))

    def header_text(self, name):
        if self.archive:
            archive_name = None if name == self.main_name else name
            return self.archive.read_header(archive_name)
        return (self.path / f"{name}.hea").read_text()

    def read_first_frame(self, name):
        if not self.archive:
            rdrecord(str(self.path / name), sampto=1, physical=False)
            return
        # Read a segment using the already-open archive. Passing the archive
        # path to rdrecord would only read the master's first segment.
        _rdrecord(
            record_name=name,
            sampfrom=0,
            sampto=1,
            channels=None,
            physical=False,
            pn_dir=None,
            m2s=True,
            smooth_frames=True,
            ignore_skew=False,
            return_res=64,
            force_channels=True,
            channel_names=None,
            warn_empty=False,
            wfdb_archive=self.archive,
        )

    def issue(self, name, code, message, file_name=None):
        if self.archive:
            target = file_name or f"{name}.hea"
            return CheckIssue(self.path, code, f"{target}: {message}")
        path = self.signal_path(file_name) if file_name else self.header_path(name)
        return CheckIssue(path, code, message)


def check_record(record_name):
    """Check one local WFDB record and return its problems.

    The header is parsed with the normal WFDB reader. The check verifies
    record and segment names, fixed-layout signals, referenced files, the
    minimum size of uncompressed signal files, and the first signal frame.
    It does not scan all samples or annotations. Both ordinary records and
    ``.wfdb`` archives use the same checks.

    Parameters
    ----------
    record_name : str or pathlib.Path
        Local record path, with no extension or with ``.hea`` or ``.wfdb``.

    Returns
    -------
    list of CheckIssue
        Problems found. An empty list means that these checks passed.

    Examples
    --------
    >>> import wfdb
    >>> wfdb.check_record("sample-data/100")
    []
    """
    path = Path(record_name)
    if path.suffix == ".hea":
        path = path.with_suffix("")
    if path.suffix not in ("", ".wfdb"):
        raise ValueError("Expected a record name, .hea file, or .wfdb archive")

    issues = []
    if path.suffix == ".wfdb":
        _check_archive(path, issues)
    else:
        source = _Source(path.parent)
        _check_record(source, path.name, issues, {}, set())
    return issues


def check_dataset(directory, ignore=None):
    """Check every WFDB header and archive below a local directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Root directory of the dataset to check.
    ignore : callable, optional
        A function that chooses which records to check. It is given the path
        of each folder and file below `directory`, relative to it, and returns
        True to skip it. Headers and archives that are skipped are not checked,
        and the contents of a skipped folder are not read. Files that a checked
        record refers to are still checked, even if `ignore` would skip them:
        its signal files, and its segments if it is a multi-segment record.

    Returns
    -------
    list of CheckIssue
        Problems found across all records.

    Examples
    --------
    Skip version control folders:

    >>> import wfdb
    >>> issues = wfdb.check_dataset(
    ...     "sample-data", ignore=lambda path: ".git" in path.parts
    ... )

    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    headers, archives = _dataset_files(directory, ignore)
    issues = []
    checked = {}
    for header in headers:
        source = _Source(header.parent)
        _check_record(source, header.stem, issues, checked, set())
    for archive in archives:
        _check_archive(archive, issues)
    return issues


def _dataset_files(directory, ignore):
    """List the headers and archives below a directory, in sorted order."""
    headers = []
    archives = []
    for folder, subfolders, files in os.walk(directory):
        folder = Path(folder)
        relative = folder.relative_to(directory)
        if ignore is not None:
            subfolders[:] = [name for name in subfolders if not ignore(relative / name)]
        for name in files:
            if ignore is not None and ignore(relative / name):
                continue
            if name.endswith(".hea"):
                headers.append(folder / name)
            elif name.endswith(".wfdb"):
                archives.append(folder / name)
    return sorted(headers), sorted(archives)


def _check_archive(path, issues):
    if path.is_symlink():
        issues.append(CheckIssue(path, "symlink", "Archive is a symbolic link"))
        return
    try:
        with WFDBArchive(path) as archive:
            for entry in archive.namelist():
                if archive.is_symlink(entry):
                    issues.append(
                        CheckIssue(path, "symlink", f"{entry}: symbolic link")
                    )
                    return
            name = Path(archive.header_file_name()).stem
            source = _Source(path, archive, name)
            _check_record(source, name, issues, {}, set())
    except Exception as exc:
        issues.append(CheckIssue(path, "archive", str(exc)))


def _check_record(
    source, name, issues, checked, active, expected_len=None, is_segment=False
):
    key = source.key(name)
    if key in active:
        issues.append(source.issue(name, "cycle", "Segment reference cycle"))
        return None
    if key in checked:
        header = checked[key]
        _check_segment_length(source, name, header, expected_len, issues)
        if is_segment and isinstance(header, MultiRecord):
            issues.append(
                source.issue(
                    name, "nested_segment", "A segment cannot be multi-segment"
                )
            )
        return header

    if not source.header_exists(name):
        code = "missing_segment" if is_segment else "header"
        issues.append(source.issue(name, code, "Header file is missing"))
        checked[key] = None
        return None
    if source.is_symlink(f"{name}.hea"):
        issues.append(source.issue(name, "symlink", "Header is a symbolic link"))
        checked[key] = None
        return None

    active.add(key)
    try:
        try:
            header = source.read_header(name)
        except Exception as exc:
            message = _describe_header_error(source, name, exc)
            issues.append(source.issue(name, "header", message))
            checked[key] = None
            return None
        checked[key] = header

        if header.record_name != name:
            issues.append(
                source.issue(
                    name,
                    "record_name",
                    f"Header names {header.record_name!r}; expected {name!r}",
                )
            )
        _check_segment_length(source, name, header, expected_len, issues)

        if isinstance(header, MultiRecord):
            if is_segment:
                issues.append(
                    source.issue(
                        name, "nested_segment", "A segment cannot be multi-segment"
                    )
                )
            else:
                _check_segments(source, name, header, issues, checked, active)
        else:
            _check_signals(source, name, header, issues)
        return header
    finally:
        active.remove(key)


def _describe_header_error(source, name, error):
    """Give line and field context for common malformed header references."""
    if not isinstance(error, HeaderSyntaxError):
        return str(error)
    try:
        lines = [
            (number, line.strip())
            for number, line in enumerate(source.header_text(name).splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except (OSError, UnicodeError, ValueError):
        return str(error)
    if not lines:
        return str(error)

    signal_lines = [
        (line_number, value.split()[0])
        for line_number, value in lines[1:]
        if len(value.split()) >= 3 and "/" in value.split()[0]
    ]
    number, line = lines[0]
    token = line.split()[0]
    parts = token.split("/")
    expected_token = None
    if len(parts) >= 2 and parts[-1] == name:
        expected_token = name
    elif len(parts) >= 3 and parts[-2] == name and parts[-1].isdigit():
        expected_token = f"{name}/{parts[-1]}"
    elif "record line" in str(error) and len(parts) >= 2 and not parts[-1].isdigit():
        expected_token = name
    if expected_token is not None and token != expected_token:
        message = (
            f"line {number}: record name {token!r} contains a directory "
            f"prefix; expected {expected_token!r} to match the header filename"
        )
        if signal_lines:
            message += "; " + _describe_signal_paths(source, signal_lines)
        return message
    if "record line" in str(error):
        return f"line {number}: {error}; record line is {line!r}"
    if "signal line" in str(error) and signal_lines:
        return _describe_signal_paths(source, signal_lines)
    return str(error)


def _describe_signal_paths(source, signal_lines):
    """Describe signal filenames that retain an obsolete directory prefix."""
    first_line = signal_lines[0][0]
    example_file = next(
        (
            file_name
            for _, file_name in signal_lines
            if source.exists(Path(file_name).name)
        ),
        signal_lines[0][1],
    )
    basename = Path(example_file).name
    location = (
        f"line {first_line}"
        if len(signal_lines) == 1
        else f"lines {first_line}-{signal_lines[-1][0]}"
    )
    message = (
        f"{location}: {len(signal_lines)} signal file "
        f"{'reference contains' if len(signal_lines) == 1 else 'references contain'} a "
        f"directory prefix (e.g. {example_file!r}); "
    )
    if source.exists(basename):
        return message + f"{basename!r} is beside the header; remove the prefix"
    return message + f"use {basename!r} if the file is beside the header"


def _check_segment_length(source, name, header, expected_len, issues):
    if (
        header is not None
        and expected_len is not None
        and header.sig_len != expected_len
    ):
        issues.append(
            source.issue(
                name,
                "segment_length",
                f"Segment has {header.sig_len} samples; parent declares {expected_len}",
            )
        )


def _check_segments(source, name, header, issues, checked, active):
    if len(header.seg_name) != header.n_seg:
        issues.append(
            source.issue(name, "segments", "Segment count does not match header")
        )
    if sum(header.seg_len) != header.sig_len:
        issues.append(
            source.issue(name, "length", "Segment lengths do not sum to record length")
        )

    reference_signals = None
    for segment, segment_len in zip(header.seg_name, header.seg_len):
        if segment == "~":
            continue
        child = _check_record(
            source, segment, issues, checked, active, segment_len, is_segment=True
        )
        if child is None or isinstance(child, MultiRecord) or header.layout != "fixed":
            continue
        signals = (child.n_sig, tuple(child.sig_name or []))
        if reference_signals is None:
            reference_signals = signals
        elif signals != reference_signals:
            issues.append(
                source.issue(
                    segment,
                    "fixed_layout",
                    "Signal count or names differ from the first segment",
                )
            )


def _check_signals(source, name, header, issues):
    if len(header.file_name or []) != header.n_sig:
        issues.append(
            source.issue(name, "signals", "Signal count does not match header")
        )
        return

    channels = defaultdict(list)
    for index, file_name in enumerate(header.file_name or []):
        if file_name != "~":
            channels[file_name].append(index)

    readable = bool(channels)
    for file_name, indices in sorted(channels.items()):
        if not source.exists(file_name):
            issues.append(
                source.issue(
                    name,
                    "missing_signal",
                    "Referenced signal file is missing",
                    file_name,
                )
            )
            readable = False
            continue
        if source.is_symlink(file_name):
            issues.append(
                source.issue(
                    name, "symlink", "Signal file is a symbolic link", file_name
                )
            )
            readable = False
            continue
        _check_signal_size(source, name, file_name, header, indices, issues)

    if readable and header.sig_len != 0:
        try:
            source.read_first_frame(name)
        except Exception as exc:
            issues.append(source.issue(name, "signal_read", str(exc)))


def _check_signal_size(source, name, file_name, header, indices, issues):
    if header.sig_len is None:
        return
    formats = {header.fmt[index] for index in indices}
    offsets = {(header.byte_offset[index] or 0) for index in indices}
    if len(formats) != 1 or len(offsets) != 1:
        issues.append(
            source.issue(
                name,
                "signal_layout",
                "Channels sharing a file have different formats or byte offsets",
                file_name,
            )
        )
        return
    fmt = formats.pop()
    if fmt in _signal.COMPRESSED_FMTS:
        return  # FLAC size does not give a useful lower bound on sample count.
    try:
        samples_per_frame = sum(header.samps_per_frame[index] or 1 for index in indices)
        required = offsets.pop() + _signal._required_byte_num(
            "read", fmt, header.sig_len * samples_per_frame
        )
        actual = source.file_size(file_name)
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        issues.append(source.issue(name, "signal_layout", str(exc), file_name))
        return
    if actual < required:
        issues.append(
            source.issue(
                name,
                "short_signal",
                f"Signal file has {actual} bytes; at least {required} are required",
                file_name,
            )
        )
