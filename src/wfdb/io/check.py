"""Check local WFDB records before sharing them."""

from dataclasses import dataclass
from pathlib import Path

from wfdb.io.archive import WFDBArchive
from wfdb.io.record import MultiRecord, _parse_header_content, rdheader, rdrecord


@dataclass(frozen=True)
class CheckIssue:
    """A problem found while checking a WFDB record.

    Attributes
    ----------
    path : pathlib.Path
        The header or signal file associated with the problem.
    code : str
        A stable identifier for the kind of problem.
    message : str
        A human-readable description of the problem.
    """

    path: Path
    code: str
    message: str


def check_record(record_name):
    """Check one local WFDB record and return its problems.

    The header is parsed with the normal WFDB reader. The check also verifies
    that the header names the expected record, that referenced signal and
    segment files exist, and that the first signal frame can be read. It does
    not scan all samples or annotations.

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
    >>> issues = wfdb.check_record("sample-data/100")
    >>> issues
    []
    """
    path = Path(record_name)
    if path.suffix == ".hea":
        path = path.with_suffix("")
    issues = []
    if path.suffix == ".wfdb":
        _check_archive(path, issues)
    else:
        _check_local_record(path, issues, set(), set())
    return issues


def check_dataset(directory):
    """Check every WFDB header and archive below a local directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Root directory of the dataset to check.

    Returns
    -------
    list of CheckIssue
        Problems found across all records.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    issues = []
    checked = set()
    for header in sorted(directory.rglob("*.hea")):
        _check_local_record(header.with_suffix(""), issues, checked, set())
    for archive in sorted(directory.rglob("*.wfdb")):
        _check_archive(archive, issues)
    return issues


def _check_local_record(path, issues, checked, active):
    path = path.resolve()
    if path in active:
        issues.append(
            CheckIssue(path.with_suffix(".hea"), "cycle", "Segment reference cycle")
        )
        return
    if path in checked:
        return
    checked.add(path)
    active.add(path)
    header_path = path.with_suffix(".hea")
    try:
        header = rdheader(str(path))
    except Exception as exc:
        issues.append(CheckIssue(header_path, "header", str(exc)))
        active.remove(path)
        return

    if header.record_name != path.name:
        issues.append(
            CheckIssue(
                header_path,
                "record_name",
                f"Header names {header.record_name!r}; expected {path.name!r}",
            )
        )

    if isinstance(header, MultiRecord):
        if len(header.seg_name) != header.n_seg:
            issues.append(
                CheckIssue(
                    header_path, "segments", "Segment count does not match header"
                )
            )
        if sum(header.seg_len) != header.sig_len:
            issues.append(
                CheckIssue(
                    header_path, "length", "Segment lengths do not sum to record length"
                )
            )
        for segment in header.seg_name:
            if segment != "~":
                _check_local_record(path.parent / segment, issues, checked, active)
        active.remove(path)
        return

    if len(header.file_name or []) != header.n_sig:
        issues.append(
            CheckIssue(header_path, "signals", "Signal count does not match header")
        )
        active.remove(path)
        return

    missing = False
    for name in sorted(set(header.file_name or [])):
        if name == "~":
            continue
        signal_path = path.parent / name
        if not signal_path.is_file():
            issues.append(
                CheckIssue(
                    signal_path, "missing_signal", "Referenced signal file is missing"
                )
            )
            missing = True

    if missing or header.n_sig == 0 or all(name == "~" for name in header.file_name):
        active.remove(path)
        return
    if header.sig_len == 0:
        active.remove(path)
        return
    try:
        rdrecord(str(path), sampto=1, physical=False)
    except Exception as exc:
        issues.append(CheckIssue(header_path, "signal_read", str(exc)))
    active.remove(path)


def _check_archive(path, issues):
    issue_count = len(issues)
    try:
        with WFDBArchive(path) as archive:
            header_name = archive.header_file_name()
            header = _check_archive_record(
                archive, Path(header_name).stem, issues, set(), set(), main=True
            )
            if (
                header is not None
                and len(issues) == issue_count
                and header.n_sig
                and header.sig_len != 0
            ):
                rdrecord(str(path), sampto=1, physical=False)
    except Exception as exc:
        issues.append(CheckIssue(path, "archive", str(exc)))


def _check_archive_record(archive, name, issues, checked, active, main=False):
    if name in active:
        issues.append(
            CheckIssue(
                Path(archive.path), "cycle", f"{name}.hea: segment reference cycle"
            )
        )
        return None
    if name in checked:
        return None
    checked.add(name)
    active.add(name)
    try:
        content = archive.read_header(None if main else name)
        header = _parse_header_content(content)
    except Exception as exc:
        issues.append(CheckIssue(Path(archive.path), "header", f"{name}.hea: {exc}"))
        active.remove(name)
        return None

    if header.record_name != name:
        issues.append(
            CheckIssue(
                Path(archive.path),
                "record_name",
                f"{name}.hea names {header.record_name!r}; expected {name!r}",
            )
        )

    if isinstance(header, MultiRecord):
        if len(header.seg_name) != header.n_seg:
            issues.append(
                CheckIssue(
                    Path(archive.path), "segments", f"{name}.hea: wrong segment count"
                )
            )
        if sum(header.seg_len) != header.sig_len:
            issues.append(
                CheckIssue(
                    Path(archive.path), "length", f"{name}.hea: wrong segment length"
                )
            )
        for segment in header.seg_name:
            if segment != "~":
                _check_archive_record(archive, segment, issues, checked, active)
    else:
        if len(header.file_name or []) != header.n_sig:
            issues.append(
                CheckIssue(
                    Path(archive.path), "signals", f"{name}.hea: wrong signal count"
                )
            )
        for signal_name in sorted(set(header.file_name or [])):
            if signal_name != "~" and not archive.exists(signal_name):
                issues.append(
                    CheckIssue(
                        Path(archive.path), "missing_signal", f"Missing {signal_name}"
                    )
                )
    active.remove(name)
    return header
