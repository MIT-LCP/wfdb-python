"""Tests for local WFDB submission checks."""

import struct
import sys
import zipfile
from pathlib import Path

import pytest

import wfdb
from wfdb.cli import main


def _write_record(directory, name="good", signal_name=None):
    directory.mkdir(parents=True, exist_ok=True)
    signal_name = signal_name or f"{name}.dat"
    (directory / f"{name}.hea").write_text(
        f"{name} 1 250 2\n{signal_name} 16 200 11 0 0 0 0 ECG\n"
    )
    (directory / f"{name}.dat").write_bytes(struct.pack("<hh", 0, 1))
    return directory / name


def _archive_records(directory, name, records):
    archive = directory / f"{name}.wfdb"
    with zipfile.ZipFile(archive, "w") as output:
        for record in records:
            for extension in ("hea", "dat"):
                file_path = record.with_suffix(f".{extension}")
                if file_path.exists():
                    output.write(file_path, file_path.name)
    return archive


def test_check_record_accepts_valid_record(tmp_path):
    record = _write_record(tmp_path)
    assert wfdb.check_record(record) == []
    assert wfdb.check_record(record.with_suffix(".hea")) == []


def test_check_record_rejects_broken_header_from_issue_528(tmp_path):
    record = _write_record(tmp_path)
    (record.with_suffix(".hea")).write_text(
        "ge_median_beats_wfdb/good 1 250 2\n"
        "ge_median_beats_wfdb/good.dat 16 200 11 0 0 0 0 ECG\n"
    )
    issues = wfdb.check_record(record)
    assert [issue.code for issue in issues] == ["header"]
    assert "line 1" in issues[0].message
    assert "ge_median_beats_wfdb/good" in issues[0].message
    assert "expected 'good'" in issues[0].message
    assert "line 2" in issues[0].message
    assert "'good.dat' is beside the header" in issues[0].message


def test_check_explains_flattened_dataset_header(tmp_path, capsys):
    record = tmp_path / "00001_medians"
    record.with_suffix(".hea").write_text(
        "ge_median_beats_wfdb/00001_medians 2 500 600\n"
        "ge_median_beats_wfdb/00001_medians.dat 32 200 32 0 0 0 0 I\n"
        "ge_median_beats_wfdb/00001_medians.dat 32 200 32 0 0 0 0 II\n"
    )
    record.with_suffix(".dat").write_bytes(b"\0" * 4800)

    assert main(["check", str(record)]) == 1
    output = capsys.readouterr().out
    assert "line 1: record name" in output
    assert "expected '00001_medians'" in output
    assert "lines 2-3: 2 signal file reference(s)" in output
    assert "'00001_medians.dat' is beside the header" in output

    header = record.with_suffix(".hea")
    header.write_text(
        header.read_text().replace(
            "ge_median_beats_wfdb/00001_medians 2", "00001_medians 2"
        )
    )
    issues = wfdb.check_record(record)
    assert [issue.code for issue in issues] == ["header"]
    assert "lines 2-3: 2 signal file reference(s)" in issues[0].message
    assert "'00001_medians.dat' is beside the header" in issues[0].message


def test_check_record_rejects_wrong_name_and_missing_signal(tmp_path):
    record = _write_record(tmp_path, signal_name="elsewhere.dat")
    header = record.with_suffix(".hea")
    header.write_text(header.read_text().replace("good 1", "wrong 1"))
    issues = wfdb.check_record(record)
    assert {issue.code for issue in issues} == {"record_name", "missing_signal"}


@pytest.mark.parametrize("in_archive", [False, True])
def test_check_record_rejects_truncated_signal(tmp_path, in_archive):
    record = _write_record(tmp_path)
    record.with_suffix(".hea").write_text(
        "good 1 250 1000\ngood.dat 16 200 11 0 0 0 0 ECG\n"
    )
    target = _archive_records(tmp_path, "good", [record]) if in_archive else record
    assert "short_signal" in [issue.code for issue in wfdb.check_record(target)]


@pytest.mark.parametrize("in_archive", [False, True])
def test_check_record_rejects_wrong_segment_length(tmp_path, in_archive):
    segment = _write_record(tmp_path, "segment")
    parent = tmp_path / "parent"
    parent.with_suffix(".hea").write_text("parent/1 1 250 4\nsegment 4\n")
    target = (
        _archive_records(tmp_path, "parent", [parent, segment])
        if in_archive
        else parent
    )
    assert "segment_length" in [issue.code for issue in wfdb.check_record(target)]


@pytest.mark.parametrize("in_archive", [False, True])
def test_check_record_rejects_inconsistent_fixed_layout(tmp_path, in_archive):
    first = _write_record(tmp_path, "first")
    second = _write_record(tmp_path, "second")
    second.with_suffix(".hea").write_text(
        second.with_suffix(".hea").read_text().replace("ECG", "ABP")
    )
    parent = tmp_path / "parent"
    parent.with_suffix(".hea").write_text("parent/2 1 250 4\nfirst 2\nsecond 2\n")
    target = (
        _archive_records(tmp_path, "parent", [parent, first, second])
        if in_archive
        else parent
    )
    assert "fixed_layout" in [issue.code for issue in wfdb.check_record(target)]


@pytest.mark.parametrize("in_archive", [False, True])
def test_check_record_reports_missing_segment(tmp_path, in_archive):
    parent = tmp_path / "parent"
    parent.with_suffix(".hea").write_text("parent/1 1 250 2\nabsent 2\n")
    target = _archive_records(tmp_path, "parent", [parent]) if in_archive else parent
    assert [issue.code for issue in wfdb.check_record(target)] == ["missing_segment"]


def test_check_record_rejects_nested_multisegment(tmp_path):
    (tmp_path / "parent.hea").write_text("parent/1 1 250 2\nchild 2\n")
    (tmp_path / "child.hea").write_text("child/1 1 250 2\nleaf 2\n")
    assert "nested_segment" in [
        issue.code for issue in wfdb.check_record(tmp_path / "parent")
    ]


def test_check_dataset_rejects_cached_nested_multisegment(tmp_path):
    (tmp_path / "zparent.hea").write_text("zparent/1 1 250 2\nachild 2\n")
    (tmp_path / "achild.hea").write_text("achild/1 1 250 2\nleaf 2\n")
    assert "nested_segment" in [issue.code for issue in wfdb.check_dataset(tmp_path)]


def test_check_dataset_checks_segments_once(tmp_path):
    _write_record(tmp_path, "segment", signal_name="missing.dat")
    (tmp_path / "parent.hea").write_text("parent/1 1 250 2\nsegment 2\n")
    issues = wfdb.check_dataset(tmp_path)
    assert [issue.code for issue in issues] == ["missing_signal"]


def test_check_record_reports_segment_cycle(tmp_path):
    (tmp_path / "parent.hea").write_text("parent/1 1 250 2\nparent 2\n")
    assert [issue.code for issue in wfdb.check_record(tmp_path / "parent")] == ["cycle"]


def test_check_archive(tmp_path):
    record = _write_record(tmp_path)
    archive = tmp_path / "good.wfdb"
    with zipfile.ZipFile(archive, "w") as output:
        output.write(record.with_suffix(".hea"), "good.hea")
        output.write(record.with_suffix(".dat"), "good.dat")
    assert wfdb.check_record(archive) == []


def test_check_archive_finds_missing_segment_signal(tmp_path):
    segment = _write_record(tmp_path, "segment", signal_name="missing.dat")
    parent_header = tmp_path / "parent.hea"
    parent_header.write_text("parent/1 1 250 2\nsegment 2\n")
    archive = tmp_path / "parent.wfdb"
    with zipfile.ZipFile(archive, "w") as output:
        output.write(parent_header, "parent.hea")
        output.write(segment.with_suffix(".hea"), "segment.hea")
    assert [issue.code for issue in wfdb.check_record(archive)] == ["missing_signal"]


def test_check_archive_checks_later_segment_data(tmp_path):
    first = _write_record(tmp_path, "first")
    second = _write_record(tmp_path, "second")
    second.with_suffix(".dat").write_bytes(b"")
    parent = tmp_path / "parent"
    parent.with_suffix(".hea").write_text("parent/2 1 250 4\nfirst 2\nsecond 2\n")
    archive = _archive_records(tmp_path, "parent", [parent, first, second])
    local_codes = {issue.code for issue in wfdb.check_record(parent)}
    archive_codes = {issue.code for issue in wfdb.check_record(archive)}
    assert "short_signal" in local_codes
    assert "short_signal" in archive_codes


@pytest.mark.parametrize("in_archive", [False, True])
def test_check_record_reads_later_segment_first_frame(tmp_path, in_archive):
    first = _write_record(tmp_path, "first")
    second = _write_record(tmp_path, "second")
    second.with_suffix(".hea").write_text(
        second.with_suffix(".hea").read_text().replace(".dat 16", ".dat 516")
    )
    second.with_suffix(".dat").write_bytes(b"bad!")
    parent = tmp_path / "parent"
    parent.with_suffix(".hea").write_text("parent/2 1 250 4\nfirst 2\nsecond 2\n")
    target = (
        _archive_records(tmp_path, "parent", [parent, first, second])
        if in_archive
        else parent
    )
    assert "signal_read" in [issue.code for issue in wfdb.check_record(target)]


def test_check_dataset_lists_same_files_as_before(tmp_path):
    """Without ignore, every header and archive below the root is checked."""
    _write_record(tmp_path / "a" / "b", "deep", signal_name="missing.dat")
    _write_record(tmp_path / ".git", "hidden", signal_name="missing.dat")
    issues = wfdb.check_dataset(tmp_path)
    assert sorted(issue.path.name for issue in issues) == ["missing.dat"] * 2


def test_check_dataset_skips_ignored_folders_and_files(tmp_path):
    """Ignored folders are not read, and ignored files are not checked."""
    _write_record(tmp_path, "good")
    _write_record(tmp_path / ".git", "hidden", signal_name="missing.dat")
    _write_record(tmp_path / "drafts", "draft", signal_name="missing.dat")
    _write_record(tmp_path, "scratch", signal_name="missing.dat")
    seen = []

    def ignore(path):
        seen.append(path)
        return path.parts[0] in (".git", "drafts") or path.name == "scratch.hea"

    assert wfdb.check_dataset(tmp_path, ignore=ignore) == []
    # The contents of ignored folders are never offered to the filter
    assert Path(".git") in seen
    assert Path(".git", "hidden.hea") not in seen


def test_check_dataset_still_checks_ignored_segments(tmp_path):
    """A segment is checked through its parent even if its own file is ignored."""
    _write_record(tmp_path, "segment", signal_name="missing.dat")
    (tmp_path / "parent.hea").write_text("parent/1 1 250 2\nsegment 2\n")
    issues = wfdb.check_dataset(
        tmp_path, ignore=lambda path: path.name == "segment.hea"
    )
    assert [issue.code for issue in issues] == ["missing_signal"]


def test_check_dataset_still_checks_ignored_signal_files(tmp_path):
    """Ignore selects records; a checked record's signal files are checked."""
    record = _write_record(tmp_path, "rec")
    record.with_suffix(".dat").write_bytes(b"\x00\x00")
    issues = wfdb.check_dataset(tmp_path, ignore=lambda path: path.name == "rec.dat")
    assert [issue.code for issue in issues] == ["short_signal"]


def test_check_dataset_ignores_archives(tmp_path):
    record = _write_record(tmp_path / "src", signal_name="missing.dat")
    _archive_records(tmp_path, "bad", [record])
    issues = wfdb.check_dataset(
        tmp_path, ignore=lambda path: path.parts[0] == "src" or path.suffix == ".wfdb"
    )
    assert issues == []


def test_check_dataset_checks_bad_archive(tmp_path):
    record = _write_record(tmp_path, signal_name="missing.dat")
    archive = _archive_records(tmp_path, "good", [record])
    issues = wfdb.check_dataset(tmp_path)
    assert any(
        issue.path == archive.resolve() and issue.code == "missing_signal"
        for issue in issues
    )


def test_check_archive_rejects_symlink_entry(tmp_path):
    archive = tmp_path / "linked.wfdb"
    info = zipfile.ZipInfo("linked.hea")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(info, "/etc/passwd")
    assert [issue.code for issue in wfdb.check_record(archive)] == ["symlink"]


@pytest.mark.skipif(
    sys.platform == "win32", reason="symlink permissions vary on Windows"
)
def test_check_record_rejects_local_symlink_header(tmp_path):
    target = tmp_path / "target"
    target.write_text("secret content\n")
    (tmp_path / "linked.hea").symlink_to(target)
    assert [issue.code for issue in wfdb.check_record(tmp_path / "linked")] == [
        "symlink"
    ]


def test_check_cli_reports_failure(tmp_path, capsys):
    record = _write_record(tmp_path, signal_name="missing.dat")
    assert main(["check", str(record)]) == 1
    assert "missing_signal" in capsys.readouterr().out


def test_check_cli_reports_bad_dataset(tmp_path, capsys):
    _write_record(tmp_path, signal_name="missing.dat")
    assert main(["check", str(tmp_path)]) == 1
    assert "missing_signal" in capsys.readouterr().out


def test_check_record_rejects_signal_path():
    with pytest.raises(ValueError, match="Expected a record name"):
        wfdb.check_record(Path("record.dat"))


def test_check_cli_explains_signal_path(capsys):
    with pytest.raises(SystemExit, match="2"):
        main(["check", "record.dat"])
    assert "Expected a record name" in capsys.readouterr().err
