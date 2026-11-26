"""Tests for local WFDB submission checks."""

import struct
import zipfile

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


def test_check_record_rejects_wrong_name_and_missing_signal(tmp_path):
    record = _write_record(tmp_path, signal_name="elsewhere.dat")
    header = record.with_suffix(".hea")
    header.write_text(header.read_text().replace("good 1", "wrong 1"))
    issues = wfdb.check_record(record)
    assert {issue.code for issue in issues} == {"record_name", "missing_signal"}


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


def test_check_cli_reports_failure(tmp_path, capsys):
    record = _write_record(tmp_path, signal_name="missing.dat")
    assert main(["check", str(record)]) == 1
    assert "missing_signal" in capsys.readouterr().out
