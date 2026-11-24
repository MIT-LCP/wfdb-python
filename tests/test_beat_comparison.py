"""Tests for the beat-by-beat comparison of annotations (bxb).

The expected outputs in tests/target-output/bxb were made with the bxb and
sumstats programs of the WFDB Software Package (10.7.0), from the reference
(.atr) and test (.tst, .qrs, and sample-data/100.qrs) annotation files there.
If bxb is installed, test_matches_wfdb_bxb also compares many generated cases
with it.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wfdb
from wfdb.processing import bxb, bxb_record, sumstats
from wfdb.processing.beat_comparison import LINE_HEADINGS

TARGET = Path(__file__).parent / "target-output" / "bxb"
SAMPLE_DATA = Path(__file__).parent.parent / "sample-data"

# The records compared for the expected outputs, and the options of each
CASES = [
    (SAMPLE_DATA / "100", "qrs", {}),
    (TARGET / "bxb360", "tst", {}),
    (TARGET / "bxb250", "tst", {"start": 81000, "stop": 110000, "window": 0.12}),
]


def results():
    return [bxb_record(str(path), "atr", test, **kw) for path, test, kw in CASES]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (CASES[0], "100-report.txt"),
        (CASES[1], "bxb360-report.txt"),
        (CASES[2], "bxb250-report.txt"),
        # A test file that ends before the reference file, compared as with
        # bxb -t 0, which stops at the end of either file
        ((TARGET / "bxb360", "qrs", {"stop": 0}), "bxb360-stop0-report.txt"),
    ],
    ids=["100-qrs", "vf-and-shutdowns", "250hz-with-options", "stop-0"],
)
def test_report_matches_bxb(case, expected):
    """The report is the same as that of bxb -S."""
    path, test, kw = case
    result = bxb_record(str(path), "atr", test, **kw)
    assert result.report() == (TARGET / expected).read_text()


def test_stop_0_stops_at_the_end_of_either_file():
    reference = wfdb.Annotation(
        "rec", "atr", sample=np.array([360, 720, 1080]), symbol=["N"] * 3, fs=360
    )
    test = wfdb.Annotation(
        "rec", "tst", sample=np.array([360, 720]), symbol=["N"] * 2, fs=360
    )

    result = bxb(reference, test, start=0, stop=0)

    # Both test beats are matched, and the reference beat after the end of the
    # test file is not counted as missed
    assert result.count("N", "n") == 2
    assert result.count("N", "o") == 0
    assert bxb(reference, test, start=0, stop=1080).count("N", "o") == 1


def test_line_format_matches_bxb():
    """The line-format reports are the same as those of bxb -L."""
    lines = [r.lines() for r in results()]
    assert LINE_HEADINGS[0] + "".join(a + "\n" for a, b in lines) == (
        (TARGET / "bxb.out").read_text()
    )
    assert LINE_HEADINGS[1] + "".join(b + "\n" for a, b in lines) == (
        (TARGET / "sd.out").read_text()
    )


def test_sumstats_matches_wfdb_sumstats():
    summary, shutdown = sumstats(results())
    assert summary == (TARGET / "sumstats-bxb.txt").read_text()
    assert shutdown == (TARGET / "sumstats-sd.txt").read_text()


def test_matrix_and_statistics():
    result = bxb_record(str(SAMPLE_DATA / "100"), "atr", "qrs")

    matrix = result.matrix
    assert list(matrix.index) == list("NSVFQOX")
    assert list(matrix.columns) == list("nsvfqox")
    assert matrix.loc["N", "n"] == 1872
    assert matrix.loc["S", "n"] == 29
    # Test beats with no reference beat have no "no test beat" columns
    assert pd.isna(matrix.loc["O", "o"]) and pd.isna(matrix.loc["X", "x"])

    stats = result.statistics()
    assert stats["QRS sensitivity"] == (1902, 1902)
    assert stats["SVEB sensitivity"] == (0, 29)
    assert stats["VEB positive predictivity"] == (0, 0)
    assert result.rms_rr_error == pytest.approx(1.96, abs=0.005)
    assert result.shutdown_seconds == 0


def test_annotations_with_symbols_only():
    """Annotations read without label_store are compared by their symbols."""
    reference = wfdb.rdann(str(SAMPLE_DATA / "100"), "atr")
    test = wfdb.rdann(str(SAMPLE_DATA / "100"), "qrs")
    assert reference.label_store is None

    result = bxb(reference, test, sig_len=650000)

    expected = bxb_record(str(SAMPLE_DATA / "100"), "atr", "qrs")
    assert result.counts == expected.counts
    assert result.rr_errors == expected.rr_errors


def test_undefined_rr_error():
    """With a single beat, there are no RR intervals to compare."""
    one = wfdb.Annotation("rec", "atr", sample=np.array([200000]), symbol=["N"], fs=360)
    result = bxb(one, one, sig_len=650000)
    assert result.rms_rr_error is None
    assert "RMS RR interval error:      -\n" in result.report()


def test_sampling_frequency_is_needed():
    ann = wfdb.Annotation("rec", "atr", sample=np.array([200000]), symbol=["N"])
    with pytest.raises(ValueError, match="sampling frequency"):
        bxb(ann, ann)


def _random_case(rng, fs):
    """Reference and test annotations with most of what bxb handles."""
    scale = fs / 360
    n = 300
    sig_len = int(rng.integers(600, 1500) * fs)
    t0 = int(rng.integers(0, sig_len - int(n * 330 * scale)))
    base = t0 + np.cumsum(rng.integers(int(150 * scale), int(500 * scale), n))
    beats = [1, 1, 1, 2, 5, 5, 6, 8, 4, 13, 12, 10, 34]
    codes = rng.choice(beats, n)
    ref = [(int(t), int(c), 0) for t, c in zip(base, codes)]
    tst = []
    for t, c in zip(base, codes):
        if rng.random() < 0.07:
            continue
        jitter = int(rng.normal(0, 25 * scale)) if rng.random() < 0.3 else 0
        tst.append(
            (int(t) + jitter, c if rng.random() < 0.8 else int(rng.choice(beats)), 0)
        )
        if rng.random() < 0.05:
            tst.append(
                (int(t) + int(rng.integers(30, 200) * scale), int(rng.choice(beats)), 0)
            )
    for f in (ref, tst):
        for _ in range(int(rng.integers(0, 5))):
            f.append((int(rng.choice(base)) + 7, int(rng.choice([16, 28, 37])), 0))
        if rng.random() < 0.4:  # a VF episode
            a = int(rng.choice(base)) + 3
            f += [(a, 32, 0), (a + int(rng.integers(2, 15) * fs), 33, 0)]
        if rng.random() < 0.5:  # a shutdown, paired or AHA style
            a = int(rng.choice(base)) + 5
            b = a + int(rng.integers(1, 20) * fs)
            f[:] = [e for e in f if not a < e[0] < b]
            if rng.random() < 0.7:
                f += [(a, 14, int(rng.choice([48, -1]))), (b, 14, 0)]
            else:
                f.append(((a + b) // 2, 14, -1))
    return sig_len, sorted(ref), sorted(tst)


def _write(directory, name, extension, annotations):
    sample, code, subtype = (np.array(x) for x in zip(*annotations))
    wfdb.Annotation(
        name, extension, sample=sample, label_store=code, subtype=subtype
    ).wrann(write_dir=str(directory))


@pytest.mark.skipif(shutil.which("bxb") is None, reason="WFDB bxb is not installed")
@pytest.mark.parametrize("seed", range(4))
def test_matches_wfdb_bxb(tmp_path, seed):
    """Generated cases give the same reports as the installed WFDB bxb."""
    rng = np.random.default_rng(seed)
    env = {**os.environ, "WFDB": str(tmp_path)}
    for k in range(50):
        fs = int(rng.choice([128, 250, 257, 360, 500, 1000]))
        sig_len, ref, tst = _random_case(rng, fs)
        name = f"r{k}"
        (tmp_path / f"{name}.hea").write_text(
            f"{name} 1 {fs} {sig_len}\n{name}.dat 16 200 11 1024 0 0 0 ECG\n"
        )
        _write(tmp_path, name, "atr", ref)
        _write(tmp_path, name, "tst", tst)
        args, kw = [], {}
        if rng.random() < 0.3:
            kw["start"] = int(rng.integers(0, sig_len // 2))
            args += ["-f", f"s{kw['start']}"]
        u = rng.random()
        if u < 0.2:
            kw["stop"] = int(rng.integers(sig_len // 2, sig_len))
            args += ["-t", f"s{kw['stop']}"]
        elif u < 0.35:
            kw["stop"] = 0
            args += ["-t", "0"]
        if rng.random() < 0.3:
            kw["window"] = float(rng.choice([0.05, 0.1, 0.2]))
            args += ["-w", str(kw["window"])]

        run = ["bxb", "-r", name, "-a", "atr", "tst", *args]
        expected = subprocess.run(
            [*run, "-S", "-"], cwd=tmp_path, env=env, capture_output=True, text=True
        ).stdout
        result = bxb_record(str(tmp_path / name), "atr", "tst", **kw)
        assert result.report() == expected, f"seed {seed}, case {k}, {args}"
