import os
import pathlib
import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import wfdb


class TestAnnotation:
    """Testing read and write of WFDB annotations, including Physionet streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24
    """

    def test_1(self, source, temp_path):
        """Read a standard annotation file.

        Target file created with:
            rdann -r sample-data/100 -a atr > ann-1
        """
        annotation = wfdb.rdann("sample-data/100", "atr")

        # This is not the fault of the script. The annotation file specifies a
        # length 3
        annotation.aux_note[0] = "(N"
        # aux_note field with a null written after '(N' which the script correctly picks
        # up. I am just getting rid of the null in this unit test to compare with the
        # regexp output below which has no null to detect in the output text file of
        # rdann.

        # Target data from WFDB software package
        with open("tests/target-output/ann-1") as f:
            lines = tuple(f)
        nannot = len(lines)

        target_time = [None] * nannot
        target_sample = np.empty(nannot, dtype="object")
        target_symbol = [None] * nannot
        target_subtype = np.empty(nannot, dtype="object")
        target_chan = np.empty(nannot, dtype="object")
        target_num = np.empty(nannot, dtype="object")
        target_aux_note = [None] * nannot

        RXannot = re.compile(
            r"[ \t]*(?P<time>[\[\]\w\.:]+) +(?P<sample>\d+) +(?P<symbol>.) +(?P<subtype>\d+) +(?P<chan>\d+) +(?P<num>\d+)\t?(?P<aux_note>.*)"
        )

        for i in range(0, nannot):
            (
                target_time[i],
                target_sample[i],
                target_symbol[i],
                target_subtype[i],
                target_chan[i],
                target_num[i],
                target_aux_note[i],
            ) = RXannot.findall(lines[i])[0]

        # Convert objects into integers
        target_sample = target_sample.astype("int")
        target_num = target_num.astype("int")
        target_subtype = target_subtype.astype("int")
        target_chan = target_chan.astype("int")

        # Compare
        comp = [
            np.array_equal(annotation.sample, target_sample),
            np.array_equal(annotation.symbol, target_symbol),
            np.array_equal(annotation.subtype, target_subtype),
            np.array_equal(annotation.chan, target_chan),
            np.array_equal(annotation.num, target_num),
            annotation.aux_note == target_aux_note,
        ]

        # Test file streaming
        if source == "physionet":
            pn_annotation = wfdb.rdann(
                "100",
                "atr",
                pn_dir="mitdb",
                return_label_elements=["label_store", "symbol"],
            )
            pn_annotation.aux_note[0] = "(N"
            pn_annotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True, write_dir=temp_path)
        write_annotation = wfdb.rdann(
            os.path.join(temp_path, "100"),
            "atr",
            return_label_elements=["label_store", "symbol"],
        )
        write_annotation.create_label_map()

        assert comp == [True] * 6
        if source == "physionet":
            assert annotation.__eq__(pn_annotation)
        assert annotation.__eq__(write_annotation)

    def test_2(self, source, temp_path):
        """Annotation file with many aux_note strings.

        Target file created with:
            rdann -r sample-data/100 -a atr > ann-2
        """
        annotation = wfdb.rdann("sample-data/12726", "anI")

        # Target data from WFDB software package
        with open("tests/target-output/ann-2") as f:
            lines = tuple(f)
        nannot = len(lines)

        target_time = [None] * nannot
        target_sample = np.empty(nannot, dtype="object")
        target_symbol = [None] * nannot
        target_subtype = np.empty(nannot, dtype="object")
        target_chan = np.empty(nannot, dtype="object")
        target_num = np.empty(nannot, dtype="object")
        target_aux_note = [None] * nannot

        RXannot = re.compile(
            r"[ \t]*(?P<time>[\[\]\w\.:]+) +(?P<sample>\d+) +(?P<symbol>.) +(?P<subtype>\d+) +(?P<chan>\d+) +(?P<num>\d+)\t?(?P<aux_note>.*)"
        )

        for i in range(0, nannot):
            (
                target_time[i],
                target_sample[i],
                target_symbol[i],
                target_subtype[i],
                target_chan[i],
                target_num[i],
                target_aux_note[i],
            ) = RXannot.findall(lines[i])[0]

        # Convert objects into integers
        target_sample = target_sample.astype("int")
        target_num = target_num.astype("int")
        target_subtype = target_subtype.astype("int")
        target_chan = target_chan.astype("int")

        # Compare
        comp = [
            np.array_equal(annotation.sample, target_sample),
            np.array_equal(annotation.symbol, target_symbol),
            np.array_equal(annotation.subtype, target_subtype),
            np.array_equal(annotation.chan, target_chan),
            np.array_equal(annotation.num, target_num),
            annotation.aux_note == target_aux_note,
        ]
        # Test file streaming
        if source == "physionet":
            pn_annotation = wfdb.rdann(
                "12726",
                "anI",
                pn_dir="prcp",
                return_label_elements=["label_store", "symbol"],
            )
            pn_annotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True, write_dir=temp_path)
        write_annotation = wfdb.rdann(
            os.path.join(temp_path, "12726"),
            "anI",
            return_label_elements=["label_store", "symbol"],
        )
        write_annotation.create_label_map()

        assert comp == [True] * 6
        if source == "physionet":
            assert annotation.__eq__(pn_annotation)
        assert annotation.__eq__(write_annotation)

    def test_3(self, source, temp_path):
        """Annotation file with custom annotation types

        Target file created with:
            rdann -r sample-data/1003 -a atr > ann-3
        """
        annotation = wfdb.rdann("sample-data/1003", "atr")

        # Target data from WFDB software package
        with open("tests/target-output/ann-3") as f:
            lines = tuple(f)
        nannot = len(lines)

        target_time = [None] * nannot
        target_sample = np.empty(nannot, dtype="object")
        target_symbol = [None] * nannot
        target_subtype = np.empty(nannot, dtype="object")
        target_chan = np.empty(nannot, dtype="object")
        target_num = np.empty(nannot, dtype="object")
        target_aux_note = [None] * nannot

        RXannot = re.compile(
            r"[ \t]*(?P<time>[\[\]\w\.:]+) +(?P<sample>\d+) +(?P<symbol>.) +(?P<subtype>\d+) +(?P<chan>\d+) +(?P<num>\d+)\t?(?P<aux_note>.*)"
        )

        for i in range(0, nannot):
            (
                target_time[i],
                target_sample[i],
                target_symbol[i],
                target_subtype[i],
                target_chan[i],
                target_num[i],
                target_aux_note[i],
            ) = RXannot.findall(lines[i])[0]

        # Convert objects into integers
        target_sample = target_sample.astype("int")
        target_num = target_num.astype("int")
        target_subtype = target_subtype.astype("int")
        target_chan = target_chan.astype("int")

        # Compare
        comp = [
            np.array_equal(annotation.sample, target_sample),
            np.array_equal(annotation.symbol, target_symbol),
            np.array_equal(annotation.subtype, target_subtype),
            np.array_equal(annotation.chan, target_chan),
            np.array_equal(annotation.num, target_num),
            annotation.aux_note == target_aux_note,
        ]

        # Test file streaming
        if source == "physionet":
            pn_annotation = wfdb.rdann(
                "1003",
                "atr",
                pn_dir="challenge-2014/set-p2",
                return_label_elements=["label_store", "symbol"],
            )
            pn_annotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True, write_dir=temp_path)
        write_annotation = wfdb.rdann(
            os.path.join(temp_path, "1003"),
            "atr",
            return_label_elements=["label_store", "symbol"],
        )
        write_annotation.create_label_map()

        assert comp == [True] * 6
        if source == "physionet":
            assert annotation.__eq__(pn_annotation)
        assert annotation.__eq__(write_annotation)

    def test_4(self, temp_path):
        """Read and write annotations with large time skips

        Annotation file created by:
            echo "xxxxxxxxx 10000000000 N 0 0 0" | wrann -r huge -a qrs
        """
        annotation = wfdb.rdann("sample-data/huge", "qrs")
        assert annotation.sample[0] == 10000000000
        annotation.wrann(write_dir=temp_path)

        annotation1 = wfdb.rdann("sample-data/huge", "qrs")
        annotation2 = wfdb.rdann(os.path.join(temp_path, "huge"), "qrs")
        assert annotation1 == annotation2

    def test_5(self, temp_path):
        """Write and read annotations with custom labels."""
        ann_idx = np.array([1, 1000, 2000, 3000])
        ann_chan = np.array([3, 1, 2, 3])
        # write custom labels
        ann_label_store = np.array([4, 2, 1, 3])
        ann_custom_labels = {
            "label_store": [1, 2, 3, 4],
            "symbol": ["v", "l", "r", "z"],
            "description": ["pvc", "lbbb", "rbbb", "pac"],
        }
        ann_custom_labels = pd.DataFrame(data=ann_custom_labels)
        wfdb.wrann(
            "CustomLabel",
            "atr",
            ann_idx,
            chan=ann_chan,
            custom_labels=ann_custom_labels,
            label_store=ann_label_store,
            write_dir=temp_path,
        )
        ann = wfdb.rdann(os.path.join(temp_path, "CustomLabel"), "atr")
        assert ann.symbol == ["z", "l", "v", "r"]


def test_aux_note_round_trip(temp_path):
    """Latin-1 aux notes up to 255 characters are read back unchanged."""
    aux_note = ["x" * 255, "37°C, Müller", "(AFIB"]
    wfdb.wrann(
        "aux",
        "atr",
        np.array([1, 2, 3]),
        symbol=["+", "N", "+"],
        aux_note=aux_note,
        write_dir=temp_path,
    )
    ann = wfdb.rdann(os.path.join(temp_path, "aux"), "atr")
    assert ann.sample.tolist() == [1, 2, 3]
    assert ann.aux_note == aux_note


@pytest.mark.parametrize(
    "aux_note, match",
    [("x" * 256, "at most 255 characters"), ("(€)", "Latin-1")],
)
def test_aux_note_rejected(temp_path, aux_note, match):
    """Aux notes that can't be stored raise an error instead of being corrupted."""
    with pytest.raises(ValueError, match=match):
        wfdb.wrann(
            "bad_aux",
            "atr",
            np.array([1, 2]),
            symbol=["+", "N"],
            aux_note=[aux_note, ""],
            write_dir=temp_path,
        )
    assert not os.path.exists(os.path.join(temp_path, "bad_aux.atr"))


def test_aux_note_none(temp_path):
    """None in aux_note means no note, as for an empty string."""
    for name, aux_note in [("none", [None, "(VT", None]), ("empty", ["", "(VT", ""])]:
        wfdb.wrann(
            name,
            "atr",
            np.array([1, 5, 9]),
            symbol=["N", "N", "V"],
            aux_note=aux_note,
            write_dir=temp_path,
        )
    ann = wfdb.rdann(os.path.join(temp_path, "none"), "atr")
    assert ann.aux_note == ["", "(VT", ""]
    with (
        open(os.path.join(temp_path, "none.atr"), "rb") as f1,
        open(os.path.join(temp_path, "empty.atr"), "rb") as f2,
    ):
        assert f1.read() == f2.read()


def test_contained_labels_from_symbols():
    """Contained labels can be found from symbols alone"""
    ann = wfdb.rdann("sample-data/100", "atr")
    assert ann.label_store is None
    labels = ann.get_contained_labels(inplace=False)
    assert list(labels.columns) == [
        "label_store",
        "symbol",
        "description",
        "n_occurrences",
    ]
    assert labels.loc[1, "symbol"] == "N"
    assert labels["n_occurrences"].sum() == len(ann.sample)


def test_wrann_float_samples(temp_path):
    """Sample numbers stored as whole-number floats are written as integers"""
    sample = np.array([100, 2500, 70000])
    symbol = ["N", "V", "N"]
    wfdb.wrann("int", "atr", sample, symbol=symbol, fs=360, write_dir=temp_path)

    float_sample = sample.astype("float64")
    wfdb.wrann("float", "atr", float_sample, symbol=symbol, fs=360, write_dir=temp_path)
    assert float_sample.dtype == np.float64

    with (
        open(os.path.join(temp_path, "int.atr"), "rb") as f1,
        open(os.path.join(temp_path, "float.atr"), "rb") as f2,
    ):
        assert f1.read() == f2.read()


@pytest.mark.parametrize("sample", [[10.0, 20.5], [10.0, np.nan], [10.0, np.inf]])
def test_wrann_non_whole_samples(temp_path, sample):
    with pytest.raises(ValueError, match="whole numbers"):
        wfdb.wrann("x", "atr", np.array(sample), symbol=["N", "N"], write_dir=temp_path)


@pytest.mark.parametrize(
    "sample", [[10, 20], [10.0, 20.0], [np.float32(10), Fraction(20)]]
)
def test_wrann_object_samples(temp_path, sample):
    """Whole numbers in an object array are written as integers"""
    wfdb.wrann(
        "x",
        "atr",
        np.array(sample, dtype=object),
        symbol=["N", "N"],
        write_dir=temp_path,
    )
    ann = wfdb.rdann(os.path.join(temp_path, "x"), "atr")
    assert ann.sample.tolist() == [10, 20]


@pytest.mark.parametrize(
    "sample",
    [[10.0, 20.5], [10, Fraction(41, 2)], [10, Decimal("20.5")], [10.0, np.nan]],
)
def test_wrann_non_whole_object_samples(temp_path, sample):
    with pytest.raises(ValueError, match="whole numbers"):
        wfdb.wrann(
            "x",
            "atr",
            np.array(sample, dtype=object),
            symbol=["N", "N"],
            write_dir=temp_path,
        )


def test_rdann_path():
    """rdann accepts a pathlib.Path as the record name"""
    expected = wfdb.rdann("sample-data/100", "atr")
    annotation = wfdb.rdann(pathlib.Path("sample-data/100"), "atr")
    assert annotation.record_name == "100"
    np.testing.assert_array_equal(annotation.sample, expected.sample)
    assert annotation.symbol == expected.symbol


def test_subtype_chan_num_round_trip(temp_path):
    """Subtype (signed), chan and num values are read back unchanged."""
    subtype = np.array([-128, -1, 0, 127])
    chan = np.array([0, 255, 255, 3])
    num = np.array([5, 5, 100, 127])
    wfdb.wrann(
        "fields",
        "atr",
        np.array([10, 20, 30, 40]),
        symbol=["N", "V", "N", "A"],
        subtype=subtype,
        chan=chan,
        num=num,
        write_dir=temp_path,
    )
    ann = wfdb.rdann(os.path.join(temp_path, "fields"), "atr")
    assert ann.subtype.tolist() == subtype.tolist()
    assert ann.chan.tolist() == chan.tolist()
    assert ann.num.tolist() == num.tolist()


def test_rdann_fs_without_header(tmp_path):
    """fs is None if the annotation file has no fs and there is no header."""
    wfdb.wrann(
        "nohea", "atr", np.array([10, 20]), symbol=["N", "N"], write_dir=tmp_path
    )
    assert not (tmp_path / "nohea.hea").exists()
    ann = wfdb.rdann(str(tmp_path / "nohea"), "atr")
    assert ann.fs is None


def test_rdann_fs_header_error(tmp_path, monkeypatch):
    """Errors other than a missing header are raised, not ignored."""
    wfdb.wrann("err", "atr", np.array([10, 20]), symbol=["N", "N"], write_dir=tmp_path)

    def rdheader(*args, **kwargs):
        raise wfdb.io._url.NetFileError("502 Error: Bad Gateway", status_code=502)

    monkeypatch.setattr(wfdb.io.record, "rdheader", rdheader)
    with pytest.raises(wfdb.io._url.NetFileError, match="Bad Gateway"):
        wfdb.rdann(str(tmp_path / "err"), "atr")


@pytest.mark.parametrize(
    "label_elements", [None, ["label_store", "symbol", "description"]]
)
def test_apply_range_keeps_every_field_consistent(label_elements):
    """apply_range gives the same result as reading only that range."""
    kwargs = {} if label_elements is None else {"return_label_elements": label_elements}
    ann = wfdb.rdann("sample-data/100", "atr", **kwargs)
    ann.apply_range(sampfrom=600000, sampto=620000)

    expected = wfdb.rdann(
        "sample-data/100", "atr", sampfrom=600000, sampto=620000, **kwargs
    )
    assert ann.ann_len == expected.ann_len == 71
    for field in ["sample", "subtype", "chan", "num"]:
        np.testing.assert_array_equal(getattr(ann, field), getattr(expected, field))
    for field in ["label_store", "symbol", "description", "aux_note"]:
        value, expected_value = getattr(ann, field), getattr(expected, field)
        if expected_value is None:
            # Fields that are not set stay None
            assert value is None
        else:
            assert list(value) == list(expected_value)


def test_rdann_sampfrom_without_sampto():
    """rdann with only sampfrom returns the annotations from sampfrom onwards."""
    full = wfdb.rdann("sample-data/100", "atr")
    start = np.searchsorted(full.sample, 600000)
    ann = wfdb.rdann("sample-data/100", "atr", sampfrom=600000)
    np.testing.assert_array_equal(ann.sample, full.sample[start:])
    assert ann.symbol == full.symbol[start:]
    assert ann.ann_len == full.ann_len - start
    shifted = wfdb.rdann("sample-data/100", "atr", sampfrom=600000, shift_samps=True)
    np.testing.assert_array_equal(shifted.sample, full.sample[start:] - 600000)


def test_rdann_sampfrom_empty_file(tmp_path):
    """rdann with only sampfrom reads an annotation file with no annotations."""
    # A file that holds only the end-of-file marker
    (tmp_path / "empty.atr").write_bytes(b"\x00\x00")
    ann = wfdb.rdann(str(tmp_path / "empty"), "atr", sampfrom=10)
    assert ann.sample.size == 0


@pytest.mark.parametrize(
    ("sampfrom", "sampto"), [(0, 0), (10, 0), (100, 100), (100, 50)]
)
def test_rdann_sampto_must_be_after_sampfrom(sampfrom, sampto):
    """As in rdrecord, sampto must be greater than sampfrom (0 is not 'no limit')."""
    with pytest.raises(ValueError, match="sampto must be greater than sampfrom"):
        wfdb.rdann("sample-data/100", "atr", sampfrom=sampfrom, sampto=sampto)


def test_rdann_summarize_labels_without_annotations():
    """summarize_labels works when the range holds no annotations."""
    ann = wfdb.rdann("sample-data/100", "atr", sampfrom=10**9, summarize_labels=True)
    assert ann.ann_len == 0
    assert ann.contained_labels.empty
    assert "n_occurrences" in ann.contained_labels.columns
