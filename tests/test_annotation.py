import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

import wfdb


class TestAnnotation(unittest.TestCase):
    """
    Testing read and write of WFDB annotations, including Physionet
    streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24
    """

    def test_1(self):
        """
        Target file created with:
            rdann -r sample-data/100 -a atr > ann-1
        """
        annotation = wfdb.rdann("sample-data/100", "atr")

        # This is not the fault of the script. The annotation file specifies a
        # length 3
        annotation.aux_note[0] = "(N"
        # aux_note field with a null written after '(N' which the script correctly picks up. I am just
        # getting rid of the null in this unit test to compare with the regexp output below which has
        # no null to detect in the output text file of rdann.

        # Target data from WFDB software package
        with open("tests/target-output/ann-1", "r") as f:
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
        pn_annotation = wfdb.rdann(
            "100",
            "atr",
            pn_dir="mitdb",
            return_label_elements=["label_store", "symbol"],
        )
        pn_annotation.aux_note[0] = "(N"
        pn_annotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True, write_dir=self.temp_path)
        write_annotation = wfdb.rdann(
            os.path.join(self.temp_path, "100"),
            "atr",
            return_label_elements=["label_store", "symbol"],
        )
        write_annotation.create_label_map()

        assert comp == [True] * 6
        assert annotation.__eq__(pn_annotation)
        assert annotation.__eq__(write_annotation)

    def test_2(self):
        """
        Annotation file with many aux_note strings.

        Target file created with:
            rdann -r sample-data/100 -a atr > ann-2
        """
        annotation = wfdb.rdann("sample-data/12726", "anI")

        # Target data from WFDB software package
        with open("tests/target-output/ann-2", "r") as f:
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
        pn_annotation = wfdb.rdann(
            "12726",
            "anI",
            pn_dir="prcp",
            return_label_elements=["label_store", "symbol"],
        )
        pn_annotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True, write_dir=self.temp_path)
        write_annotation = wfdb.rdann(
            os.path.join(self.temp_path, "12726"),
            "anI",
            return_label_elements=["label_store", "symbol"],
        )
        write_annotation.create_label_map()

        assert comp == [True] * 6
        assert annotation.__eq__(pn_annotation)
        assert annotation.__eq__(write_annotation)

    def test_3(self):
        """
        Annotation file with custom annotation types

        Target file created with:
            rdann -r sample-data/1003 -a atr > ann-3
        """
        annotation = wfdb.rdann("sample-data/1003", "atr")

        # Target data from WFDB software package
        with open("tests/target-output/ann-3", "r") as f:
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
        pn_annotation = wfdb.rdann(
            "1003",
            "atr",
            pn_dir="challenge-2014/set-p2",
            return_label_elements=["label_store", "symbol"],
        )
        pn_annotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True, write_dir=self.temp_path)
        write_annotation = wfdb.rdann(
            os.path.join(self.temp_path, "1003"),
            "atr",
            return_label_elements=["label_store", "symbol"],
        )
        write_annotation.create_label_map()

        assert comp == [True] * 6
        assert annotation.__eq__(pn_annotation)
        assert annotation.__eq__(write_annotation)

    def test_4(self):
        """
        Read and write annotations with large time skips

        Annotation file created by:
            echo "xxxxxxxxx 10000000000 N 0 0 0" | wrann -r huge -a qrs
        """
        annotation = wfdb.rdann("sample-data/huge", "qrs")
        self.assertEqual(annotation.sample[0], 10000000000)
        annotation.wrann(write_dir=self.temp_path)

        annotation1 = wfdb.rdann("sample-data/huge", "qrs")
        annotation2 = wfdb.rdann(os.path.join(self.temp_path, "huge"), "qrs")
        self.assertEqual(annotation1, annotation2)

    def test_5(self):
        """
        Write and read annotations with custom labels.
        """
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
            write_dir=self.temp_path,
        )
        ann = wfdb.rdann(os.path.join(self.temp_path, "CustomLabel"), "atr")
        self.assertEqual(ann.symbol, ["z", "l", "v", "r"])

    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_directory.name

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
