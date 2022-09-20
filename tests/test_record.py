import datetime
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import wfdb


class TestRecord(unittest.TestCase):
    """
    Test read and write of single segment WFDB records, including
    PhysioNet streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24

    """

    # ----------------------- 1. Basic Tests -----------------------#

    def test_1a(self):
        """
        Format 16, entire signal, digital.

        Target file created with:
            rdsamp -r sample-data/test01_00s | cut -f 2- > record-1a
        """
        record = wfdb.rdrecord(
            "sample-data/test01_00s", physical=False, return_res=16
        )
        sig = record.d_signal
        sig_target = np.genfromtxt("tests/target-output/record-1a")

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "test01_00s", pn_dir="macecgdb", physical=False, return_res=16
        )

        # Test file writing
        record_2 = wfdb.rdrecord(
            "sample-data/test01_00s", physical=False, return_res=16
        )
        record_2.sig_name = ["ECG_1", "ECG_2", "ECG_3", "ECG_4"]
        record_2.wrsamp(write_dir=self.temp_path)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "test01_00s"),
            physical=False,
            return_res=16,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record_2.__eq__(record_write)

    def test_1b(self):
        """
        Format 16, byte offset, selected duration, selected channels,
        physical.

        Target file created with:
            rdsamp -r sample-data/a103l -f 50 -t 160 -s 2 0 -P | cut -f 2- > record-1b
        """
        sig, fields = wfdb.rdsamp(
            "sample-data/a103l", sampfrom=12500, sampto=40000, channels=[2, 0]
        )
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-1b")

        # Compare data streaming from Physionet
        sig_pn, fields_pn = wfdb.rdsamp(
            "a103l",
            pn_dir="challenge-2015/training",
            sampfrom=12500,
            sampto=40000,
            channels=[2, 0],
        )

        # Option of selecting channels by name
        sig_named, fields_named = wfdb.rdsamp(
            "sample-data/a103l",
            sampfrom=12500,
            sampto=40000,
            channel_names=["PLETH", "II"],
        )

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pn) and fields == fields_pn
        assert np.array_equal(sig, sig_named) and fields == fields_named

    def test_1c(self):
        """
        Format 16, byte offset, selected duration, selected channels,
        digital, expanded format.

        Target file created with:
            rdsamp -r sample-data/a103l -f 80 -s 0 1 | cut -f 2- > record-1c
        """
        record = wfdb.rdrecord(
            "sample-data/a103l",
            sampfrom=20000,
            channels=[0, 1],
            physical=False,
            smooth_frames=False,
        )
        # convert expanded to uniform array
        sig = np.zeros((record.sig_len, record.n_sig))
        for i in range(record.n_sig):
            sig[:, i] = record.e_d_signal[i]

        sig_target = np.genfromtxt("tests/target-output/record-1c")

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "a103l",
            pn_dir="challenge-2015/training",
            sampfrom=20000,
            channels=[0, 1],
            physical=False,
            smooth_frames=False,
        )

        # Test file writing
        record.wrsamp(write_dir=self.temp_path, expanded=True)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "a103l"),
            physical=False,
            smooth_frames=False,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record.__eq__(record_write)

    def test_1d(self):
        """
        Format 80, selected duration, selected channels, physical

        Target file created with:
            rdsamp -r sample-data/3000003_0003 -f 1 -t 8 -s 1 -P | cut -f 2- > record-1d
        """
        sig, fields = wfdb.rdsamp(
            "sample-data/3000003_0003", sampfrom=125, sampto=1000, channels=[1]
        )
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-1d")
        sig_target = sig_target.reshape(len(sig_target), 1)

        # Compare data streaming from Physionet
        sig_pn, fields_pn = wfdb.rdsamp(
            "3000003_0003",
            pn_dir="mimic3wdb/30/3000003/",
            sampfrom=125,
            sampto=1000,
            channels=[1],
        )

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pn) and fields == fields_pn

    def test_1e(self):
        """
        Format 24, entire signal, digital.

        Target file created with:
            rdsamp -r sample-data/n8_evoked_raw_95_F1_R9 | cut -f 2- |
            gzip -9 -n > record-1e.gz
        """
        record = wfdb.rdrecord(
            "sample-data/n8_evoked_raw_95_F1_R9", physical=False
        )
        sig = record.d_signal
        sig_target = np.genfromtxt("tests/target-output/record-1e.gz")
        sig_target[sig_target == -32768] = -(2**23)

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "n8_evoked_raw_95_F1_R9", physical=False, pn_dir="earndb/raw/N8"
        )

        # Test file writing
        record_2 = wfdb.rdrecord(
            "sample-data/n8_evoked_raw_95_F1_R9", physical=False
        )
        record_2.wrsamp(write_dir=self.temp_path)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "n8_evoked_raw_95_F1_R9"),
            physical=False,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record_2.__eq__(record_write)

    def test_1f(self):
        """
        All binary formats, multiple signal files in one record.

        Target file created with:
            rdsamp -r sample-data/binformats | cut -f 2- |
            gzip -9 -n > record-1f.gz
        """
        record = wfdb.rdrecord("sample-data/binformats", physical=False)
        sig_target = np.genfromtxt("tests/target-output/record-1f.gz")

        for n, name in enumerate(record.sig_name):
            np.testing.assert_array_equal(
                record.d_signal[:, n], sig_target[:, n], "Mismatch in %s" % name
            )

        for sampfrom in range(0, 3):
            for sampto in range(record.sig_len - 3, record.sig_len):
                record_2 = wfdb.rdrecord(
                    "sample-data/binformats",
                    physical=False,
                    sampfrom=sampfrom,
                    sampto=sampto,
                )
                for n, name in enumerate(record.sig_name):
                    if record.fmt[n] != "8":
                        np.testing.assert_array_equal(
                            record_2.d_signal[:, n],
                            sig_target[sampfrom:sampto, n],
                            "Mismatch in %s" % name,
                        )

    def test_read_write_flac(self):
        """
        All FLAC formats, multiple signal files in one record.

        Target file created with:
            rdsamp -r sample-data/flacformats | cut -f 2- |
            gzip -9 -n > record-flac.gz
        """
        record = wfdb.rdrecord("sample-data/flacformats", physical=False)
        sig_target = np.genfromtxt("tests/target-output/record-flac.gz")

        for n, name in enumerate(record.sig_name):
            np.testing.assert_array_equal(
                record.d_signal[:, n], sig_target[:, n], f"Mismatch in {name}"
            )

        for sampfrom in range(0, 3):
            for sampto in range(record.sig_len - 3, record.sig_len):
                record_2 = wfdb.rdrecord(
                    "sample-data/flacformats",
                    physical=False,
                    sampfrom=sampfrom,
                    sampto=sampto,
                )
                for n, name in enumerate(record.sig_name):
                    np.testing.assert_array_equal(
                        record_2.d_signal[:, n],
                        sig_target[sampfrom:sampto, n],
                        f"Mismatch in {name}",
                    )

        # Test file writing
        record.wrsamp(write_dir=self.temp_path)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "flacformats"),
            physical=False,
        )
        assert record == record_write

    def test_read_write_flac_multifrequency(self):
        """
        Format 516 with multiple signal files and variable samples per frame.
        """
        # Check that we can read a record and write it out again
        record = wfdb.rdrecord(
            "sample-data/mixedsignals",
            physical=False,
            smooth_frames=False,
        )
        record.wrsamp(write_dir=self.temp_path, expanded=True)

        # Check that result matches the original
        record = wfdb.rdrecord("sample-data/mixedsignals", smooth_frames=False)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "mixedsignals"),
            smooth_frames=False,
        )
        assert record == record_write

    def test_read_flac_longduration(self):
        """
        Three signals multiplexed in a FLAC file, over 2**24 samples.

        Input file created with:
            yes 25 50 75 | head -5600000 |
            wrsamp -O 508 -o flac_3_constant 0 1 2

        Note that the total number of samples (across the three
        channels) exceeds 2**24.  There is a bug in libsndfile that
        causes it to break if we try to read more than 2**24 total
        samples at a time, when the number of channels is not a power
        of two.
        """
        record = wfdb.rdrecord("sample-data/flac_3_constant")
        sig_target = np.repeat(
            np.array([[0.125, 0.25, 0.375]], dtype="float64"),
            5600000,
            axis=0,
        )
        np.testing.assert_array_equal(record.p_signal, sig_target)

    # ------------------ 2. Special format records ------------------ #

    def test_2a(self):
        """
        Format 212, entire signal, physical.

        Target file created with:
            rdsamp -r sample-data/100 -P | cut -f 2- > record-2a
        """
        sig, fields = wfdb.rdsamp("sample-data/100")
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-2a")

        # Compare data streaming from Physionet
        sig_pn, fields_pn = wfdb.rdsamp("100", pn_dir="mitdb")
        # This comment line was manually added and is not present in the
        # original PhysioNet record
        del fields["comments"][0]

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pn) and fields == fields_pn

    def test_2b(self):
        """
        Format 212, selected duration, selected channel, digital.

        Target file created with:
            rdsamp -r sample-data/100 -f 0.002 -t 30 -s 1 | cut -f 2- > record-2b
        """
        record = wfdb.rdrecord(
            "sample-data/100",
            sampfrom=1,
            sampto=10800,
            channels=[1],
            physical=False,
        )
        sig = record.d_signal
        sig_target = np.genfromtxt("tests/target-output/record-2b")
        sig_target = sig_target.reshape(len(sig_target), 1)

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "100",
            sampfrom=1,
            sampto=10800,
            channels=[1],
            physical=False,
            pn_dir="mitdb",
        )
        # This comment line was manually added and is not present in the
        # original PhysioNet record
        del record.comments[0]

        # Option of selecting channels by name
        record_named = wfdb.rdrecord(
            "sample-data/100",
            sampfrom=1,
            sampto=10800,
            channel_names=["V5"],
            physical=False,
        )
        del record_named.comments[0]

        # Test file writing
        record.wrsamp(write_dir=self.temp_path)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "100"),
            physical=False,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record.__eq__(record_named)
        assert record.__eq__(record_write)

    def test_2c(self):
        """
        Format 212, entire signal, physical, odd sampled record.

        Target file created with:
            rdsamp -r sample-data/100_3chan -P | cut -f 2- > record-2c
        """
        record = wfdb.rdrecord("sample-data/100_3chan")
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-2c")

        # Test file writing
        record.d_signal = record.adc()
        record.wrsamp(write_dir=self.temp_path)
        record_write = wfdb.rdrecord(os.path.join(self.temp_path, "100_3chan"))
        record.d_signal = None

        assert np.array_equal(sig_round, sig_target)
        assert record.__eq__(record_write)

    def test_2d(self):
        """
        Format 310, selected duration, digital
        Target file created with:
            rdsamp -r sample-data/3000003_0003 -f 0 -t 8.21 | cut -f 2- | wrsamp -o 310derive -O 310
            rdsamp -r 310derive -f 0.007 | cut -f 2- > record-2d
        """
        record = wfdb.rdrecord(
            "sample-data/310derive", sampfrom=2, physical=False
        )
        sig = record.d_signal
        sig_target = np.genfromtxt("tests/target-output/record-2d")
        assert np.array_equal(sig, sig_target)

    def test_2e(self):
        """
        Format 311, selected duration, physical.

        Target file created with:
            rdsamp -r sample-data/3000003_0003 -f 0 -t 8.21 -s 1 | cut -f 2- | wrsamp -o 311derive -O 311
            rdsamp -r 311derive -f 0.005 -t 3.91 -P | cut -f 2- > record-2e
        """
        sig, fields = wfdb.rdsamp(
            "sample-data/311derive", sampfrom=1, sampto=978
        )
        sig = np.round(sig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-2e")
        sig_target = sig_target.reshape([977, 1])
        assert np.array_equal(sig, sig_target)

    # --------------------- 3. Multi-dat records --------------------- #

    def test_3a(self):
        """
        Multi-dat, entire signal, digital
        Target file created with:
            rdsamp -r sample-data/s0010_re | cut -f 2- > record-3a
        """
        record = wfdb.rdrecord("sample-data/s0010_re", physical=False)
        sig = record.d_signal
        sig_target = np.genfromtxt("tests/target-output/record-3a")

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "s0010_re", physical=False, pn_dir="ptbdb/patient001"
        )

        # Test file writing
        record.wrsamp(write_dir=self.temp_path)
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "s0010_re"),
            physical=False,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record.__eq__(record_write)

    def test_3b(self):
        """
        Multi-dat, selected duration, selected channels, physical.

        Target file created with:
            rdsamp -r sample-data/s0010_re -f 5 -t 38 -P -s 13 0 4 8 3 | cut -f 2- > record-3b
        """
        sig, fields = wfdb.rdsamp(
            "sample-data/s0010_re",
            sampfrom=5000,
            sampto=38000,
            channels=[13, 0, 4, 8, 3],
        )
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-3b")

        # Compare data streaming from Physionet
        sig_pn, fields_pn = wfdb.rdsamp(
            "s0010_re",
            sampfrom=5000,
            pn_dir="ptbdb/patient001",
            sampto=38000,
            channels=[13, 0, 4, 8, 3],
        )

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pn) and fields == fields_pn

    # -------------- 4. Skew and multiple samples/frame -------------- #

    def test_4a(self):
        """
        Format 16, multi-samples per frame, skew, digital.

        Target file created with:
            rdsamp -r sample-data/test01_00s_skewframe | cut -f 2- > record-4a
        """
        record = wfdb.rdrecord(
            "sample-data/test01_00s_skewframe", physical=False
        )
        sig = record.d_signal
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew. The WFDB python rdsamp does return the final
        # N samples, filling in NANs for end of skewed channels only.
        sig = sig[:-3, :]

        sig_target = np.genfromtxt("tests/target-output/record-4a")

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        record_no_skew = wfdb.rdrecord(
            "sample-data/test01_00s_skewframe",
            physical=False,
            smooth_frames=False,
            ignore_skew=True,
        )
        record_no_skew.wrsamp(write_dir=self.temp_path, expanded=True)
        # Read the written record
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "test01_00s_skewframe"),
            physical=False,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_write)

    def test_4b(self):
        """
        Format 12, multi-samples per frame, skew, entire signal, digital.

        Target file created with:
            rdsamp -r sample-data/03700181 | cut -f 2- > record-4b
        """
        record = wfdb.rdrecord("sample-data/03700181", physical=False)
        sig = record.d_signal
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew.
        sig = sig[:-4, :]
        # The WFDB python rdsamp does return the final N samples, filling in
        # NANs for end of skewed channels only.
        sig_target = np.genfromtxt("tests/target-output/record-4b")

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "03700181", physical=False, pn_dir="mimicdb/037"
        )

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        record_no_skew = wfdb.rdrecord(
            "sample-data/03700181",
            physical=False,
            smooth_frames=False,
            ignore_skew=True,
        )
        record_no_skew.wrsamp(write_dir=self.temp_path, expanded=True)
        # Read the written record
        record_write = wfdb.rdrecord(
            os.path.join(self.temp_path, "03700181"),
            physical=False,
        )

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record.__eq__(record_write)

    def test_4c(self):
        """
        Format 12, multi-samples per frame, skew, selected suration,
        selected channels, physical.

        Target file created with:
            rdsamp -r sample-data/03700181 -f 8 -t 128 -s 0 2 -P | cut -f 2- > record-4c
        """
        sig, fields = wfdb.rdsamp(
            "sample-data/03700181", channels=[0, 2], sampfrom=1000, sampto=16000
        )
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-4c")

        # Compare data streaming from Physionet
        sig_pn, fields_pn = wfdb.rdsamp(
            "03700181",
            pn_dir="mimicdb/037",
            channels=[0, 2],
            sampfrom=1000,
            sampto=16000,
        )

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        record_no_skew = wfdb.rdrecord(
            "sample-data/03700181",
            physical=False,
            smooth_frames=False,
            ignore_skew=True,
        )
        record_no_skew.wrsamp(write_dir=self.temp_path, expanded=True)
        # Read the written record
        writesig, writefields = wfdb.rdsamp(
            os.path.join(self.temp_path, "03700181"),
            channels=[0, 2],
            sampfrom=1000,
            sampto=16000,
        )

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pn) and fields == fields_pn
        assert np.array_equal(sig, writesig) and fields == writefields

    def test_4d(self):
        """
        Format 16, multi-samples per frame, skew, read expanded signals

        Target file created with:
            rdsamp -r sample-data/test01_00s_skewframe -P -H | cut -f 2- > record-4d
        """
        record = wfdb.rdrecord(
            "sample-data/test01_00s_skewframe", smooth_frames=False
        )

        # Upsample the channels with lower samples/frame
        expandsig = np.zeros((7994, 3))
        expandsig[:, 0] = np.repeat(record.e_p_signal[0][:-3], 2)
        expandsig[:, 1] = record.e_p_signal[1][:-6]
        expandsig[:, 2] = np.repeat(record.e_p_signal[2][:-3], 2)

        sig_round = np.round(expandsig, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-4d")

        assert np.array_equal(sig_round, sig_target)

    def test_to_dataframe(self):
        record = wfdb.rdrecord("sample-data/test01_00s")
        df = record.to_dataframe()

        self.assertEqual(record.sig_name, list(df.columns))
        self.assertEqual(len(df), record.sig_len)
        self.assertEqual(df.index[0], pd.Timedelta(0))
        self.assertEqual(
            df.index[-1],
            pd.Timedelta(seconds=1 / record.fs * (record.sig_len - 1)),
        )
        assert np.array_equal(record.p_signal, df.values)

    def test_header_with_non_utf8(self):
        """
        Ignores non-utf8 characters in the header part.
        """
        record = wfdb.rdrecord("sample-data/test_generator_2")
        sig_units_target = [
            "uV",
            "uV",
            "uV",
            "uV",
            "uV",
            "uV",
            "uV",
            "uV",
            "mV",
            "mV",
            "uV",
            "mV",
        ]
        assert record.units.__eq__(sig_units_target)

    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_directory.name

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()


class TestMultiRecord(unittest.TestCase):
    """
    Test read and write of multi segment WFDB records, including
    PhysioNet streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24

    """

    def test_multi_fixed_a(self):
        """
        Multi-segment, fixed layout, read entire signal.

        Target file created with:
            rdsamp -r sample-data/multi-segment/fixed1/v102s -P | cut -f 2- > record-multi-fixed-a
        """
        record = wfdb.rdrecord("sample-data/multi-segment/fixed1/v102s")
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-multi-fixed-a")

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_fixed_b(self):
        """
        Multi-segment, fixed layout, selected duration, samples read
        from one segment.

        Target file created with:
            rdsamp -r sample-data/multi-segment/fixed1/v102s -t s75000 -P | cut -f 2- > record-multi-fixed-b
        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/fixed1/v102s", sampto=75000
        )
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-multi-fixed-b")

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_fixed_c(self):
        """
        Multi-segment, fixed layout, selected duration and channels,
        samples read from multiple segments

        Target file created with:
            rdsamp -r sample-data/multi-segment/fixed1/v102s -f s70000 -t s80000 -s 1 0 3 -P | cut -f 2- > record-multi-fixed-c
        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/fixed1/v102s",
            sampfrom=70000,
            sampto=80000,
            channels=[1, 0, 3],
        )
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt("tests/target-output/record-multi-fixed-c")

        # Option of selecting channels by name
        record_named = wfdb.rdrecord(
            "sample-data/multi-segment/fixed1/v102s",
            sampfrom=70000,
            sampto=80000,
            channel_names=["V", "II", "RESP"],
        )

        np.testing.assert_equal(sig_round, sig_target)
        assert record.__eq__(record_named)

    def test_multi_fixed_d(self):
        """
        Multi-segment, fixed layout, multi-frequency, selected channels

        Target file created with:
            rdsamp -r sample-data/multi-segment/041s/ -s 3 2 1 -H |
            cut -f 2- | sed s/-32768/-2048/ |
            gzip -9 -n > tests/target-output/record-multi-fixed-d.gz
        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/041s/041s",
            channels=[3, 2, 1],
            physical=False,
            smooth_frames=False,
        )

        # Convert expanded to uniform array (high-resolution)
        sig = np.zeros((record.sig_len * 4, record.n_sig), dtype=int)
        for i, s in enumerate(record.e_d_signal):
            sig[:, i] = np.repeat(s, len(sig[:, i]) // len(s))

        sig_target = np.genfromtxt(
            "tests/target-output/record-multi-fixed-d.gz"
        )

        record_named = wfdb.rdrecord(
            "sample-data/multi-segment/041s/041s",
            channel_names=["ABP", "V", "I"],
            physical=False,
            smooth_frames=False,
        )

        # Sample values should match the output of rdsamp -H
        np.testing.assert_array_equal(sig, sig_target)
        # channel_names=[...] should give the same result as channels=[...]
        self.assertEqual(record, record_named)

    def test_multi_variable_a(self):
        """
        Multi-segment, variable layout, selected duration, samples read
        from one segment only.

        Target file created with:
            rdsamp -r sample-data/multi-segment/s00001/s00001-2896-10-10-00-31 -f s14428365 -t s14428375 -P | cut -f 2- > record-multi-variable-a
        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/s00001/s00001-2896-10-10-00-31",
            sampfrom=14428365,
            sampto=14428375,
        )
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt(
            "tests/target-output/record-multi-variable-a"
        )

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_variable_b(self):
        """
        Multi-segment, variable layout, selected duration, samples read
        from several segments.

        Target file created with:
            rdsamp -r sample-data/multi-segment/s00001/s00001-2896-10-10-00-31 -f s14428364 -t s14428375 -P | cut -f 2- > record-multi-variable-b
        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/s00001/s00001-2896-10-10-00-31",
            sampfrom=14428364,
            sampto=14428375,
        )
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt(
            "tests/target-output/record-multi-variable-b"
        )

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_variable_c(self):
        """
        Multi-segment, variable layout, entire signal, physical, expanded

        The reference signal creation cannot be made with rdsamp
        directly because the WFDB c package (10.5.24) applies the single
        adcgain and baseline values from the layout specification
        header, which is undesired in multi-segment signals with
        different adcgain/baseline values across segments.

        Target file created with:
        ```
        for i in {01..18}
        do
            rdsamp -r sample-data/multi-segment/s25047/3234460_00$i -P | cut -f 2- >> record-multi-variable-c
        done
        ```

        Entire signal has 543240 samples.
        - 25740 length empty segment.
        - First 16 segments have same 2 channels, length 420000
        - Last 2 segments have same 3 channels, length 97500

        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/s25047/s25047-2704-05-04-10-44",
            smooth_frames=False,
        )

        # convert expanded to uniform array and round to 8 digits
        sig_round = np.zeros((record.sig_len, record.n_sig))
        for i in range(record.n_sig):
            sig_round[:, i] = np.round(record.e_p_signal[i], decimals=8)

        sig_target_a = np.full((25740, 3), np.nan)
        sig_target_b = np.concatenate(
            (
                np.genfromtxt(
                    "tests/target-output/record-multi-variable-c",
                    skip_footer=97500,
                ),
                np.full((420000, 1), np.nan),
            ),
            axis=1,
        )
        sig_target_c = np.genfromtxt(
            "tests/target-output/record-multi-variable-c", skip_header=420000
        )
        sig_target = np.concatenate((sig_target_a, sig_target_b, sig_target_c))

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_variable_d(self):
        """
        Multi-segment, variable layout, selected duration, selected
        channels, digital. There are two channels: PLETH, and II. Their
        fmt, adc_gain, and baseline do not change between the segments.

        Target file created with:
            rdsamp -r sample-data/multi-segment/p000878/p000878-2137-10-26-16-57 -f s3550 -t s7500 -s 0 1 | cut -f 2- | perl -p -e 's/-32768/  -128/g;' > record-multi-variable-d

        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57",
            sampfrom=3550,
            sampto=7500,
            channels=[0, 1],
            physical=False,
        )
        sig = record.d_signal

        # Compare data streaming from Physionet
        record_pn = wfdb.rdrecord(
            "p000878-2137-10-26-16-57",
            pn_dir="mimic3wdb/matched/p00/p000878/",
            sampfrom=3550,
            sampto=7500,
            channels=[0, 1],
            physical=False,
        )
        sig_target = np.genfromtxt(
            "tests/target-output/record-multi-variable-d"
        )

        # Option of selecting channels by name
        record_named = wfdb.rdrecord(
            "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57",
            sampfrom=3550,
            sampto=7500,
            physical=False,
            channel_names=["PLETH", "II"],
        )

        np.testing.assert_equal(sig, sig_target)
        assert record.__eq__(record_pn)
        assert record.__eq__(record_named)


class TestTimeConversion(unittest.TestCase):
    """
    Test cases for time conversion
    """

    def test_single(self):
        """
        Time conversion for a single-segment record

        This checks the get_frame_number, get_elapsed_time, and
        get_absolute_time methods for a Record object.  The example record
        has no base date defined, so attempting to convert to/from absolute
        time should raise an exception.

        """
        header = wfdb.rdheader("sample-data/test01_00s")

        # these time values should be equivalent
        n = 123 * header.fs
        t = datetime.timedelta(seconds=123)
        self.assertEqual(header.get_frame_number(n), n)
        self.assertEqual(header.get_frame_number(t), n)
        self.assertEqual(header.get_elapsed_time(n), t)
        self.assertEqual(header.get_elapsed_time(t), t)

        # record test01_00s has no base date, so absolute time conversions
        # should fail
        self.assertIsNone(header.base_date)
        d = datetime.datetime(2001, 1, 1, 12, 0, 0)
        self.assertRaises(ValueError, header.get_frame_number, d)
        self.assertRaises(ValueError, header.get_absolute_time, n)
        self.assertRaises(ValueError, header.get_absolute_time, t)

    def test_multisegment_with_date(self):
        """
        Time conversion for a multi-segment record with base date

        This checks the get_frame_number, get_elapsed_time, and
        get_absolute_time methods for a MultiRecord object.  The example
        record has a base date, so we can convert timestamps between all
        three of the supported representations.

        """
        header = wfdb.rdheader(
            "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57"
        )

        # these time values should be equivalent
        n = 123 * header.fs
        t = datetime.timedelta(seconds=123)
        d = t + header.base_datetime
        self.assertEqual(header.get_frame_number(n), n)
        self.assertEqual(header.get_frame_number(t), n)
        self.assertEqual(header.get_frame_number(d), n)
        self.assertEqual(header.get_elapsed_time(n), t)
        self.assertEqual(header.get_elapsed_time(t), t)
        self.assertEqual(header.get_elapsed_time(d), t)
        self.assertEqual(header.get_absolute_time(n), d)
        self.assertEqual(header.get_absolute_time(t), d)
        self.assertEqual(header.get_absolute_time(d), d)


class TestSignal(unittest.TestCase):
    """
    For lower level signal tests

    """

    def test_infer_sig_len(self):
        """
        Infer the signal length of a record without the sig_len header
        Read two headers. The records should be the same.
        """

        record = wfdb.rdrecord("sample-data/drive02")
        record_2 = wfdb.rdrecord("sample-data/drive02-no-len")
        record_2.record_name = record.record_name

        assert record_2.__eq__(record)

        record = wfdb.rdrecord("sample-data/a103l")
        record_2 = wfdb.rdrecord("sample-data/a103l-no-len")
        record_2.record_name = record.record_name

        assert record_2.__eq__(record)

    def test_physical_conversion(self):
        n_sig = 3
        adc_gain = [1.0, 1234.567, 765.4321]
        baseline = [10, 20, -30]
        d_signal = np.repeat(np.arange(-100, 100), 3).reshape(-1, 3)
        e_d_signal = list(d_signal.transpose())
        fmt = ["16", "16", "16"]

        # Test adding or subtracting a small offset (0.01 ADU) to check
        # that we correctly round to the nearest integer
        for offset in (0, -0.01, 0.01):
            p_signal = (d_signal + offset - baseline) / adc_gain
            e_p_signal = list(p_signal.transpose())

            # Test converting p_signal to d_signal

            record = wfdb.Record(
                n_sig=n_sig,
                p_signal=p_signal.copy(),
                adc_gain=adc_gain,
                baseline=baseline,
                fmt=fmt,
            )

            d_signal_converted = record.adc(expanded=False, inplace=False)
            np.testing.assert_array_equal(d_signal_converted, d_signal)

            record.adc(expanded=False, inplace=True)
            np.testing.assert_array_equal(record.d_signal, d_signal)

            # Test converting e_p_signal to e_d_signal

            record = wfdb.Record(
                n_sig=n_sig,
                e_p_signal=[s.copy() for s in e_p_signal],
                adc_gain=adc_gain,
                baseline=baseline,
                fmt=fmt,
            )

            e_d_signal_converted = record.adc(expanded=True, inplace=False)
            self.assertEqual(len(e_d_signal_converted), n_sig)
            for x, y in zip(e_d_signal_converted, e_d_signal):
                np.testing.assert_array_equal(x, y)

            record.adc(expanded=True, inplace=True)
            self.assertEqual(len(record.e_d_signal), n_sig)
            for x, y in zip(record.e_d_signal, e_d_signal):
                np.testing.assert_array_equal(x, y)

            # Test automatic conversion using wfdb.wrsamp()

            wfdb.wrsamp(
                "test_physical_conversion",
                fs=1000,
                sig_name=["X", "Y", "Z"],
                units=["mV", "mV", "mV"],
                p_signal=p_signal,
                adc_gain=adc_gain,
                baseline=baseline,
                fmt=["16", "16", "16"],
                write_dir=self.temp_path,
            )
            record = wfdb.rdrecord(
                os.path.join(self.temp_path, "test_physical_conversion"),
                physical=False,
            )
            np.testing.assert_array_equal(record.d_signal, d_signal)

            record = wfdb.rdrecord(
                os.path.join(self.temp_path, "test_physical_conversion"),
                physical=True,
            )
            for ch, gain in enumerate(adc_gain):
                np.testing.assert_allclose(
                    record.p_signal[:, ch],
                    p_signal[:, ch],
                    rtol=0.0000001,
                    atol=(0.05 / gain),
                )

    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_directory.name

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()


class TestDownload(unittest.TestCase):
    # Test that we can download records with no "dat" file
    # Regression test for https://github.com/MIT-LCP/wfdb-python/issues/118
    def test_dl_database_no_dat_file(self):
        wfdb.dl_database("afdb", self.temp_path, ["00735"])

    # Test that we can download records that *do* have a "dat" file.
    def test_dl_database_with_dat_file(self):
        wfdb.dl_database("afdb", self.temp_path, ["04015"])

    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_directory.name

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
    print("Everything passed!")
