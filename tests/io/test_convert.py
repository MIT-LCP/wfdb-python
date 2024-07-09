import os
import shutil
import unittest

import numpy as np

from wfdb.io.record import rdrecord
from wfdb.io.convert.edf import read_edf
from wfdb.io.convert.csv import csv_to_wfdb


class TestEdfToWfdb:
    """
    Tests for the io.convert.edf module.
    """

    def test_edf_uniform(self):
        """
        EDF format conversion to MIT for uniform sample rates.
        """
        # Uniform sample rates
        record_MIT = rdrecord("sample-data/n16").__dict__
        record_EDF = read_edf("sample-data/n16.edf").__dict__

        fields = list(record_MIT.keys())
        # Original MIT format method of checksum is outdated, sometimes
        # the same value though
        fields.remove("checksum")
        # Original MIT format units are less comprehensive since they
        # default to mV if unknown.. therefore added more default labels
        fields.remove("units")

        test_results = []
        for field in fields:
            # Signal value will be slightly off due to C to Python type conversion
            if field == "p_signal":
                true_array = np.array(record_MIT[field]).flatten()
                pred_array = np.array(record_EDF[field]).flatten()
                # Prevent divide by zero warning
                for i, v in enumerate(true_array):
                    if v == 0:
                        true_array[i] = 1
                        pred_array[i] = 1
                sig_diff = np.abs((pred_array - true_array) / true_array)
                sig_diff[sig_diff == -np.inf] = 0
                sig_diff[sig_diff == np.inf] = 0
                sig_diff = np.nanmean(sig_diff, 0)
                # 5% tolerance
                if np.max(sig_diff) <= 5:
                    test_results.append(True)
                else:
                    test_results.append(False)
            elif field == "init_value":
                signal_diff = [
                    abs(record_MIT[field][i] - record_EDF[field][i])
                    for i in range(len(record_MIT[field]))
                ]
                if abs(max(min(signal_diff), max(signal_diff), key=abs)) <= 2:
                    test_results.append(True)
                else:
                    test_results.append(False)
            else:
                test_results.append(record_MIT[field] == record_MIT[field])

        target_results = len(fields) * [True]
        assert np.array_equal(test_results, target_results)

    def test_edf_non_uniform(self):
        """
        EDF format conversion to MIT for non-uniform sample rates.
        """
        # Non-uniform sample rates
        record_MIT = rdrecord("sample-data/wave_4").__dict__
        record_EDF = read_edf("sample-data/wave_4.edf").__dict__

        fields = list(record_MIT.keys())
        # Original MIT format method of checksum is outdated, sometimes
        # the same value though
        fields.remove("checksum")
        # Original MIT format units are less comprehensive since they
        # default to mV if unknown.. therefore added more default labels
        fields.remove("units")

        test_results = []
        for field in fields:
            # Signal value will be slightly off due to C to Python type conversion
            if field == "p_signal":
                true_array = np.array(record_MIT[field]).flatten()
                pred_array = np.array(record_EDF[field]).flatten()
                # Prevent divide by zero warning
                for i, v in enumerate(true_array):
                    if v == 0:
                        true_array[i] = 1
                        pred_array[i] = 1
                sig_diff = np.abs((pred_array - true_array) / true_array)
                sig_diff[sig_diff == -np.inf] = 0
                sig_diff[sig_diff == np.inf] = 0
                sig_diff = np.nanmean(sig_diff, 0)
                # 5% tolerance
                if np.max(sig_diff) <= 5:
                    test_results.append(True)
                else:
                    test_results.append(False)
            elif field == "init_value":
                signal_diff = [
                    abs(record_MIT[field][i] - record_EDF[field][i])
                    for i in range(len(record_MIT[field]))
                ]
                if abs(max(min(signal_diff), max(signal_diff), key=abs)) <= 2:
                    test_results.append(True)
                else:
                    test_results.append(False)
            else:
                test_results.append(record_MIT[field] == record_MIT[field])

        target_results = len(fields) * [True]
        assert np.array_equal(test_results, target_results)


class TestCsvToWfdb(unittest.TestCase):
    """
    Tests for the io.convert.csv module.
    """

    def setUp(self):
        """
        Create a temporary directory containing data for testing.

        Load 100.dat file for comparison to 100.csv file.
        """
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)

        self.record_100_csv = "sample-data/100.csv"
        self.record_100_dat = rdrecord("sample-data/100", physical=True)

    def tearDown(self):
        """
        Remove the temporary directory after the test.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_write_dir(self):
        """
        Call the function with the write_dir argument.
        """
        csv_to_wfdb(
            file_name=self.record_100_csv,
            fs=360,
            units="mV",
            write_dir=self.test_dir,
        )

        # Check if the output files are created in the specified directory
        base_name = os.path.splitext(os.path.basename(self.record_100_csv))[0]
        expected_dat_file = os.path.join(self.test_dir, f"{base_name}.dat")
        expected_hea_file = os.path.join(self.test_dir, f"{base_name}.hea")

        self.assertTrue(os.path.exists(expected_dat_file))
        self.assertTrue(os.path.exists(expected_hea_file))

        # Check that newly written file matches the 100.dat file
        record_write = rdrecord(os.path.join(self.test_dir, base_name))

        self.assertEqual(record_write.fs, 360)
        self.assertEqual(record_write.fs, self.record_100_dat.fs)
        self.assertEqual(record_write.units, ["mV", "mV"])
        self.assertEqual(record_write.units, self.record_100_dat.units)
        self.assertEqual(record_write.sig_name, ["MLII", "V5"])
        self.assertEqual(record_write.sig_name, self.record_100_dat.sig_name)
        self.assertEqual(record_write.p_signal.size, 1300000)
        self.assertEqual(
            record_write.p_signal.size, self.record_100_dat.p_signal.size
        )


if __name__ == "__main__":
    unittest.main()
