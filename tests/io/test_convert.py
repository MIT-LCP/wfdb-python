import os

import numpy as np

from wfdb.io.annotation import rdann
from wfdb.io.convert.csv import csv2ann, csv_to_wfdb
from wfdb.io.convert.edf import read_edf
from wfdb.io.record import rdrecord


class TestEdfToWfdb:
    """Tests for the io.convert.edf module."""

    def test_edf_uniform(self):
        """EDF format conversion to MIT for uniform sample rates."""
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
        """EDF format conversion to MIT for non-uniform sample rates."""
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


class TestCsvToWfdb:
    """Tests for the io.convert.csv module."""

    def test_write_dir(self, tmp_path):
        """Call the function with the write_dir argument."""
        test_dir = str(tmp_path)
        record_100_csv = "sample-data/100.csv"
        # Load 100.dat file for comparison to 100.csv file
        record_100_dat = rdrecord("sample-data/100", physical=True)

        csv_to_wfdb(
            file_name=record_100_csv,
            fs=360,
            units="mV",
            write_dir=test_dir,
        )

        # Check if the output files are created in the specified directory
        base_name = os.path.splitext(os.path.basename(record_100_csv))[0]
        expected_dat_file = os.path.join(test_dir, f"{base_name}.dat")
        expected_hea_file = os.path.join(test_dir, f"{base_name}.hea")

        assert os.path.exists(expected_dat_file)
        assert os.path.exists(expected_hea_file)

        # Check that newly written file matches the 100.dat file
        record_write = rdrecord(os.path.join(test_dir, base_name))

        assert record_write.fs == 360
        assert record_write.fs == record_100_dat.fs
        assert record_write.units == ["mV", "mV"]
        assert record_write.units == record_100_dat.units
        assert record_write.sig_name == ["MLII", "V5"]
        assert record_write.sig_name == record_100_dat.sig_name
        assert record_write.p_signal.size == 1300000
        assert record_write.p_signal.size == record_100_dat.p_signal.size


class TestCsv2Ann:
    """Tests for csv2ann."""

    def test_time_onsets(self, tmp_path):
        """Time onsets are converted to samples and written to write_dir."""
        csv_file = tmp_path / "in" / "example.csv"
        csv_file.parent.mkdir()
        csv_file.write_text("onset,description\n0.2,p-wave\n0.8,qrs\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        csv2ann(str(csv_file), fs=360, write_dir=str(out_dir))

        ann = rdann(str(out_dir / "example"), "atr")
        assert ann.sample.tolist() == [72, 288]
        assert ann.symbol == ['"', '"']
        assert ann.aux_note == ["p-wave", "qrs"]

    def test_sample_onsets_no_header(self, tmp_path):
        """Sample onsets are used as-is when time_onset is False."""
        csv_file = tmp_path / "example.csv"
        csv_file.write_text("5,p-wave\n13,qrs\n")

        ann = csv2ann(
            str(csv_file), fs=10, time_onset=False, header=False, record_only=True
        )

        assert ann.sample.tolist() == [5, 13]
        assert ann.aux_note == ["p-wave", "qrs"]

    def test_record_name_with_dots(self, tmp_path, monkeypatch):
        """Dots in the directory or a ./ prefix don't truncate the name (#595)."""
        data_dir = tmp_path / "data" / "v1.2"
        data_dir.mkdir(parents=True)
        (data_dir / "x.csv").write_text("onset,description\n0.5,qrs\n")
        monkeypatch.chdir(data_dir)

        ann = csv2ann("./x.csv", fs=100, record_only=True)
        assert ann.record_name == "x"

        ann = csv2ann(str(data_dir / "x.csv"), fs=100, record_only=True)
        assert ann.record_name == "x"

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        csv2ann(str(data_dir / "x.csv"), fs=100, write_dir=str(out_dir))
        assert (out_dir / "x.atr").exists()
