import os
import wave

import numpy as np
import pytest

from wfdb.io.annotation import rdann
from wfdb.io.convert import wfdb_to_wav
from wfdb.io.convert.csv import csv2ann, csv_to_wfdb
from wfdb.io.convert.edf import read_edf
from wfdb.io.record import rdheader, rdrecord


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

    @staticmethod
    def _write_csv(path):
        path.write_text("I,II\n0.1,-0.25\n0.5,0.75\n-1.0,0.0\n")
        return str(path)

    def test_record_matches_written_header(self, tmp_path):
        """record_only returns the same fields that are written (#593)."""
        csv_file = self._write_csv(tmp_path / "rec.csv")

        record = csv_to_wfdb(csv_file, fs=100, units="mV", record_only=True)
        csv_to_wfdb(csv_file, fs=100, units="mV", write_dir=str(tmp_path))
        header = rdheader(str(tmp_path / "rec"))

        # Initial values and checksums of the digital samples (gain 200)
        assert record.init_value == [20, -50]
        assert record.checksum == [(20 + 100 - 200) % 65536, 100]
        for field in ["fmt", "adc_gain", "baseline", "adc_res", "adc_zero"]:
            assert getattr(record, field) == getattr(header, field), field
        assert record.init_value == header.init_value
        assert record.checksum == header.checksum

    def test_scalars_broadcast(self, tmp_path):
        """Floats and NumPy scalars are used for all channels (#593)."""
        csv_file = self._write_csv(tmp_path / "rec.csv")

        record = csv_to_wfdb(
            csv_file,
            fs=100,
            units="mV",
            adc_gain=200.5,
            baseline=np.int64(3),
            record_only=True,
        )

        assert record.adc_gain == [200.5, 200.5]
        assert record.baseline == [3, 3]

    def test_zero_and_list_defaults(self, tmp_path):
        """An explicit 0 is kept, and baseline defaults to adc_zero (#593)."""
        csv_file = self._write_csv(tmp_path / "rec.csv")

        record = csv_to_wfdb(
            csv_file, fs=100, units="mV", baseline=0, adc_zero=1024, record_only=True
        )
        assert record.baseline == [0, 0]
        assert record.adc_zero == [1024, 1024]

        record = csv_to_wfdb(
            csv_file, fs=100, units="mV", adc_zero=[1, 2], record_only=True
        )
        assert record.baseline == [1, 2]

    def test_parameters_are_written(self, tmp_path):
        """Header parameters reach the written files (#592)."""
        csv_file = self._write_csv(tmp_path / "rec.csv")

        csv_to_wfdb(
            csv_file,
            fs=100,
            units="mV",
            counter_freq=10,
            base_counter=5,
            dat_file_name="signals.dat",
            skew=[0, 1],
            byte_offset=4,
            adc_res=15,
            adc_zero=[3, 4],
            block_size=0,
            write_dir=str(tmp_path),
        )
        header = rdheader(str(tmp_path / "rec"))

        assert header.counter_freq == 10
        assert header.base_counter == 5
        assert header.file_name == ["signals.dat", "signals.dat"]
        assert header.skew == [0, 1]
        assert header.byte_offset == [4, 4]
        assert header.adc_res == [15, 15]
        assert header.adc_zero == [3, 4]
        assert header.baseline == [3, 4]
        assert (tmp_path / "signals.dat").stat().st_size == 4 + 3 * 2 * 2

    @pytest.mark.parametrize(
        "kwargs",
        [{"checksum": [1, 2]}, {"init_value": 0}, {"samps_per_frame": 2}],
    )
    def test_deprecated_arguments(self, tmp_path, kwargs):
        """Arguments that follow from the samples are ignored with a warning."""
        csv_file = self._write_csv(tmp_path / "rec.csv")

        with pytest.warns(FutureWarning):
            record = csv_to_wfdb(
                csv_file, fs=100, units="mV", record_only=True, **kwargs
            )

        assert record.checksum == [(20 + 100 - 200) % 65536, 100]
        assert record.init_value == [20, -50]
        assert record.samps_per_frame is None

    def test_bool_rejected(self, tmp_path):
        """A bool is not accepted as a numeric header value."""
        csv_file = self._write_csv(tmp_path / "rec.csv")

        with pytest.raises(TypeError, match="adc_gain"):
            csv_to_wfdb(csv_file, fs=100, units="mV", adc_gain=True)

    def test_missing_values(self, tmp_path):
        """Empty CSV fields are written as invalid samples and read back as NaN."""
        csv_file = tmp_path / "rec.csv"
        csv_file.write_text("I,II\n0.1,-0.25\n,0.75\n-1.0,0.0\n")

        csv_to_wfdb(str(csv_file), fs=100, units="mV", write_dir=str(tmp_path))
        record = rdrecord(str(tmp_path / "rec"))

        expected = np.array([[0.1, -0.25], [np.nan, 0.75], [-1.0, 0.0]])
        np.testing.assert_allclose(record.p_signal, expected)
        assert record.init_value == [20, -50]

    def test_no_rows(self, tmp_path):
        """A CSV with a header but no samples gives a clear error."""
        csv_file = tmp_path / "rec.csv"
        csv_file.write_text("I,II\n")

        with pytest.raises(ValueError, match="no samples"):
            csv_to_wfdb(str(csv_file), fs=100, units="mV", write_dir=str(tmp_path))


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


class TestWfdbToWav:
    """Tests for wfdb_to_wav."""

    # 11-bit samples, and a signal with two samples per frame
    @pytest.mark.parametrize("record_name", ["100", "test01_00s_frame"])
    def test_samples(self, tmp_path, record_name):
        """One 16-bit sample per frame is written for each signal (#410)."""
        wav_file = str(tmp_path / "out.wav")
        wfdb_to_wav(f"sample-data/{record_name}", output_filename=wav_file)

        # Samples are offset by the ADC zero and shifted to fill 16 bits
        record = rdrecord(f"sample-data/{record_name}", physical=False)
        shift = np.array([16 - res for res in record.adc_res])
        expected = (record.d_signal - record.adc_zero) << shift

        with wave.open(wav_file, "rb") as wav:
            assert wav.getnchannels() == record.n_sig
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == record.fs
            assert wav.getnframes() == record.sig_len
            frames = wav.readframes(record.sig_len)
        samples = np.frombuffer(frames, dtype="<i2").reshape(expected.shape)
        np.testing.assert_array_equal(samples, expected)

    @pytest.mark.parametrize("record_name", ["100", "test01_00s_frame"])
    def test_write_header(self, tmp_path, monkeypatch, record_name):
        """The header written with write_header=True describes the .wav file."""
        record_path = os.path.abspath(f"sample-data/{record_name}")
        monkeypatch.chdir(tmp_path)
        wfdb_to_wav(record_path, write_header=True)

        # Read back as one sample per frame, with the same physical values
        record = rdrecord(record_name)
        expected = rdrecord(record_path)
        assert record.samps_per_frame == [1] * expected.n_sig
        np.testing.assert_array_equal(record.p_signal, expected.p_signal)
