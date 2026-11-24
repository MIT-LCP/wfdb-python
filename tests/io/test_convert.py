import os
import shutil
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import loadmat

from wfdb.io.annotation import rdann
from wfdb.io.convert import wfdb_to_wav
from wfdb.io.convert.csv import csv2ann, csv_to_wfdb
from wfdb.io.convert.edf import rdedfann, read_edf, wfdb_to_edf
from wfdb.io.convert.matlab import wfdb_to_mat
from wfdb.io.record import Record, rdheader, rdrecord, wrsamp
from wfdb.plot import plot_wfdb


class TestEdfToWfdb:
    """Tests for the io.convert.edf module."""

    def test_rdedfann(self):
        """Read EDF+ annotations without passing obsolete read_edf options."""
        info = rdedfann("sample-data/test_edfann.edf")
        assert info["comment"][:2] == ["Sleep stage W", "Sleep stage W"]
        assert all(sample is None for sample in info["sample_num"])

    def test_rdedfann_parses_signed_tals_and_multiple_annotations(self, tmp_path):
        """TAL byte delimiters separate events without splitting annotation text."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("512", 8),
            ("EDF+C", 44),
            ("1", 8),
            ("1", 8),
            ("1", 4),
        ]
        signal_fields = [
            ("EDF Annotations", 16),
            ("", 80),
            ("", 8),
            ("-1", 8),
            ("1", 8),
            ("-32768", 8),
            ("32767", 8),
            ("", 80),
            ("128", 8),
            ("", 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        header += "".join(value.ljust(width) for value, width in signal_fields)
        tal_bytes = (
            b"+0\x14\x14\x00"
            b"-0.25\x14Pre+event\x14Other event\x14\x00"
            b"+0.5\x15.2\x14Duration+event\x14\x00"
            b"+0.75\x14No duration\x14\x00"
        )
        edf_file = tmp_path / "signed_tals.edf"
        edf_file.write_bytes(header.encode() + tal_bytes.ljust(256, b"\x00"))

        info = rdedfann(str(edf_file))
        assert info["onset_time"] == [
            "-00:00:00.250",
            "-00:00:00.250",
            "00:00:00.500",
            "00:00:00.750",
        ]
        assert info["comment"] == [
            "Pre+event",
            "Other event",
            "Duration+event",
            "No duration",
        ]
        assert info["duration"] == [0.0, 0.0, 0.2, 0.0]

    @pytest.mark.parametrize("onset, expected_sample", [(1.3, 42), (1.265625, 40)])
    def test_rdedfann_sample_num_uses_fs(self, tmp_path, onset, expected_sample):
        """Both return modes round onset seconds to the nearest sample."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("768", 8),
            ("EDF+C", 44),
            ("2", 8),
            ("1", 8),
            ("2", 4),
        ]
        signal_fields = [
            (["ECG", "EDF Annotations"], 16),
            (["", ""], 80),
            (["mV", ""], 8),
            (["-32767", "-1"], 8),
            (["32767", "1"], 8),
            (["-32767", "-32768"], 8),
            (["32767", "32767"], 8),
            (["", ""], 80),
            (["32", "32"], 8),
            (["", ""], 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        for values, width in signal_fields:
            header += "".join(value.ljust(width) for value in values)
        ecg_block = np.zeros(32, dtype="<i2").tobytes()
        first_block = ecg_block + b"+0\x14\x14\x00".ljust(64, b"\x00")
        second_block = ecg_block + (
            b"+1\x14\x14\x00" + f"+{onset}\x15.5\x14Event\x14\x00".encode()
        ).ljust(64, b"\x00")
        edf_file = tmp_path / "onset.edf"
        edf_file.write_bytes(header.encode() + first_block + second_block)

        metadata = read_edf(str(edf_file), header_only=True, rdedfann_flag=True)
        assert metadata["fs"] == 32
        assert metadata["sig_len"] == 64
        info = rdedfann(str(edf_file))
        assert info["sample_num"] == [expected_sample]
        assert info["comment"] == ["Event"]
        annotation = rdedfann(str(edf_file), info_only=False, record_only=True)
        assert annotation.sample[0] == expected_sample

    def test_rdedfann_annotation_only_requires_fs(self):
        """Annotation-only records have no implicit sample rate."""
        source = "sample-data/test_edfann.edf"
        metadata = read_edf(source, header_only=True, rdedfann_flag=True)
        assert metadata["fs"] is None
        assert metadata["sig_len"] == 0
        assert metadata["n_sig"] == 0
        with pytest.raises(ValueError, match="no signal channels"):
            read_edf(source)
        info = rdedfann(source)
        assert info["fs"] is None
        assert info["sample_num"] and all(
            sample is None for sample in info["sample_num"]
        )
        with pytest.raises(ValueError, match="fs is required"):
            rdedfann(source, info_only=False, record_only=True)
        with_rate = rdedfann(source, fs=32)
        assert with_rate["fs"] == 32
        assert len(with_rate["sample_num"]) == len(info["sample_num"])
        assert all(isinstance(sample, int) for sample in with_rate["sample_num"])
        annotation = rdedfann(source, info_only=False, record_only=True, fs=32)
        assert annotation.fs == 32

    @pytest.mark.parametrize(
        "signal_rate, annotation_capacity, expected_samples",
        [
            (100, 60, [37, 150]),
            (200, 30, [74, 300]),
            (256, 30, [95, 384]),
        ],
    )
    def test_edf_annotations_do_not_set_signal_rate(
        self, tmp_path, signal_rate, annotation_capacity, expected_samples
    ):
        """Annotation text capacity does not downsample physiologic signals."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("768", 8),
            ("EDF+C", 44),
            ("2", 8),
            ("1", 8),
            ("2", 4),
        ]
        signal_fields = [
            (["ECG", "EDF Annotations"], 16),
            (["", ""], 80),
            (["mV", ""], 8),
            (["-32767", "-1"], 8),
            (["32767", "1"], 8),
            (["-32767", "-32768"], 8),
            (["32767", "32767"], 8),
            (["", ""], 80),
            ([str(signal_rate), str(annotation_capacity)], 8),
            (["", ""], 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        for values, width in signal_fields:
            header += "".join(value.ljust(width) for value in values)
        samples = np.arange(2 * signal_rate, dtype="<i2")
        first_tals = b"+0\x14\x14\x00+0.37\x15.0\x14First\x14\x00"
        second_tals = b"+1\x14\x14\x00+1.5\x15.0\x14Second\x14\x00"
        block_bytes = 2 * annotation_capacity
        data = (
            samples[:signal_rate].tobytes()
            + first_tals.ljust(block_bytes, b"\x00")
            + samples[signal_rate:].tobytes()
            + second_tals.ljust(block_bytes, b"\x00")
        )
        edf_file = tmp_path / "ecg_with_annotations.edf"
        edf_file.write_bytes(header.encode() + data)

        metadata = read_edf(str(edf_file), header_only=True)
        assert metadata["fs"] == signal_rate
        assert metadata["sig_len"] == 2 * signal_rate
        assert metadata["n_sig"] == 1
        assert metadata["sig_name"] == ["ECG"]
        record = read_edf(str(edf_file))
        assert record.fs == signal_rate
        assert record.samps_per_frame == [1]
        assert record.sig_len == 2 * signal_rate
        assert record.sig_name == ["ECG"]
        np.testing.assert_array_equal(record.p_signal[:, 0], samples)
        info = rdedfann(str(edf_file))
        assert info["fs"] == signal_rate
        assert info["sample_num"] == expected_samples
        assert info["comment"] == ["First", "Second"]
        annotation = rdedfann(str(edf_file), info_only=False, record_only=True)
        np.testing.assert_array_equal(annotation.sample, expected_samples)

    def test_rdedfann_uses_fastest_signal_rate_for_mixed_rates(self, tmp_path):
        """WFDB annotations retain onset timing across mixed signal rates."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("1024", 8),
            ("EDF+C", 44),
            ("2", 8),
            ("1", 8),
            ("3", 4),
        ]
        signal_fields = [
            (["ECG", "EEG", "EDF Annotations"], 16),
            (["", "", ""], 80),
            (["mV", "mV", ""], 8),
            (["-32767", "-32767", "-1"], 8),
            (["32767", "32767", "1"], 8),
            (["-32767", "-32767", "-32768"], 8),
            (["32767", "32767", "32767"], 8),
            (["", "", ""], 80),
            (["256", "100", "30"], 8),
            (["", "", ""], 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        for values, width in signal_fields:
            header += "".join(value.ljust(width) for value in values)
        ecg = np.arange(512, dtype="<i2")
        eeg = np.arange(200, dtype="<i2")
        first_tals = b"+0\x14\x14\x00+0.37\x14First\x14\x00"
        second_tals = b"+1\x14\x14\x00+1.5\x14Second\x14\x00"
        data = (
            ecg[:256].tobytes()
            + eeg[:100].tobytes()
            + first_tals.ljust(60, b"\x00")
            + ecg[256:].tobytes()
            + eeg[100:].tobytes()
            + second_tals.ljust(60, b"\x00")
        )
        edf_file = tmp_path / "mixed_rates.edf"
        edf_file.write_bytes(header.encode() + data)

        record = read_edf(str(edf_file), rdedfann_flag=True)
        assert record.fs == 4
        assert record.samps_per_frame == [64, 25]
        info = rdedfann(str(edf_file))
        assert info["fs"] == 256
        assert info["sample_num"] == [95, 384]
        annotation = rdedfann(str(edf_file), info_only=False, record_only=True)
        assert annotation.fs == 256
        np.testing.assert_array_equal(annotation.sample, [95, 384])
        annotation.wrann(write_fs=True, write_dir=str(tmp_path))
        written = rdann(str(tmp_path / "mixed_rates"), "atr")
        assert written.fs == 256
        np.testing.assert_array_equal(written.sample, [95, 384])

        record.wrsamp(write_dir=str(tmp_path))
        written_record = rdrecord(str(tmp_path / "mixed_rates"))
        assert written_record.fs == 4
        fig = plot_wfdb(written_record, written, time_units="samples", return_fig=True)
        plt.close(fig)
        annotations = fig.axes[0].lines[1]
        np.testing.assert_allclose(annotations.get_xdata(), [95 / 64, 6])
        np.testing.assert_allclose(
            annotations.get_ydata(), written_record.p_signal[[1, 6], 0]
        )
        assert annotations.get_xdata()[-1] < written_record.sig_len

        fig = plot_wfdb(written_record, written, time_units="seconds", return_fig=True)
        plt.close(fig)
        np.testing.assert_allclose(fig.axes[0].lines[1].get_xdata(), [95 / 256, 1.5])

        overridden = rdedfann(str(edf_file), fs=1000)
        assert overridden["fs"] == 1000
        assert overridden["sample_num"] == [370, 1500]

    def test_rdedfann_ignores_physical_calibration(self, tmp_path):
        """Annotation bytes have the same meaning with different EDF scales."""
        source = "sample-data/test_edfann.edf"
        expected = rdedfann(source)
        data = bytearray(Path(source).read_bytes())
        n_sig = int(data[252:256])
        labels = [
            data[256 + 16 * i : 256 + 16 * (i + 1)].decode().strip()
            for i in range(n_sig)
        ]
        channel = labels.index("EDF Annotations")
        physical_min = 256 + n_sig * (16 + 80 + 8) + 8 * channel
        physical_max = physical_min + 8 * n_sig
        data[physical_min : physical_min + 8] = b"-1".ljust(8, b" ")
        data[physical_max : physical_max + 8] = b"1".ljust(8, b" ")
        altered = tmp_path / "scaled_ann.edf"
        altered.write_bytes(data)

        assert rdedfann(str(altered)) == expected

    @pytest.mark.parametrize(
        "comment, wire_encoding, override",
        [
            ("Event", "utf-8", None),
            ("Schlaf ä – Wach", "utf-8", None),
            ("Schlaf ä", "iso8859-1", "iso8859-1"),
        ],
    )
    def test_rdedfann_with_mixed_rate_signal(
        self, tmp_path, comment, wire_encoding, override
    ):
        """Decode annotation bytes without averaging or garbling UTF-8 text."""
        n_sig = 2
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            (str(256 * (n_sig + 1)), 8),
            ("EDF+C", 44),
            ("1", 8),
            ("1", 8),
            (str(n_sig), 4),
        ]
        signal_fields = [
            (["ECG", "EDF Annotations"], 16),
            (["", ""], 80),
            (["mV", ""], 8),
            (["-1", "-1"], 8),
            (["1", "1"], 8),
            (["-32768", "-32768"], 8),
            (["32767", "32767"], 8),
            (["", ""], 80),
            (["4", "64"], 8),
            (["", ""], 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        for values, width in signal_fields:
            header += "".join(value.ljust(width) for value in values)
        annotation = f"+0\x14\x14\x00+1\x15.5\x14{comment}\x14\x00".encode(
            wire_encoding
        )
        samples = np.array([1, 2, 3, 4], dtype="<i2").tobytes()
        edf_file = tmp_path / "mixed_rate_ann.edf"
        edf_file.write_bytes(header.encode() + samples + annotation.ljust(128, b"\x00"))

        info = rdedfann(str(edf_file), encoding=override)
        assert info["comment"] == [comment]
        assert info["duration"] == [0.5]

    def test_rdedfann_keeps_local_file(self, tmp_path):
        """The delete_file option does not remove a local EDF file."""
        source = "sample-data/test_edfann.edf"
        local_file = tmp_path / "test_edfann.edf"
        shutil.copyfile(source, local_file)
        rdedfann(str(local_file), delete_file=True)
        assert local_file.exists()

    def test_rdedfann_deletes_download(self, tmp_path, monkeypatch):
        """Delete a downloaded EDF file when requested."""
        local_file = tmp_path / "test_edfann.edf"
        shutil.copyfile("sample-data/test_edfann.edf", local_file)

        def read_local_file(record_name, **kwargs):
            kwargs.pop("pn_dir")
            return read_edf(record_name, **kwargs)

        monkeypatch.setattr("wfdb.io.convert.edf.read_edf", read_local_file)
        rdedfann(str(local_file), pn_dir="example")
        assert not local_file.exists()

    def test_edf_uniform(self):
        """EDF format conversion to MIT for uniform sample rates."""
        # Uniform sample rates
        record_MIT = rdrecord("sample-data/n16").__dict__
        converted = read_edf("sample-data/n16.edf")
        record_EDF = converted.__dict__
        assert record_EDF["adc_res"] == [16] * 5
        assert record_EDF["checksum"][0] == 56364
        np.testing.assert_equal(
            record_EDF["checksum"],
            [int(channel.sum() % 65536) for channel in converted.adc().T],
        )

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
        # Inverted EDF channels use a reflected WFDB digital scale.
        fields.remove("init_value")
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

    def test_float_checksum(self, tmp_path):
        """Checksums sum converted digital samples rather than physical values."""
        # Two channels of ten digital samples, each with physical value 0.2.
        n_sig = 2
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            (str(256 * (n_sig + 1)), 8),
            ("", 44),
            ("1", 8),
            ("1", 8),
            (str(n_sig), 4),
        ]
        signal_fields = [
            ("x", 16),
            ("", 80),
            ("mV", 8),
            ("-3276.8", 8),
            ("3276.7", 8),
            ("-32768", 8),
            ("32767", 8),
            ("", 80),
            ("10", 8),
            ("", 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        for value, width in signal_fields:
            header += value.ljust(width) * n_sig
        edf_file = tmp_path / "float.edf"
        samples = np.full(10 * n_sig, 2, "<i2")
        edf_file.write_bytes(header.encode() + samples.tobytes())

        record = read_edf(str(edf_file))
        np.testing.assert_allclose(record.p_signal, 0.2)
        assert record.checksum == [20, 20]
        np.testing.assert_array_equal(record.adc(), 2)

    def test_edf_mixed_rate_scaling(self, tmp_path):
        """Average high-rate samples before applying the physical scale."""
        n_sig = 2
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            (str(256 * (n_sig + 1)), 8),
            ("", 44),
            ("2", 8),
            ("1", 8),
            (str(n_sig), 4),
        ]
        signal_fields = [
            (["fast", "slow"], 16),
            (["", ""], 80),
            (["mV", "mV"], 8),
            (["0", "0"], 8),
            (["50", "50"], 8),
            (["0", "0"], 8),
            (["100", "100"], 8),
            (["", ""], 80),
            (["2", "1"], 8),
            (["", ""], 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        for values, width in signal_fields:
            header += "".join(value.ljust(width) for value in values)
        edf_file = tmp_path / "mixed.edf"
        samples = np.array([2, 4, 10, 6, 8, 20], dtype="<i2")
        edf_file.write_bytes(header.encode() + samples.tobytes())

        record = read_edf(str(edf_file))
        assert record.samps_per_frame == [2, 1]
        np.testing.assert_allclose(record.p_signal, [[1.5, 5], [3.5, 10]])
        np.testing.assert_array_equal(record.adc(), [[3, 10], [7, 20]])
        assert record.checksum == [10, 30]

    def test_edf_fractional_baseline(self, tmp_path):
        """EDF physical extrema map exactly, even with a fractional offset."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("512", 8),
            ("", 44),
            ("1", 8),
            ("1", 8),
            ("1", 4),
        ]
        signal_fields = [
            ("ECG", 16),
            ("", 80),
            ("mV", 8),
            ("-100", 8),
            ("100", 8),
            ("-32768", 8),
            ("32767", 8),
            ("", 80),
            ("2", 8),
            ("", 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        header += "".join(value.ljust(width) for value, width in signal_fields)
        edf_file = tmp_path / "fractional_baseline.edf"
        edf_file.write_bytes(
            header.encode() + np.array([-32768, 32767], dtype="<i2").tobytes()
        )

        record = read_edf(str(edf_file))
        np.testing.assert_allclose(record.p_signal[:, 0], [-100, 100])
        assert record.fmt == ["32"]
        record.wrsamp(write_dir=str(tmp_path))
        np.testing.assert_allclose(
            rdrecord(str(tmp_path / "fractional_baseline")).p_signal,
            record.p_signal,
            atol=1 / record.adc_gain[0],
            rtol=0,
        )

    @pytest.mark.parametrize(
        "samples, expected_fmt, expected_digital",
        [
            ([-32768, 32767], "32", [32767, -32768]),
            ([0, 1], "16", [-1, -2]),
        ],
    )
    def test_edf_inverted_range_can_be_written(
        self, tmp_path, samples, expected_fmt, expected_digital
    ):
        """Reflect inverted EDF samples without changing physical values."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("512", 8),
            ("", 44),
            ("1", 8),
            ("1", 8),
            ("1", 4),
        ]
        signal_fields = [
            ("temperature", 16),
            ("", 80),
            ("degreeC", 8),
            ("1", 8),
            ("0", 8),
            ("-32768", 8),
            ("32767", 8),
            ("", 80),
            ("2", 8),
            ("", 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        header += "".join(value.ljust(width) for value, width in signal_fields)
        edf_file = tmp_path / "inverted.edf"
        edf_file.write_bytes(header.encode() + np.array(samples, dtype="<i2").tobytes())

        record = read_edf(str(edf_file))
        assert record.adc_gain == [65535.0]
        assert record.fmt == [expected_fmt]
        assert record.baseline == [-32768]
        np.testing.assert_allclose(
            record.p_signal[:, 0], [(32767 - sample) / 65535 for sample in samples]
        )
        np.testing.assert_array_equal(record.adc()[:, 0], expected_digital)
        assert record.checksum == [sum(expected_digital) % 65536]

        record.wrsamp(write_dir=str(tmp_path))
        np.testing.assert_allclose(
            rdrecord(str(tmp_path / "inverted")).p_signal,
            record.p_signal,
            atol=1 / record.adc_gain[0],
            rtol=0,
        )

    def test_wave_4_inverted_channel_can_be_written(self, tmp_path):
        """Write the mixed-rate example with its inverted temperature scale."""
        record = read_edf("sample-data/wave_4.edf")
        channel = record.sig_name.index("DEV_Temperature")
        assert record.adc_gain[channel] > 0
        original = record.p_signal[:100].copy()
        record.p_signal = record.p_signal[:100]
        record.sig_len = 100

        record.wrsamp(write_dir=str(tmp_path))
        written = rdrecord(str(tmp_path / "wave_4"))
        tolerance = max(1 / gain for gain in record.adc_gain)
        np.testing.assert_allclose(written.p_signal, original, atol=tolerance, rtol=0)

    @pytest.mark.parametrize(
        "digital_min, digital_max, expected_res",
        [(-32768, 32767, 16), (-2048, 2047, 12)],
    )
    def test_edf_adc_res(self, tmp_path, digital_min, digital_max, expected_res):
        """The inclusive EDF digital range determines ADC resolution."""
        record_fields = [
            ("0", 8),
            ("", 80),
            ("", 80),
            ("01.01.00", 8),
            ("00.00.00", 8),
            ("512", 8),
            ("", 44),
            ("1", 8),
            ("1", 8),
            ("1", 4),
        ]
        signal_fields = [
            ("ECG", 16),
            ("", 80),
            ("mV", 8),
            ("-1", 8),
            ("1", 8),
            (str(digital_min), 8),
            (str(digital_max), 8),
            ("", 80),
            ("2", 8),
            ("", 32),
        ]
        header = "".join(value.ljust(width) for value, width in record_fields)
        header += "".join(value.ljust(width) for value, width in signal_fields)
        samples = np.array([digital_min, digital_max], dtype="<i2")
        edf_file = tmp_path / "resolution.edf"
        edf_file.write_bytes(header.encode() + samples.tobytes())

        record = read_edf(str(edf_file))
        assert record.adc_res == [expected_res]
        np.testing.assert_allclose(record.p_signal[:, 0], [-1, 1])

    def test_edf_round_trip_has_no_digital_offset(self, tmp_path):
        """Reading a converted WFDB record does not shift every sample."""
        output = tmp_path / "100.edf"
        wfdb_to_edf("sample-data/100", sampto=3600, output_filename=str(output))

        original = rdrecord("sample-data/100", sampto=3600)
        converted = read_edf(str(output))
        assert converted.fs == 360
        assert isinstance(converted.fs, int)
        assert converted.sig_len == original.sig_len
        np.testing.assert_allclose(converted.p_signal, original.p_signal, atol=1e-10)

    def test_read_edf_record_can_be_written(self, tmp_path):
        """The baseline of a converted record satisfies WFDB field checks."""
        edf_path = tmp_path / "100.edf"
        wfdb_to_edf(
            "sample-data/100", sampto=100, channels=[0], output_filename=str(edf_path)
        )

        record = read_edf(str(edf_path))
        assert isinstance(record.baseline, list)
        record.wrsamp(write_dir=str(tmp_path))

        written = rdrecord(str(tmp_path / "100"))
        np.testing.assert_allclose(written.p_signal[:100], record.p_signal[:100])

    def test_edf_mixed_rate_uses_consistent_frame_count(self, tmp_path):
        """Read a converted file with fractional fs and varied sample rates."""
        output = tmp_path / "drive02.edf"
        wfdb_to_edf("sample-data/drive02", sampto=1000, output_filename=str(output))

        record = read_edf(str(output))
        assert record.fs == 15.5
        assert isinstance(record.fs, float)
        assert record.samps_per_frame == [32, 2, 1, 1, 2]
        assert record.sig_len == 1085
        assert record.p_signal.shape == (1085, 5)
        header = read_edf(str(output), header_only=True)
        assert header["fs"] == record.fs
        assert header["sig_len"] == record.sig_len

    def test_wfdb_to_edf_interleaves_blocks(self, tmp_path):
        """Write each channel's samples once per EDF data block."""
        output = tmp_path / "copy.edf"
        wfdb_to_edf("sample-data/100", sampto=3601, output_filename=str(output))
        digital = rdrecord("sample-data/100", sampto=3601).adc()
        first_block = np.concatenate((digital[:3600, 0], digital[:3600, 1]))
        last_block = digital[3600, :]
        raw = np.frombuffer(output.read_bytes()[256 * 3 :], dtype="<i2")

        np.testing.assert_array_equal(
            raw[: 2 * 3601], np.concatenate((first_block, last_block))
        )
        np.testing.assert_array_equal(raw[2 * 3601 :], -32768)
        assert raw.size == 2 * 2 * 3600

    def test_wfdb_to_edf_rejects_samples_beyond_16_bits(self, tmp_path):
        """Values that EDF cannot store raise an error instead of wrapping."""
        sig = np.array([[0], [50000], [-70000], [100]], dtype=np.int32)
        wrsamp(
            "wide",
            fs=250,
            units=["mV"],
            sig_name=["ECG"],
            d_signal=sig,
            fmt=["32"],
            adc_gain=[1000.0],
            baseline=[0],
            write_dir=str(tmp_path),
        )
        output = tmp_path / "wide.edf"

        with pytest.raises(ValueError, match="signal 'ECG' has digital values"):
            wfdb_to_edf(str(tmp_path / "wide"), output_filename=str(output))
        assert not output.exists()


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

    @staticmethod
    def read_wav(wav_file, n_sig):
        """Read the samples of a .wav file as an array with one column per signal."""
        with wave.open(wav_file, "rb") as wav:
            assert wav.getnchannels() == n_sig
            frames = wav.readframes(wav.getnframes())
        return np.frombuffer(frames, dtype="<i2").reshape((-1, n_sig))

    @staticmethod
    def expected_samples(record):
        """Shift the digital samples of a record to fill 16 bits."""
        shift = np.array([16 - res for res in record.adc_res])
        return (record.d_signal - record.adc_zero) << shift

    def test_sampfrom_sampto(self, tmp_path, monkeypatch):
        """Only the samples from sampfrom to sampto are written."""
        record_path = os.path.abspath("sample-data/100")
        monkeypatch.chdir(tmp_path)
        wfdb_to_wav(record_path, sampfrom=100, sampto=600, write_header=True)

        record = rdrecord(record_path, sampfrom=100, sampto=600, physical=False)
        expected = self.expected_samples(record)
        np.testing.assert_array_equal(self.read_wav("100.wav", 2), expected)
        assert rdheader("100").init_value == expected[0].tolist()

    def test_write_header_skew(self, tmp_path, monkeypatch):
        """The header does not repeat a skew already applied to the samples."""
        record_path = os.path.abspath("sample-data/100skew")
        monkeypatch.chdir(tmp_path)
        wfdb_to_wav(record_path, sampto=1000, write_header=True)

        expected = rdrecord(record_path, sampto=1000)
        record = rdrecord("100skew")
        assert not any(record.skew)
        np.testing.assert_array_equal(record.p_signal, expected.p_signal)

    def test_channels(self, tmp_path):
        """Only the selected channels are written."""
        wav_file = str(tmp_path / "out.wav")
        wfdb_to_wav("sample-data/100", channels=[1], output_filename=wav_file)

        record = rdrecord("sample-data/100", channels=[1], physical=False)
        expected = self.expected_samples(record)
        np.testing.assert_array_equal(self.read_wav(wav_file, 1), expected)

    @pytest.mark.parametrize("adc_res", [11, 12])
    def test_invalid_samples(self, tmp_path, monkeypatch, adc_res):
        """Invalid samples are written as -32768, the format 16 invalid value."""
        monkeypatch.chdir(tmp_path)
        d_signal = np.array([[100], [-2048], [-200]])
        wrsamp(
            "invalid",
            fs=100,
            units=["mV"],
            sig_name=["x"],
            d_signal=d_signal,
            fmt=["212"],
            adc_gain=[200],
            baseline=[0],
        )
        header = rdheader("invalid")
        header.adc_res = [adc_res]
        header.wrheader()

        # Write the .wav file and its header in a separate directory
        record_path = str(tmp_path / "invalid")
        (tmp_path / "out").mkdir()
        monkeypatch.chdir(tmp_path / "out")
        wfdb_to_wav(record_path, write_header=True)

        # -2048 is the format 212 invalid value
        shift = 16 - adc_res
        expected = [[100 << shift], [-32768], [-200 << shift]]
        np.testing.assert_array_equal(self.read_wav("invalid.wav", 1), expected)

        # Read back as missing
        record = rdrecord("invalid")
        expected = rdrecord(record_path)
        np.testing.assert_array_equal(record.p_signal, expected.p_signal)
        assert np.isnan(record.p_signal[1, 0])


class TestWfdbToMat:
    """Tests for the io.convert.matlab module."""

    SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample-data"

    def test_basic_roundtrip_and_padding(self, tmp_path, monkeypatch):
        """16-bit MAT-file output pads to 8-byte boundary and round-trips cleanly."""
        monkeypatch.chdir(tmp_path)
        src = str(self.SAMPLE_DATA / "100")

        # 2 channels * 5 samples * 2 bytes = 20 bytes (requires 4 bytes padding)
        wfdb_to_mat(src, sampto=5)

        mat_path = tmp_path / "100m.mat"
        assert mat_path.stat().st_size == 192 + 24

        mat_val = loadmat(mat_path)["val"]
        orig_d = rdrecord(src, sampto=5, physical=False)
        orig_p = rdrecord(src, sampto=5)
        expected_val = (orig_d.d_signal - np.asarray(orig_d.adc_zero)).T
        assert mat_val.dtype == np.int16
        np.testing.assert_array_equal(mat_val, expected_val)

        reloaded = rdrecord(str(tmp_path / "100m"))
        reloaded_d = rdrecord(str(tmp_path / "100m"), physical=False)
        np.testing.assert_allclose(reloaded.p_signal, orig_p.p_signal)
        assert reloaded.checksum == reloaded_d.calc_checksum()

    def test_channels_and_sampfrom_slice(self, tmp_path, monkeypatch):
        """channels selection and sampfrom without sampto produce valid output."""
        monkeypatch.chdir(tmp_path)
        sig = np.arange(40, dtype=np.int16).reshape(20, 2)
        wrsamp(
            "tiny",
            fs=100,
            units=["mV", "mV"],
            sig_name=["I", "II"],
            d_signal=sig,
            fmt=["16", "16"],
            adc_gain=[100, 200],
            baseline=[0, 0],
            write_dir=str(tmp_path),
        )

        # sampfrom > sig_len / 2 used to make desired_length negative
        wfdb_to_mat(str(tmp_path / "tiny"), sampfrom=12, channels=[1])

        mat_val = loadmat(tmp_path / "tinym.mat")["val"]
        assert mat_val.shape == (1, 8)
        np.testing.assert_array_equal(mat_val[0], sig[12:, 1])

        reloaded = rdrecord(str(tmp_path / "tinym"))
        orig = rdrecord(str(tmp_path / "tiny"), sampfrom=12, channels=[1])
        assert reloaded.n_sig == 1
        assert reloaded.sig_name == ["II"]
        np.testing.assert_allclose(reloaded.p_signal, orig.p_signal)

    def test_format_80_uint8(self, tmp_path, monkeypatch):
        """8-bit (format 80) records write offset-binary uint8 MAT-files."""
        monkeypatch.chdir(tmp_path)
        src = str(self.SAMPLE_DATA / "binformats")

        wfdb_to_mat(src, sampto=10, channels=[3])

        mat_val = loadmat(tmp_path / "binformatsm.mat")["val"]
        orig_d = rdrecord(src, sampto=10, channels=[3], physical=False)
        orig_p = rdrecord(src, sampto=10, channels=[3])
        assert mat_val.dtype == np.uint8
        np.testing.assert_array_equal(mat_val, (orig_d.d_signal + 128).T)

        reloaded = rdrecord(str(tmp_path / "binformatsm"))
        reloaded_d = rdrecord(str(tmp_path / "binformatsm"), physical=False)
        assert reloaded.fmt == ["80"]
        np.testing.assert_allclose(reloaded.p_signal, orig_p.p_signal)
        assert reloaded.checksum == reloaded_d.calc_checksum()

    def test_format_32_int32_and_missing_adc_fields(self, tmp_path, monkeypatch):
        """24/32-bit records and headers with unset adc_res/adc_zero write int32."""
        monkeypatch.chdir(tmp_path)
        src = str(self.SAMPLE_DATA / "n8_evoked_raw_95_F1_R9")

        wfdb_to_mat(src, sampto=10)

        mat_val = loadmat(tmp_path / "n8_evoked_raw_95_F1_R9m.mat")["val"]
        orig_d = rdrecord(src, sampto=10, physical=False)
        orig_p = rdrecord(src, sampto=10)
        assert mat_val.dtype == np.int32
        np.testing.assert_array_equal(mat_val, orig_d.d_signal.T)

        reloaded = rdrecord(str(tmp_path / "n8_evoked_raw_95_F1_R9m"))
        reloaded_d = rdrecord(str(tmp_path / "n8_evoked_raw_95_F1_R9m"), physical=False)
        assert reloaded.fmt == ["32", "32"]
        np.testing.assert_allclose(reloaded.p_signal, orig_p.p_signal)
        assert reloaded.checksum == reloaded_d.calc_checksum()

    @pytest.mark.parametrize("adc_zero", [-1, 1])
    def test_shifted_samples_promote_to_int32(self, tmp_path, monkeypatch, adc_zero):
        """Do not wrap a valid shifted sample to the missing value."""
        monkeypatch.chdir(tmp_path)
        samples = np.array([[-32768], [-32767], [0], [32767]], dtype=np.int16)
        Record(
            record_name="shifted",
            n_sig=1,
            sig_len=len(samples),
            fs=100,
            units=["mV"],
            sig_name=["I"],
            d_signal=samples,
            file_name=["shifted.dat"],
            fmt=["16"],
            adc_gain=[200],
            baseline=[0],
            adc_res=[16],
            adc_zero=[adc_zero],
            block_size=[0],
            init_value=[int(samples[0, 0])],
            checksum=[int(samples.sum() % 65536)],
        ).wrsamp()

        wfdb_to_mat("shifted")

        mat_val = loadmat("shiftedm.mat")["val"]
        assert mat_val.dtype == np.int32
        expected = samples[:, 0].astype(np.int64) - adc_zero
        expected[0] = -2147483648
        np.testing.assert_array_equal(mat_val[0], expected)
        output = rdrecord("shiftedm", physical=False)
        np.testing.assert_array_equal(output.d_signal[:, 0], expected)

    def test_adc_res_understates_sample_range(self, tmp_path, monkeypatch):
        """Choose the MAT type from actual samples as well as adc_res."""
        monkeypatch.chdir(tmp_path)
        samples = np.array([[100], [300], [-300]], dtype=np.int16)
        Record(
            record_name="understated",
            n_sig=1,
            sig_len=len(samples),
            fs=100,
            units=["mV"],
            sig_name=["I"],
            d_signal=samples,
            file_name=["understated.dat"],
            fmt=["16"],
            adc_gain=[200],
            baseline=[0],
            adc_res=[8],
            adc_zero=[0],
            block_size=[0],
            init_value=[int(samples[0, 0])],
            checksum=[int(samples.sum() % 65536)],
        ).wrsamp()

        wfdb_to_mat("understated")

        mat_val = loadmat("understatedm.mat")["val"]
        assert mat_val.dtype == np.int16
        np.testing.assert_array_equal(mat_val[0], samples[:, 0])

    def test_shifted_samples_exceed_int32(self, tmp_path, monkeypatch):
        """Fail before writing when no supported type preserves valid samples."""
        monkeypatch.chdir(tmp_path)
        Record(
            record_name="too_large",
            n_sig=1,
            sig_len=1,
            fs=100,
            units=["mV"],
            sig_name=["I"],
            d_signal=np.array([[2147483647]], dtype=np.int32),
            file_name=["too_large.dat"],
            fmt=["32"],
            adc_gain=[200],
            baseline=[0],
            adc_res=[32],
            adc_zero=[-1],
            block_size=[0],
            init_value=[2147483647],
            checksum=[65535],
        ).wrsamp()

        with pytest.raises(ValueError, match="Shifted samples exceed"):
            wfdb_to_mat("too_large")
        assert not (tmp_path / "too_largem.mat").exists()

    def test_missing_samples_preserved(self, tmp_path, monkeypatch):
        """Missing samples are preserved without overflow."""
        monkeypatch.chdir(tmp_path)

        # Missing samples with non-zero adc_zero (format 516)
        mixed_src = str(self.SAMPLE_DATA / "mixedsignals")
        wfdb_to_mat(mixed_src, sampto=20)
        mixed_orig = rdrecord(mixed_src, sampto=20)
        mixed_reloaded = rdrecord(str(tmp_path / "mixedsignalsm"))
        np.testing.assert_allclose(
            mixed_reloaded.p_signal, mixed_orig.p_signal, equal_nan=True
        )

    def test_skew_not_applied_twice(self, tmp_path, monkeypatch):
        """Skewed record should not be skewed a second time when read back."""
        monkeypatch.chdir(tmp_path)

        skew_src = str(self.SAMPLE_DATA / "100skew")
        wfdb_to_mat(skew_src, sampto=20)
        skew_orig = rdrecord(skew_src, sampto=20)
        skew_reloaded = rdrecord(str(tmp_path / "100skewm"))
        np.testing.assert_allclose(skew_reloaded.p_signal, skew_orig.p_signal)

    def test_forward_slash_path(self, tmp_path, monkeypatch):
        """Forward-slash record paths work cross-platform and write into cwd."""
        monkeypatch.chdir(tmp_path)
        # The bug this guards against only appeared on Windows; on other
        # platforms this checks that the basename is still used.
        src = (self.SAMPLE_DATA / "100").as_posix()

        wfdb_to_mat(src, sampto=8)

        assert (tmp_path / "100m.mat").exists()
        assert (tmp_path / "100m.hea").exists()
