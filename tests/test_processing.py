from unittest import mock

import numpy as np
import pytest

import wfdb
from wfdb import processing


class TestProcessing:
    """Test processing functions"""

    def test_resample_single(self):
        sig, fields = wfdb.rdsamp("sample-data/100")
        ann = wfdb.rdann("sample-data/100", "atr")

        fs = fields["fs"]
        fs_target = 50

        new_sig, new_ann = processing.resample_singlechan(sig[:, 0], ann, fs, fs_target)

        expected_length = int(sig.shape[0] * fs_target / fs)

        assert new_sig.shape[0] == expected_length

    def test_resample_multi(self):
        sig, fields = wfdb.rdsamp("sample-data/100")
        ann = wfdb.rdann("sample-data/100", "atr")

        fs = fields["fs"]
        fs_target = 50

        new_sig, new_ann = processing.resample_multichan(sig, ann, fs, fs_target)

        expected_length = int(sig.shape[0] * fs_target / fs)

        assert new_sig.shape[0] == expected_length
        assert new_sig.shape[1] == sig.shape[1]

    def test_resample_leading_nan(self):
        """Leading NaN values take the first valid value before resampling."""
        x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
        filled = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])

        new_x, _ = processing.resample_sig(x, fs=2, fs_target=4)
        expected, _ = processing.resample_sig(filled, fs=2, fs_target=4)

        assert np.isfinite(new_x).all()
        np.testing.assert_allclose(new_x, expected)

    def test_resample_trailing_and_inner_nan(self):
        """Inner NaN values are interpolated, and trailing ones take the last."""
        x = np.array([1.0, np.nan, 3.0, 4.0, np.nan, np.nan])
        filled = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])

        new_x, _ = processing.resample_sig(x, fs=2, fs_target=4)
        expected, _ = processing.resample_sig(filled, fs=2, fs_target=4)

        np.testing.assert_allclose(new_x, expected)

    def test_normalize_bound(self):
        sig, _ = wfdb.rdsamp("sample-data/100")
        lb = -5
        ub = 15

        x = processing.normalize_bound(sig[:, 0], lb, ub)
        assert x.shape[0] == sig.shape[0]
        assert np.min(x) >= lb
        assert np.max(x) <= ub

    def test_find_peaks(self):
        x = [0, 2, 1, 0, -10, -15, -15, -15, 9, 8, 0, 0, 1, 2, 10]
        hp, sp = processing.find_peaks(x)
        assert np.array_equal(hp, [1, 8])
        assert np.array_equal(sp, [6, 10])

    def test_find_peaks_empty(self):
        x = []
        hp, sp = processing.find_peaks(x)
        assert hp.shape == (0,)
        assert sp.shape == (0,)

    def test_gqrs(self):
        record = wfdb.rdrecord(
            "sample-data/100",
            channels=[0],
            sampfrom=9998,
            sampto=19998,
            physical=False,
        )

        expected_peaks = [
            271,
            580,
            884,
            1181,
            1469,
            1770,
            2055,
            2339,
            2634,
            2939,
            3255,
            3551,
            3831,
            4120,
            4412,
            4700,
            5000,
            5299,
            5596,
            5889,
            6172,
            6454,
            6744,
            7047,
            7347,
            7646,
            7936,
            8216,
            8503,
            8785,
            9070,
            9377,
            9682,
        ]

        peaks = processing.gqrs_detect(
            d_sig=record.d_signal[:, 0],
            fs=record.fs,
            adc_gain=record.adc_gain[0],
            adc_zero=record.adc_zero[0],
            threshold=1.0,
        )

        assert np.array_equal(peaks, expected_peaks)

    def test_correct_peaks(self):
        sig, fields = wfdb.rdsamp("sample-data/100")
        ann = wfdb.rdann("sample-data/100", "atr")
        fs = fields["fs"]
        min_bpm = 10
        max_bpm = 350
        min_gap = fs * 60 / min_bpm
        max_gap = fs * 60 / max_bpm

        y_idxs = processing.correct_peaks(
            sig=sig[:, 0],
            peak_inds=ann.sample,
            search_radius=int(max_gap),
            smooth_window_size=150,
        )

        yz = np.zeros(sig.shape[0])
        yz[y_idxs] = 1
        yz = np.where(yz[:10000] == 1)[0]

        expected_peaks = [
            77,
            370,
            663,
            947,
            1231,
            1515,
            1809,
            2045,
            2403,
            2706,
            2998,
            3283,
            3560,
            3863,
            4171,
            4466,
            4765,
            5061,
            5347,
            5634,
            5919,
            6215,
            6527,
            6824,
            7106,
            7393,
            7670,
            7953,
            8246,
            8539,
            8837,
            9142,
            9432,
            9710,
            9998,
        ]

        assert np.array_equal(yz, expected_peaks)

    def test_correct_peaks_window_edges(self):
        """Peaks at the edge of the search radius or the signal are found."""
        sig = np.zeros(30)
        sig[13] = 5
        sig[29] = 5

        # Peak exactly search_radius samples after the guess
        assert processing.correct_peaks(
            sig, np.array([10]), search_radius=3, smooth_window_size=9, peak_dir="up"
        ).tolist() == [13]
        # Peak on the last sample of the signal
        assert processing.correct_peaks(
            sig, np.array([27]), search_radius=5, smooth_window_size=9, peak_dir="up"
        ).tolist() == [29]

    @pytest.mark.parametrize("peak", [0, 1, 4, 6])
    @pytest.mark.parametrize("peak_dir", ["up", "down"])
    def test_correct_peaks_near_start(self, peak, peak_dir):
        """Peaks closer to the start than search_radius aren't shifted below 0."""
        sig = np.zeros(30)
        sig[peak] = 5 if peak_dir == "up" else -5

        assert processing.correct_peaks(
            sig, np.array([2]), search_radius=5, smooth_window_size=9, peak_dir=peak_dir
        ).tolist() == [peak]


class TestComputeHr:
    """Tests for compute_hr."""

    def test_uses_last_rr_interval(self):
        """The rate after the last beat comes from the last RR interval."""
        hr = processing.compute_hr(sig_len=14, qrs_inds=[0, 2, 6, 12], fs=60)

        expected = np.full(14, np.nan)
        expected[3:7] = 60 * 60 / 2
        expected[7:13] = 60 * 60 / 4
        expected[13:] = 60 * 60 / 6
        np.testing.assert_allclose(hr, expected)

    def test_two_beats(self):
        """Two beats give one RR interval, so the rate is known after them."""
        hr = processing.compute_hr(sig_len=10, qrs_inds=[2, 6], fs=1)

        expected = np.full(10, np.nan)
        expected[7:] = 60 / 4
        np.testing.assert_allclose(hr, expected)

    def test_last_beat_past_end(self):
        """A last beat beyond the signal doesn't raise an IndexError."""
        hr = processing.compute_hr(sig_len=8, qrs_inds=[0, 2, 6, 12], fs=60)

        expected = np.full(8, np.nan)
        expected[3:7] = 60 * 60 / 2
        expected[7:] = 60 * 60 / 4
        np.testing.assert_allclose(hr, expected)

    @pytest.mark.parametrize("qrs_inds", [[2, 8, 8, 15], [2, 8, 5, 15], [2, 8, 8]])
    def test_rejects_unordered_indices(self, qrs_inds):
        """Repeated or decreasing beats raise a ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            processing.compute_hr(sig_len=20, qrs_inds=qrs_inds, fs=1)

    def test_rejects_unordered_array(self):
        """The check also applies to NumPy arrays."""
        with pytest.raises(ValueError, match=r"qrs_inds\[1\] = 8"):
            processing.compute_hr(sig_len=20, qrs_inds=np.array([2, 8, 8]), fs=1)

    def test_rejects_unordered_unsigned(self):
        """Decreasing unsigned indices are caught, although np.diff wraps."""
        qrs_inds = np.array([2, 8, 5], dtype=np.uint32)
        with pytest.raises(ValueError, match=r"qrs_inds\[1\] = 8"):
            processing.compute_hr(sig_len=20, qrs_inds=qrs_inds, fs=1)


class TestCompareAnnotations:
    """Tests for compare_annotations."""

    def test_test_annotation_matched_once(self):
        """A test annotation isn't matched to two reference annotations."""
        comparitor = processing.compare_annotations(
            np.array([0, 2, 4, 5]), np.array([1, 5]), window_width=6
        )

        assert comparitor.matching_sample_nums.tolist() == [0, -1, -1, 1]
        assert (comparitor.tp, comparitor.fp, comparitor.fn) == (2, 0, 2)
        assert comparitor.positive_predictivity == 1.0

    def test_random_annotations(self):
        """Matches are unique, in order and within the window."""
        rng = np.random.default_rng(0)
        for _ in range(2000):
            ref = np.sort(rng.choice(40, rng.integers(1, 9), replace=False))
            test = np.sort(rng.choice(40, rng.integers(1, 9), replace=False))
            window_width = int(rng.integers(1, 10))
            comparitor = processing.compare_annotations(ref, test, window_width)

            matches = comparitor.matching_sample_nums
            matched = matches[matches != -1]
            assert np.all(np.diff(matched) > 0)
            assert np.all(np.abs(ref[matches != -1] - test[matched]) < window_width)
            assert 0 <= comparitor.fp <= len(test)


class TestQrs:
    """Testing QRS detectors"""

    def test_xqrs(self):
        """Run XQRS detector on record 100 and compare to reference annotations"""
        sig, fields = wfdb.rdsamp("sample-data/100", channels=[0])
        ann_ref = wfdb.rdann("sample-data/100", "atr")

        xqrs = processing.XQRS(sig=sig[:, 0], fs=fields["fs"])
        xqrs.detect()

        comparitor = processing.compare_annotations(
            ann_ref.sample[1:], xqrs.qrs_inds, int(0.1 * fields["fs"])
        )

        assert comparitor.sensitivity > 0.99
        assert comparitor.positive_predictivity > 0.99

    def test_xqrs_backsearch_checks_each_peak_once(self):
        """Backsearch does not recheck rejected peaks in a gap (issue #352)."""
        sig, fields = wfdb.rdsamp("sample-data/100", channels=[0])
        fs = fields["fs"]
        x = sig[:, 0]
        # Beats, then low-amplitude noise (as from a disconnected lead), then beats
        rng = np.random.default_rng(0)
        gap = 0.01 * rng.standard_normal(120 * fs)
        y = np.concatenate([x[: 30 * fs], gap, x[30 * fs : 60 * fs]])

        xqrs = processing.XQRS(sig=y, fs=fs)
        checked = []
        is_qrs = xqrs._is_qrs

        def spy(peak_num, backsearch=False):
            if backsearch:
                checked.append(peak_num)
            return is_qrs(peak_num, backsearch=backsearch)

        with mock.patch.object(xqrs, "_is_qrs", side_effect=spy):
            xqrs.detect(verbose=False)

        assert len(checked) > 0
        assert len(checked) == len(set(checked))
        # Beats on both sides of the gap are still found
        n_before = np.sum(xqrs.qrs_inds < 30 * fs)
        n_after = np.sum(xqrs.qrs_inds > 150 * fs)
        assert n_before > 30 and n_after > 30

    def test_xqrs_backsearch_skip_assumptions(self):
        """The backsearch skip relies on how XQRS updates its state (#597)."""
        sig, fields = wfdb.rdsamp("sample-data/100", channels=[0])
        fs = fields["fs"]
        x = sig[:, 0]
        rng = np.random.default_rng(0)
        gap = 0.01 * rng.standard_normal(120 * fs)
        # Scaled-down beats after the gap are only found by backsearch
        y = np.concatenate([x[: 30 * fs], gap, 0.4 * x[30 * fs : 60 * fs]])

        xqrs = processing.XQRS(sig=y, fs=fs)
        update_noise = xqrs._update_noise
        update_qrs = xqrs._update_qrs

        def noise_spy(peak_num):
            qrs_thr, last_qrs_ind = xqrs.qrs_thr, xqrs.last_qrs_ind
            update_noise(peak_num)
            # A noise update must not change the result of a backsearch
            assert (xqrs.qrs_thr, xqrs.last_qrs_ind) == (qrs_thr, last_qrs_ind)

        def qrs_spy(peak_num, backsearch=False):
            update_qrs(peak_num, backsearch=backsearch)
            # Peaks up to the current one count as already searched
            assert xqrs.last_qrs_peak_num == xqrs.peak_num

        with (
            mock.patch.object(xqrs, "_update_noise", side_effect=noise_spy),
            mock.patch.object(xqrs, "_update_qrs", side_effect=qrs_spy),
        ):
            xqrs.detect(verbose=False)

        assert len(xqrs.backsearch_qrs_inds) > 0


class TestXQRSDetect:
    """Tests for the xqrs_detect convenience wrapper."""

    def test_xqrs_detect_forwards_learn(self):
        """xqrs_detect forwards its learn argument to XQRS.detect (issue #473)."""
        sig, fields = wfdb.rdsamp("sample-data/100", channels=[0])

        def fake_detect(self, *args, **kwargs):
            # Stand in for the real detector so xqrs_detect can return.
            self.qrs_inds = np.array([], dtype=int)

        with mock.patch.object(
            processing.XQRS, "detect", autospec=True, side_effect=fake_detect
        ) as mock_detect:
            processing.xqrs_detect(sig=sig[:, 0], fs=fields["fs"], learn=False)

        assert mock_detect.call_count == 1
        assert mock_detect.call_args.kwargs["learn"] is False


class TestRr2Ann:
    """Tests for rr2ann."""

    def test_times_are_rounded(self, tmp_path, monkeypatch):
        """Times are rounded to the nearest sample, not truncated."""
        monkeypatch.chdir(tmp_path)
        times = np.array([0.29, 0.57, 1.13])
        processing.hr.rr2ann(times, "t", "atr", fs=100, as_time=True)

        ann = wfdb.rdann("t", "atr")
        np.testing.assert_array_equal(ann.sample, [29, 57, 113])

    def test_intervals_with_symbols(self, tmp_path, monkeypatch):
        """Intervals with a column of symbols are summed into samples."""
        monkeypatch.chdir(tmp_path)
        rr_array = np.array([[4, 17, 18], ["V", "N", "V"]]).T
        processing.hr.rr2ann(rr_array, "t", "atr")

        ann = wfdb.rdann("t", "atr")
        np.testing.assert_array_equal(ann.sample, [4, 21, 39])
        assert ann.symbol == ["V", "N", "V"]

    def test_times_with_symbols(self, tmp_path, monkeypatch):
        """Times with a column of symbols, as in the docstring example."""
        monkeypatch.chdir(tmp_path)
        rr_array = np.array([[0.2, 0.6, 1.3], ["V", "N", "V"]]).T
        processing.hr.rr2ann(rr_array, "t", "atr", fs=100, as_time=True)

        ann = wfdb.rdann("t", "atr")
        np.testing.assert_array_equal(ann.sample, [20, 60, 130])
        assert ann.symbol == ["V", "N", "V"]

    def test_fractional_intervals(self, tmp_path, monkeypatch):
        """Fractional intervals are summed before rounding, so they don't drift."""
        monkeypatch.chdir(tmp_path)
        processing.hr.rr2ann(np.array([4.5, 4.5, 4.5]), "t", "atr")
        np.testing.assert_array_equal(wfdb.rdann("t", "atr").sample, [4, 9, 14])

        rr_array = np.array([[4.5, 4.5, 4.5], ["V", "N", "V"]]).T
        processing.hr.rr2ann(rr_array, "t", "atr")
        np.testing.assert_array_equal(wfdb.rdann("t", "atr").sample, [4, 9, 14])
