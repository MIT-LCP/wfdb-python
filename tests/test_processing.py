from unittest import mock

import numpy as np

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
