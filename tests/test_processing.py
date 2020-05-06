import numpy as np

import wfdb
from wfdb import processing


class test_processing():
    """
    Test processing functions
    """
    def test_resample_single(self):
        sig, fields = wfdb.rdsamp('sample-data/100')
        ann = wfdb.rdann('sample-data/100', 'atr')

        fs = fields['fs']
        fs_target = 50

        new_sig, new_ann = processing.resample_singlechan(sig[:, 0], ann, fs,
                                                          fs_target)

        expected_length = int(sig.shape[0] * fs_target / fs)

        assert new_sig.shape[0] == expected_length

    def test_resample_multi(self):
        sig, fields = wfdb.rdsamp('sample-data/100')
        ann = wfdb.rdann('sample-data/100', 'atr')

        fs = fields['fs']
        fs_target = 50

        new_sig, new_ann = processing.resample_multichan(sig, ann, fs, fs_target)

        expected_length = int(sig.shape[0]*fs_target/fs)

        assert new_sig.shape[0] == expected_length
        assert new_sig.shape[1] == sig.shape[1]

    def test_normalize_bound(self):
        sig, _ = wfdb.rdsamp('sample-data/100')
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

        record = wfdb.rdrecord('sample-data/100', channels=[0],
                                  sampfrom=9998, sampto=19998, physical=False)

        expected_peaks = [271, 580, 884, 1181, 1469, 1770, 2055, 2339, 2634,
                          2939, 3255, 3551, 3831, 4120, 4412, 4700, 5000, 5299,
                          5596, 5889, 6172, 6454, 6744, 7047, 7347, 7646, 7936,
                          8216, 8503, 8785, 9070, 9377, 9682]

        peaks = processing.gqrs_detect(d_sig=record.d_signal[:,0],
                                       fs=record.fs,
                                       adc_gain=record.adc_gain[0],
                                       adc_zero=record.adc_zero[0],
                                       threshold=1.0)

        assert np.array_equal(peaks, expected_peaks)

    def test_correct_peaks(self):
        sig, fields = wfdb.rdsamp('sample-data/100')
        ann = wfdb.rdann('sample-data/100', 'atr')
        fs = fields['fs']
        min_bpm = 10
        max_bpm = 350
        min_gap = fs*60/min_bpm
        max_gap = fs * 60 / max_bpm

        y_idxs = processing.correct_peaks(sig=sig[:,0], peak_inds=ann.sample,
                                          search_radius=int(max_gap),
                                          smooth_window_size=150)

        yz = np.zeros(sig.shape[0])
        yz[y_idxs] = 1
        yz = np.where(yz[:10000]==1)[0]

        expected_peaks = [77, 370, 663, 947, 1231, 1515, 1809, 2045, 2403,
                          2706, 2998, 3283, 3560, 3863, 4171, 4466, 4765, 5061,
                          5347, 5634, 5919, 6215, 6527, 6824, 7106, 7393, 7670,
                          7953, 8246, 8539, 8837, 9142, 9432, 9710, 9998]

        assert np.array_equal(yz, expected_peaks)

class test_qrs():
    """
    Testing QRS detectors
    """
    def test_xqrs(self):
        """
        Run XQRS detector on record 100 and compare to reference annotations
        """
        sig, fields = wfdb.rdsamp('sample-data/100', channels=[0])
        ann_ref = wfdb.rdann('sample-data/100','atr')

        xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
        xqrs.detect()

        comparitor = processing.compare_annotations(ann_ref.sample[1:],
                                                    xqrs.qrs_inds,
                                                    int(0.1 * fields['fs']))

        assert comparitor.sensitivity > 0.99
        assert comparitor.positive_predictivity > 0.99
