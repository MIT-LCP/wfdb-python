import wfdb
import numpy

class test_processing():

    def test_1(self):
        sig, fields = wfdb.srdsamp('sampledata/100')
        ann = wfdb.rdann('sampledata/100', 'atr')

        fs = fields['fs']
        fs_target = 50

        new_sig, new_ann = wfdb.processing.resample_singlechan(sig[:, 0], ann, fs, fs_target)

        expected_length = int(sig.shape[0]*fs_target/fs)

        assert new_sig.shape[0] == expected_length

    def test_2(self):
        sig, fields = wfdb.srdsamp('sampledata/100')
        ann = wfdb.rdann('sampledata/100', 'atr')

        fs = fields['fs']
        fs_target = 50

        new_sig, new_ann = wfdb.processing.resample_multichan(sig, ann, fs, fs_target)

        expected_length = int(sig.shape[0]*fs_target/fs)

        assert new_sig.shape[0] == expected_length
        assert new_sig.shape[1] == sig.shape[1]

    def test_3(self):
        sig, _ = wfdb.srdsamp('sampledata/100')
        lb = -5
        ub = 15

        x = wfdb.processing.normalize(sig[:, 0], lb, ub)
        assert x.shape[0] == sig.shape[0]
        assert numpy.min(x) >= lb
        assert numpy.max(x) <= ub

    def test_4(self):
        x = []
        hp, sp = wfdb.processing.find_peaks(x)
        assert hp.shape == (0,)
        assert sp.shape == (0,)

    def test_5(self):
        x = [0, 2, 1, 0, -10, -15, -15, -15, 9, 8, 0, 0, 1, 2, 10]
        hp, sp = wfdb.processing.find_peaks(x)
        assert numpy.array_equal(hp, [1, 8])
        assert numpy.array_equal(sp, [6, 10])
