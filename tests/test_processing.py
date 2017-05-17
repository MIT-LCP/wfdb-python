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
