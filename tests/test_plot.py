import wfdb
from wfdb.plot import plot
import unittest


class TestPlot(unittest.TestCase):
    def test_get_plot_dims(self):
        sampfrom = 0
        sampto = 3000
        record = wfdb.rdrecord(
            "sample-data/100", physical=True, sampfrom=sampfrom, sampto=sampto
        )
        ann = wfdb.rdann(
            "sample-data/100", "atr", sampfrom=sampfrom, sampto=sampto
        )
        sig_len, n_sig, n_annot, n_subplots = plot._get_plot_dims(
            signal=record.p_signal, ann_samp=[ann.sample]
        )

        assert sig_len == sampto - sampfrom
        assert n_sig == record.n_sig
        assert n_annot == 1
        assert n_subplots == record.n_sig

    def test_create_figure_single_subplots(self):
        n_subplots = 1
        fig, axes = plot.create_figure(
            n_subplots, sharex=True, sharey=True, figsize=None
        )
        assert fig is not None
        assert axes is not None
        assert len(axes) == n_subplots

    def test_create_figure_multiple_subplots(self):
        n_subplots = 5
        fig, axes = plot.create_figure(
            n_subplots, sharex=True, sharey=True, figsize=None
        )
        assert fig is not None
        assert axes is not None
        assert len(axes) == n_subplots
