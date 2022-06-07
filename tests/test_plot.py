import unittest

import matplotlib.pyplot as plt
import numpy as np

import wfdb
from wfdb.plot import plot


class TestPlotWfdb(unittest.TestCase):
    """
    Tests for the wfdb.plot_wfdb function
    """

    def assertAxesMatchSignal(self, axes, signal, t_divisor=1):
        """
        Check that axis limits are reasonable for plotting a signal array.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            An Axes object.
        signal : numpy.ndarray
            A one-dimensional array of sample values.
        t_divisor : float, optional
            The intended plotting resolution (number of samples of `signal`
            per unit of the X axis.)

        """
        xmin, xmax = axes.get_xlim()
        tmin = 0
        tmax = (len(signal) - 1) / t_divisor
        # The range from tmin to tmax should fit within the plot.
        self.assertLessEqual(
            xmin,
            tmin,
            msg=f"X range is [{xmin}, {xmax}]; expected [{tmin}, {tmax}]",
        )
        self.assertGreaterEqual(
            xmax,
            tmax,
            msg=f"X range is [{xmin}, {xmax}]; expected [{tmin}, {tmax}]",
        )
        # The padding on left and right sides should be approximately equal.
        self.assertAlmostEqual(
            xmin - tmin,
            tmax - xmax,
            delta=(tmax - tmin) / 10 + 1 / t_divisor,
            msg=f"X range is [{xmin}, {xmax}]; expected [{tmin}, {tmax}]",
        )

        ymin, ymax = axes.get_ylim()
        vmin = np.nanmin(signal)
        vmax = np.nanmax(signal)
        # The range from vmin to vmax should fit within the plot.
        self.assertLessEqual(
            ymin,
            vmin,
            msg=f"Y range is [{ymin}, {ymax}]; expected [{vmin}, {vmax}]",
        )
        self.assertGreaterEqual(
            ymax,
            vmax,
            msg=f"Y range is [{ymin}, {ymax}]; expected [{vmin}, {vmax}]",
        )
        # The padding on top and bottom should be approximately equal.
        self.assertAlmostEqual(
            ymin - vmin,
            vmax - ymax,
            delta=(vmax - vmin) / 10,
            msg=f"Y range is [{ymin}, {ymax}]; expected [{vmin}, {vmax}]",
        )

    def test_physical_smooth(self):
        """
        Plot a record with physical, single-frequency data
        """
        record = wfdb.rdrecord(
            "sample-data/100",
            sampto=1000,
            physical=True,
            smooth_frames=True,
        )
        self.assertIsNotNone(record.p_signal)

        annotation = wfdb.rdann("sample-data/100", "atr", sampto=1000)

        fig = wfdb.plot_wfdb(
            record,
            annotation,
            time_units="samples",
            ecg_grids="all",
            return_fig=True,
        )
        plt.close(fig)

        self.assertEqual(len(fig.axes), record.n_sig)
        for ch in range(record.n_sig):
            self.assertAxesMatchSignal(fig.axes[ch], record.p_signal[:, ch])

    def test_digital_smooth(self):
        """
        Plot a record with digital, single-frequency data
        """
        record = wfdb.rdrecord(
            "sample-data/drive02",
            sampto=1000,
            physical=False,
            smooth_frames=True,
        )
        self.assertIsNotNone(record.d_signal)

        fig = wfdb.plot_wfdb(record, time_units="seconds", return_fig=True)
        plt.close(fig)

        self.assertEqual(len(fig.axes), record.n_sig)
        for ch in range(record.n_sig):
            self.assertAxesMatchSignal(
                fig.axes[ch], record.d_signal[:, ch], record.fs
            )

    def test_physical_multifrequency(self):
        """
        Plot a record with physical, multi-frequency data
        """
        record = wfdb.rdrecord(
            "sample-data/wave_4",
            sampto=10,
            physical=True,
            smooth_frames=False,
        )
        self.assertIsNotNone(record.e_p_signal)

        fig = wfdb.plot_wfdb(record, time_units="seconds", return_fig=True)
        plt.close(fig)

        self.assertEqual(len(fig.axes), record.n_sig)
        for ch in range(record.n_sig):
            self.assertAxesMatchSignal(
                fig.axes[ch],
                record.e_p_signal[ch],
                record.fs * record.samps_per_frame[ch],
            )

    def test_digital_multifrequency(self):
        """
        Plot a record with digital, multi-frequency data
        """
        record = wfdb.rdrecord(
            "sample-data/multi-segment/041s/041s",
            sampto=1000,
            physical=False,
            smooth_frames=False,
        )
        self.assertIsNotNone(record.e_d_signal)

        fig = wfdb.plot_wfdb(record, time_units="seconds", return_fig=True)
        plt.close(fig)

        self.assertEqual(len(fig.axes), record.n_sig)
        for ch in range(record.n_sig):
            self.assertAxesMatchSignal(
                fig.axes[ch],
                record.e_d_signal[ch],
                record.fs * record.samps_per_frame[ch],
            )


class TestPlotInternal(unittest.TestCase):
    """
    Unit tests for internal wfdb.plot.plot functions
    """

    def test_get_plot_dims(self):
        sampfrom = 0
        sampto = 3000
        record = wfdb.rdrecord(
            "sample-data/100", physical=True, sampfrom=sampfrom, sampto=sampto
        )
        ann = wfdb.rdann(
            "sample-data/100", "atr", sampfrom=sampfrom, sampto=sampto
        )
        sig_len, n_sig, n_annot, n_subplots = plot.get_plot_dims(
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


if __name__ == "__main__":
    unittest.main()
