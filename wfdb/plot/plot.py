import os

import numpy as np

from wfdb.io.record import Record, rdrecord
from wfdb.io.util import downround, upround
from wfdb.io.annotation import Annotation


def _expand_channels(signal):
    """
    Convert application-specified signal data to a list.

    Parameters
    ----------
    signal : 1d or 2d numpy array or list or None
        The signal or signals to be plotted.  If signal is a
        one-dimensional array, it is assumed to represent a single channel.
        If it is a two-dimensional array, axes 0 and 1 must represent time
        and channel number respectively.  Otherwise it must be a list of
        one-dimensional arrays (one for each channel.)

    Returns
    -------
    signal : list
        A list of one-dimensional arrays (one for each channel.)

    """
    if signal is None:
        return []
    elif hasattr(signal, "ndim"):
        if signal.ndim == 1:
            return [signal]
        elif signal.ndim == 2:
            return list(signal.transpose())
        else:
            raise ValueError(
                "invalid shape for signal array: {}".format(signal.shape)
            )
    else:
        signal = list(signal)
        if any(s.ndim != 1 for s in signal):
            raise ValueError(
                "invalid shape for signal array(s): {}".format(
                    [s.shape for s in signal]
                )
            )
        return signal


def _get_sampling_freq(sampling_freq, n_sig, frame_freq):
    """
    Convert application-specified sampling frequency to a list.

    Parameters
    ----------
    sampling_freq : number or sequence or None
        The sampling frequency or frequencies of the signals.  If this is a
        list, its length must equal `n_sig`.  If unset, defaults to
        `frame_freq`.
    n_sig : int
        Number of channels.
    frame_freq : number or None
        Default sampling frequency (record frame frequency).

    Returns
    -------
    sampling_freq : list
        The sampling frequency for each channel (a list of length `n_sig`.)

    """
    if sampling_freq is None:
        return [frame_freq] * n_sig
    elif hasattr(sampling_freq, "__len__"):
        if len(sampling_freq) != n_sig:
            raise ValueError(
                "length mismatch: n_sig = {}, "
                "len(sampling_freq) = {}".format(n_sig, len(sampling_freq))
            )
        return list(sampling_freq)
    else:
        return [sampling_freq] * n_sig


def _get_ann_freq(ann_freq, n_annot, frame_freq):
    """
    Convert application-specified annotation frequency to a list.

    Parameters
    ----------
    ann_freq : number or sequence or None
        The sampling frequency or frequencies of the annotations.  If this
        is a list, its length must equal `n_annot`.  If unset, defaults to
        `frame_freq`.
    n_annot : int
        Number of channels.
    frame_freq : number or None
        Default sampling frequency (record frame frequency).

    Returns
    -------
    ann_freq : list
        The sampling frequency for each channel (a list of length `n_annot`).

    """
    if ann_freq is None:
        return [frame_freq] * n_annot
    elif hasattr(ann_freq, "__len__"):
        if len(ann_freq) != n_annot:
            raise ValueError(
                "length mismatch: n_annot = {}, "
                "len(ann_freq) = {}".format(n_annot, len(ann_freq))
            )
        return list(ann_freq)
    else:
        return [ann_freq] * n_annot


def plot_items(
    signal=None,
    ann_samp=None,
    ann_sym=None,
    fs=None,
    time_units="samples",
    sig_name=None,
    sig_units=None,
    xlabel=None,
    ylabel=None,
    title=None,
    sig_style=[""],
    ann_style=["r*"],
    ecg_grids=[],
    figsize=None,
    sharex=False,
    sharey=False,
    return_fig=False,
    return_fig_axes=False,
    sampling_freq=None,
    ann_freq=None,
):
    """
    Subplot individual channels of signals and/or annotations.

    Parameters
    ----------
    signal : 1d or 2d numpy array or list, optional
        The signal or signals to be plotted.  If signal is a
        one-dimensional array, it is assumed to represent a single channel.
        If it is a two-dimensional array, axes 0 and 1 must represent time
        and channel number respectively.  Otherwise it must be a list of
        one-dimensional arrays (one for each channel).
    ann_samp: list, optional
        A list of annotation locations to plot, with each list item
        corresponding to a different channel. List items may be:

        - 1d numpy array, with values representing sample indices. Empty
          arrays are skipped.
        - list, with values representing sample indices. Empty lists
          are skipped.
        - None. For channels in which nothing is to be plotted.

        If `signal` is defined, the annotation locations will be overlaid on
        the signals, with the list index corresponding to the signal channel.
        The length of `annotation` does not have to match the number of
        channels of `signal`.
    ann_sym: list, optional
        A list of annotation symbols to plot, with each list item
        corresponding to a different channel. List items should be lists of
        strings. The symbols are plotted over the corresponding `ann_samp`
        index locations.
    fs : int, float, optional
        The sampling frequency of the signals and/or annotations. Used to
        calculate time intervals if `time_units` is not 'samples'. Also
        required for plotting ECG grids.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    sig_name : list, optional
        A list of strings specifying the signal names. Used with `sig_units`
        to form y labels, if `ylabel` is not set.
    sig_units : list, optional
        A list of strings specifying the units of each signal channel. Used
        with `sig_name` to form y labels, if `ylabel` is not set. This
        parameter is required for plotting ECG grids.
    xlabel : list, optional
        A list of strings specifying the final x labels to be used. If this
        option is present, no 'time/'`time_units` is used.
    ylabel : list, optional
        A list of strings specifying the final y labels. If this option is
        present, `sig_name` and `sig_units` will not be used for labels.
    title : str, optional
        The title of the graph.
    sig_style : list, optional
        A list of strings, specifying the style of the matplotlib plot
        for each signal channel. The list length should match the number
        of signal channels. If the list has a length of 1, the style
        will be used for all channels.
    ann_style : list, optional
        A list of strings, specifying the style of the matplotlib plot for each
        annotation channel. If the list has a length of 1, the style will be
        used for all channels.
    ecg_grids : list, optional
        A list of integers specifying channels in which to plot ECG grids. May
        also be set to 'all' for all channels. Major grids at 0.5mV, and minor
        grids at 0.125mV. All channels to be plotted with grids must have
        `sig_units` equal to 'uV', 'mV', or 'V'.
    sharex, sharey : bool, optional
        Controls sharing of properties among x (`sharex`) or y (`sharey`) axes.
        If True: x- or y-axis will be shared among all subplots.
        If False, each subplot x- or y-axis will be independent.
    figsize : tuple, optional
        Tuple pair specifying the width, and height of the figure. It is the
        'figsize' argument passed into matplotlib.pyplot's `figure` function.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.
    sampling_freq : number or sequence, optional
        The sampling frequency or frequencies of the signals.  If this is a
        list, it must have the same length as the number of channels.  If
        unspecified, defaults to `fs`.
    ann_freq : number or sequence, optional
        The sampling frequency or frequencies of the annotations.  If this
        is a list, it must have the same length as `ann_samp`.  If
        unspecified, defaults to `fs`.

    Returns
    -------
    fig : matplotlib figure, optional
        The matplotlib figure generated. Only returned if the 'return_fig'
        or 'return_fig_axes' parameter is set to True.
    axes : matplotlib axes, optional
        The matplotlib axes generated. Only returned if the 'return_fig_axes'
        parameter is set to True.

    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/100', sampto=3000)
    >>> ann = wfdb.rdann('sample-data/100', 'atr', sampto=3000)

    >>> wfdb.plot_items(signal=record.p_signal,
                        ann_samp=[ann.sample, ann.sample],
                        title='MIT-BIH Record 100', time_units='seconds',
                        figsize=(10,4), ecg_grids='all')

    """
    import matplotlib.pyplot as plt

    # Convert signal to a list if needed
    signal = _expand_channels(signal)

    # Figure out number of subplots required
    sig_len, n_sig, n_annot, n_subplots = _get_plot_dims(signal, ann_samp)

    # Convert sampling_freq and ann_freq to lists if needed
    sampling_freq = _get_sampling_freq(sampling_freq, n_sig, fs)
    ann_freq = _get_ann_freq(ann_freq, n_annot, fs)

    # Create figure
    fig, axes = _create_figure(n_subplots, sharex, sharey, figsize)
    try:
        if signal is not None:
            _plot_signal(
                signal,
                sig_len,
                n_sig,
                fs,
                time_units,
                sig_style,
                axes,
                sampling_freq=sampling_freq,
            )

        if ann_samp is not None:
            _plot_annotation(
                ann_samp,
                n_annot,
                ann_sym,
                signal,
                n_sig,
                fs,
                time_units,
                ann_style,
                axes,
                sampling_freq=sampling_freq,
                ann_freq=ann_freq,
            )

        if ecg_grids:
            _plot_ecg_grids(
                ecg_grids,
                fs,
                sig_units,
                time_units,
                axes,
                sampling_freq=sampling_freq,
            )

        # Add title and axis labels.
        # First, make sure that xlabel and ylabel inputs are valid
        if xlabel:
            if len(xlabel) != signal.shape[1]:
                raise Exception(
                    "The length of the xlabel must be the same as the "
                    "signal: {} values".format(signal.shape[1])
                )

        if ylabel:
            if len(ylabel) != n_subplots:
                raise Exception(
                    "The length of the ylabel must be the same as the "
                    "signal: {} values".format(n_subplots)
                )

        _label_figure(
            axes,
            n_subplots,
            time_units,
            sig_name,
            sig_units,
            xlabel,
            ylabel,
            title,
        )
    except BaseException:
        plt.close(fig)
        raise

    if return_fig:
        return fig

    if return_fig_axes:
        return fig, axes

    plt.show()


def _get_plot_dims(signal, ann_samp):
    """
    Figure out the number of plot channels.

    Parameters
    ----------
    signal : 1d or 2d numpy array or list, optional
        The signal or signals to be plotted.  If signal is a
        one-dimensional array, it is assumed to represent a single channel.
        If it is a two-dimensional array, axes 0 and 1 must represent time
        and channel number respectively.  Otherwise it must be a list of
        one-dimensional arrays (one for each channel).
    ann_samp: list, optional
        A list of annotation locations to plot, with each list item
        corresponding to a different channel. List items may be:

        - 1d numpy array, with values representing sample indices. Empty
          arrays are skipped.
        - list, with values representing sample indices. Empty lists
          are skipped.
        - None. For channels in which nothing is to be plotted.

        If `signal` is defined, the annotation locations will be overlaid on
        the signals, with the list index corresponding to the signal channel.
        The length of `annotation` does not have to match the number of
        channels of `signal`.

    Returns
    -------
    sig_len : int
        The signal length (per channel) of the dat file.  Deprecated.
    n_sig : int
        The number of signals contained in the dat file.
    n_annot : int
        The number of annotations contained in the dat file.
    int
        The max between number of signals and annotations.

    """
    # Convert signal to a list if needed
    signal = _expand_channels(signal)

    if signal:
        n_sig = len(signal)
        sig_len = len(signal[0])
        if any(len(s) != sig_len for s in signal):
            sig_len = None
    else:
        sig_len = 0
        n_sig = 0

    if ann_samp is not None:
        n_annot = len(ann_samp)
    else:
        n_annot = 0

    return sig_len, n_sig, n_annot, max(n_sig, n_annot)


def _create_figure(n_subplots, sharex, sharey, figsize):
    """
    Create the plot figure and subplot axes.

    Parameters
    ----------
    n_subplots : int
        The number of subplots to generate.
    figsize : tuple
        The figure's width, height in inches.
    sharex, sharey : bool, optional
        Controls sharing of properties among x (`sharex`) or y (`sharey`) axes.
        If True: x- or y-axis will be shared among all subplots.
        If False, each subplot x- or y-axis will be independent.

    Returns
    -------
    fig : matplotlib plot object
        The entire figure that will hold each subplot.
    axes : list
        The information needed for each subplot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=n_subplots, ncols=1, sharex=sharex, sharey=sharey, figsize=figsize
    )
    if n_subplots == 1:
        axes = [axes]
    return fig, axes


def _plot_signal(
    signal, sig_len, n_sig, fs, time_units, sig_style, axes, sampling_freq=None
):
    """
    Plot signal channels.

    Parameters
    ----------
    signal : 1d or 2d numpy array or list
        The signal or signals to be plotted.  If signal is a
        one-dimensional array, it is assumed to represent a single channel.
        If it is a two-dimensional array, axes 0 and 1 must represent time
        and channel number respectively.  Otherwise it must be a list of
        one-dimensional arrays (one for each channel).
    sig_len : int
        The signal length (per channel) of the dat file.  Deprecated.
    n_sig : int
        The number of signals contained in the dat file.
    fs : float
        The sampling frequency of the record.
    time_units : str
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    sig_style : list
        A list of strings, specifying the style of the matplotlib plot
        for each signal channel. The list length should match the number
        of signal channels. If the list has a length of 1, the style
        will be used for all channels.
    axes : list
        The information needed for each subplot.
    sampling_freq : number or sequence, optional
        The sampling frequency or frequencies of the signals.  If this is a
        list, it must have the same length as the number of channels.  If
        unspecified, defaults to `fs`.

    Returns
    -------
    N/A

    """
    # Convert signal to a list if needed
    signal = _expand_channels(signal)
    if n_sig == 0:
        return

    # Extend signal style if necessary
    if len(sig_style) == 1:
        sig_style = n_sig * sig_style

    # Convert sampling_freq to a list if needed
    sampling_freq = _get_sampling_freq(sampling_freq, n_sig, fs)

    tarrays = {}

    # Plot the signals
    for ch in range(n_sig):
        ch_len = len(signal[ch])
        ch_freq = sampling_freq[ch]

        # Figure out time indices
        try:
            t = tarrays[ch_len, ch_freq]
        except KeyError:
            if time_units == "samples":
                t = np.linspace(0, ch_len - 1, ch_len)
            else:
                downsample_factor = {
                    "seconds": ch_freq,
                    "minutes": ch_freq * 60,
                    "hours": ch_freq * 3600,
                }
                t = np.linspace(0, ch_len - 1, ch_len)
                t /= downsample_factor[time_units]
            tarrays[ch_len, ch_freq] = t

        axes[ch].plot(t, signal[ch], sig_style[ch], zorder=3)


def _plot_annotation(
    ann_samp,
    n_annot,
    ann_sym,
    signal,
    n_sig,
    fs,
    time_units,
    ann_style,
    axes,
    sampling_freq=None,
    ann_freq=None,
):
    """
    Plot annotations, possibly overlaid on signals.
    ann_samp, n_annot, ann_sym, signal, n_sig, fs, time_units, ann_style, axes

    Parameters
    ----------
    ann_samp : list
        The values of the annotation locations.
    n_annot : int
        The number of annotations contained in the dat file.
    ann_sym : list
        The values of the annotation symbol locations.
    signal : 1d or 2d numpy array or list
        The signal or signals to be plotted.  If signal is a
        one-dimensional array, it is assumed to represent a single channel.
        If it is a two-dimensional array, axes 0 and 1 must represent time
        and channel number respectively.  Otherwise it must be a list of
        one-dimensional arrays (one for each channel).
    n_sig : int
        The number of signals contained in the dat file.
    fs : float
        The sampling frequency of the record.
    time_units : str
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    sig_style : list, optional
        A list of strings, specifying the style of the matplotlib plot
        for each signal channel. The list length should match the number
        of signal channels. If the list has a length of 1, the style
        will be used for all channels.
    axes : list
        The information needed for each subplot.
    sampling_freq : number or sequence, optional
        The sampling frequency or frequencies of the signals.  If this is a
        list, it must have the same length as the number of channels.  If
        unspecified, defaults to `fs`.
    ann_freq : number or sequence, optional
        The sampling frequency or frequencies of the annotations.  If this
        is a list, it must have the same length as `ann_samp`.  If
        unspecified, defaults to `fs`.

    Returns
    -------
    N/A

    """
    # Convert signal to a list if needed
    signal = _expand_channels(signal)

    # Extend annotation style if necessary
    if len(ann_style) == 1:
        ann_style = n_annot * ann_style

    # Convert sampling_freq and ann_freq to lists if needed
    sampling_freq = _get_sampling_freq(sampling_freq, n_sig, fs)
    ann_freq = _get_ann_freq(ann_freq, n_annot, fs)

    # Plot the annotations
    for ch in range(n_annot):
        afreq = ann_freq[ch]
        if ch < n_sig:
            sfreq = sampling_freq[ch]
        else:
            sfreq = afreq

        # Figure out downsample factor for time indices
        if time_units == "samples":
            if afreq is None and sfreq is None:
                downsample_factor = 1
            else:
                downsample_factor = afreq / sfreq
        else:
            downsample_factor = {
                "seconds": float(afreq),
                "minutes": float(afreq) * 60,
                "hours": float(afreq) * 3600,
            }[time_units]

        if ann_samp[ch] is not None and len(ann_samp[ch]):
            # Figure out the y values to plot on a channel basis

            # 1 dimensional signals
            try:
                if n_sig > ch:
                    if sfreq == afreq:
                        index = ann_samp[ch]
                    else:
                        index = (sfreq / afreq * ann_samp[ch]).astype("int")
                    y = signal[ch][index]
                else:
                    y = np.zeros(len(ann_samp[ch]))
            except IndexError:
                raise Exception(
                    "IndexError: try setting shift_samps=True in "
                    'the "rdann" function?'
                )

            axes[ch].plot(
                ann_samp[ch] / downsample_factor, y, ann_style[ch], zorder=4
            )

            # Plot the annotation symbols if any
            if ann_sym is not None and ann_sym[ch] is not None:
                for i, s in enumerate(ann_sym[ch]):
                    axes[ch].annotate(
                        s, (ann_samp[ch][i] / downsample_factor, y[i])
                    )


def _plot_ecg_grids(ecg_grids, fs, units, time_units, axes, sampling_freq=None):
    """
    Add ECG grids to the axes.

    Parameters
    ----------
    ecg_grids : list, str
        Whether to add a grid for all the plots ('all') or not.
    fs : float
        The sampling frequency of the record.
    units : list
        The units used for plotting each signal.
    time_units : str
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    axes : list
        The information needed for each subplot.
    sampling_freq : number or sequence, optional
        The sampling frequency or frequencies of the signals.  If this is a
        list, it must have the same length as the number of channels.  If
        unspecified, defaults to `fs`.

    Returns
    -------
    N/A

    """
    if ecg_grids == "all":
        ecg_grids = range(0, len(axes))

    # Convert sampling_freq to a list if needed
    sampling_freq = _get_sampling_freq(sampling_freq, len(axes), fs)

    for ch in ecg_grids:
        # Get the initial plot limits
        auto_xlims = axes[ch].get_xlim()
        auto_ylims = axes[ch].get_ylim()

        (
            major_ticks_x,
            minor_ticks_x,
            major_ticks_y,
            minor_ticks_y,
        ) = _calc_ecg_grids(
            auto_ylims[0],
            auto_ylims[1],
            units[ch],
            sampling_freq[ch],
            auto_xlims[1],
            time_units,
        )

        min_x, max_x = np.min(minor_ticks_x), np.max(minor_ticks_x)
        min_y, max_y = np.min(minor_ticks_y), np.max(minor_ticks_y)

        for tick in minor_ticks_x:
            axes[ch].plot(
                [tick, tick], [min_y, max_y], c="#ededed", marker="|", zorder=1
            )
        for tick in major_ticks_x:
            axes[ch].plot(
                [tick, tick], [min_y, max_y], c="#bababa", marker="|", zorder=2
            )
        for tick in minor_ticks_y:
            axes[ch].plot(
                [min_x, max_x], [tick, tick], c="#ededed", marker="_", zorder=1
            )
        for tick in major_ticks_y:
            axes[ch].plot(
                [min_x, max_x], [tick, tick], c="#bababa", marker="_", zorder=2
            )

        # Plotting the lines changes the graph. Set the limits back
        axes[ch].set_xlim(auto_xlims)
        axes[ch].set_ylim(auto_ylims)


def _calc_ecg_grids(minsig, maxsig, sig_units, fs, maxt, time_units):
    """
    Calculate tick intervals for ECG grids.

    - 5mm 0.2s major grids, 0.04s minor grids.
    - 0.5mV major grids, 0.125 minor grids.

    10 mm is equal to 1mV in voltage.

    Parameters
    ----------
    minsig : float
        The min value of the signal.
    maxsig : float
        The max value of the signal.
    sig_units : list
        The units used for plotting each signal.
    fs : float
        The sampling frequency of the signal.
    maxt : float
        The max time of the signal.
    time_units : str
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.

    Returns
    -------
    major_ticks_x : ndarray
        The locations of the major ticks on the x-axis.
    minor_ticks_x : ndarray
        The locations of the minor ticks on the x-axis.
    major_ticks_y : ndarray
        The locations of the major ticks on the y-axis.
    minor_ticks_y : ndarray
        The locations of the minor ticks on the y-axis.

    """
    # Get the grid interval of the x axis
    if time_units == "samples":
        majorx = 0.2 * fs
        minorx = 0.04 * fs
    elif time_units == "seconds":
        majorx = 0.2
        minorx = 0.04
    elif time_units == "minutes":
        majorx = 0.2 / 60
        minorx = 0.04 / 60
    elif time_units == "hours":
        majorx = 0.2 / 3600
        minorx = 0.04 / 3600

    # Get the grid interval of the y axis
    if sig_units.lower() == "uv":
        majory = 500
        minory = 125
    elif sig_units.lower() == "mv":
        majory = 0.5
        minory = 0.125
    elif sig_units.lower() == "v":
        majory = 0.0005
        minory = 0.000125
    else:
        raise ValueError("Signal units must be uV, mV, or V to plot ECG grids.")

    major_ticks_x = np.arange(0, upround(maxt, majorx) + 0.0001, majorx)
    minor_ticks_x = np.arange(0, upround(maxt, majorx) + 0.0001, minorx)

    major_ticks_y = np.arange(
        downround(minsig, majory), upround(maxsig, majory) + 0.0001, majory
    )
    minor_ticks_y = np.arange(
        downround(minsig, majory), upround(maxsig, majory) + 0.0001, minory
    )

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)


def _label_figure(
    axes, n_subplots, time_units, sig_name, sig_units, xlabel, ylabel, title
):
    """
    Add title, and axes labels.

    Parameters
    ----------
    axes : list
        The information needed for each subplot.
    n_subplots : int
        The number of subplots to generate.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    sig_name : list, optional
        A list of strings specifying the signal names. Used with `sig_units`
        to form y labels, if `ylabel` is not set.
    sig_units : list, optional
        A list of strings specifying the units of each signal channel. Used
        with `sig_name` to form y labels, if `ylabel` is not set. This
        parameter is required for plotting ECG grids.
    xlabel : list, optional
         A list of strings specifying the final x labels to be used. If this
         option is present, no 'time/'`time_units` is used.
    ylabel : list, optional
        A list of strings specifying the final y labels. If this option is
        present, `sig_name` and `sig_units` will not be used for labels.
    title : str, optional
        The title of the graph.

    Returns
    -------
    N/A

    """
    if title:
        axes[0].set_title(title)

    # Determine x label
    # Explicit labels take precedence if present. Otherwise, construct labels
    # using signal time units
    if not xlabel:
        axes[-1].set_xlabel("/".join(["time", time_units[:-1]]))
    else:
        for ch in range(n_subplots):
            axes[ch].set_xlabel(xlabel[ch])

    # Determine y label
    # Explicit labels take precedence if present. Otherwise, construct labels
    # using signal names and units
    if not ylabel:
        ylabel = []
        # Set default channel and signal names if needed
        if not sig_name:
            sig_name = ["ch_" + str(i) for i in range(n_subplots)]
        if not sig_units:
            sig_units = n_subplots * ["NU"]

        ylabel = ["/".join(pair) for pair in zip(sig_name, sig_units)]

        # If there are annotations with channels outside of signal range
        # put placeholders
        n_missing_labels = n_subplots - len(ylabel)
        if n_missing_labels:
            ylabel = ylabel + [
                "ch_%d/NU" % i for i in range(len(ylabel), n_subplots)
            ]

    for ch in range(n_subplots):
        axes[ch].set_ylabel(ylabel[ch])


def plot_wfdb(
    record=None,
    annotation=None,
    plot_sym=False,
    time_units="seconds",
    title=None,
    sig_style=[""],
    ann_style=["r*"],
    ecg_grids=[],
    figsize=None,
    return_fig=False,
    sharex="auto",
):
    """
    Subplot individual channels of a WFDB record and/or annotation.

    This function implements the base functionality of the `plot_items`
    function, while allowing direct input of WFDB objects.

    If the record object is input, the function will extract from it:
      - signal values, from the `e_p_signal`, `e_d_signal`, `p_signal`, or
        `d_signal` attribute (in that order of priority.)
      - frame frequency, from the `fs` attribute
      - signal names, from the `sig_name` attribute
      - signal units, from the `units` attribute

    If the annotation object is input, the function will extract from it:
      - sample locations, from the `sample` attribute
      - symbols, from the `symbol` attribute
      - the annotation channels, from the `chan` attribute
      - the sampling frequency, from the `fs` attribute if present, and if fs
        was not already extracted from the `record` argument.

    Parameters
    ----------
    record : WFDB Record, optional
        The Record object to be plotted.
    annotation : WFDB Annotation, optional
        The Annotation object to be plotted.
    plot_sym : bool, optional
        Whether to plot the annotation symbols on the graph.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds',
        'minutes', and 'hours'.
    title : str, optional
        The title of the graph.
    sig_style : list, optional
        A list of strings, specifying the style of the matplotlib plot
        for each signal channel. The list length should match the number
        of signal channels. If the list has a length of 1, the style
        will be used for all channels.
    ann_style : list, optional
        A list of strings, specifying the style of the matplotlib plot
        for each annotation channel. The list length should match the
        number of annotation channels. If the list has a length of 1,
        the style will be used for all channels.
    ecg_grids : list, optional
        A list of integers specifying channels in which to plot ECG grids. May
        also be set to 'all' for all channels. Major grids at 0.5mV, and minor
        grids at 0.125mV. All channels to be plotted with grids must have
        `sig_units` equal to 'uV', 'mV', or 'V'.
    figsize : tuple, optional
        Tuple pair specifying the width, and height of the figure. It is the
        'figsize' argument passed into matplotlib.pyplot's `figure` function.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.
    sharex : bool or 'auto', optional
        Whether the X axis should be shared between all subplots.  If set
        to True, then all signals will be aligned with each other.  If set
        to False, then each subplot can be panned/zoomed independently.  If
        set to 'auto' (default), then the X axis will be shared unless
        record is multi-frequency and the time units are set to 'samples'.

    Returns
    -------
    figure : matplotlib figure, optional
        The matplotlib figure generated. Only returned if the 'return_fig'
        option is set to True.

    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/100', sampto=3000)
    >>> annotation = wfdb.rdann('sample-data/100', 'atr', sampto=3000)

    >>> wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True
                       time_units='seconds', title='MIT-BIH Record 100',
                       figsize=(10,4), ecg_grids='all')

    """
    (
        signal,
        ann_samp,
        ann_sym,
        fs,
        ylabel,
        record_name,
        sig_units,
    ) = _get_wfdb_plot_items(
        record=record, annotation=annotation, plot_sym=plot_sym
    )

    if record:
        if record.e_p_signal is not None or record.e_d_signal is not None:
            sampling_freq = [spf * record.fs for spf in record.samps_per_frame]
        else:
            sampling_freq = record.fs
    else:
        sampling_freq = None

    if sharex == "auto":
        # If the sampling frequencies are equal, or if we are using
        # hours/minutes/seconds as the time unit, then share the X axes so
        # that the channels are synchronized.  If time units are 'samples'
        # and sampling frequencies are not uniform, then sharing X axes
        # doesn't work and may even be misleading.
        if (
            time_units == "samples"
            and isinstance(sampling_freq, list)
            and any(f != sampling_freq[0] for f in sampling_freq)
        ):
            sharex = False
        else:
            sharex = True

    if annotation and annotation.fs is not None:
        ann_freq = annotation.fs
    elif record:
        ann_freq = record.fs
    else:
        ann_freq = None

    return plot_items(
        signal=signal,
        ann_samp=ann_samp,
        ann_sym=ann_sym,
        fs=fs,
        time_units=time_units,
        ylabel=ylabel,
        title=(title or record_name),
        sig_style=sig_style,
        sig_units=sig_units,
        ann_style=ann_style,
        ecg_grids=ecg_grids,
        figsize=figsize,
        return_fig=return_fig,
        sampling_freq=sampling_freq,
        ann_freq=ann_freq,
        sharex=sharex,
    )


def _get_wfdb_plot_items(record, annotation, plot_sym):
    """
    Get items to plot from WFDB objects.

    Parameters
    ----------
    record : WFDB Record
        The Record object to be plotted
    annotation : WFDB Annotation
        The Annotation object to be plotted
    plot_sym : bool
        Whether to plot the annotation symbols on the graph.

    Returns
    -------
    signal : 1d or 2d numpy array
        The uniformly sampled signal to be plotted. If signal.ndim is 1, it is
        assumed to be a one channel signal. If it is 2, axes 0 and 1, must
        represent time and channel number respectively.
    ann_samp: list
        A list of annotation locations to plot, with each list item
        corresponding to a different channel. List items may be:

        - 1d numpy array, with values representing sample indices. Empty
          arrays are skipped.
        - list, with values representing sample indices. Empty lists
          are skipped.
        - None. For channels in which nothing is to be plotted.

        If `signal` is defined, the annotation locations will be overlaid on
        the signals, with the list index corresponding to the signal channel.
        The length of `annotation` does not have to match the number of
        channels of `signal`.
    ann_sym: list
        A list of annotation symbols to plot, with each list item
        corresponding to a different channel. List items should be lists of
        strings. The symbols are plotted over the corresponding `ann_samp`
        index locations.
    fs : int, float
        The sampling frequency of the signals and/or annotations. Used to
        calculate time intervals if `time_units` is not 'samples'. Also
        required for plotting ECG grids.
    ylabel : list
        A list of strings specifying the final y labels. If this option is
        present, `sig_name` and `sig_units` will not be used for labels.
    record_name : str
        The string name of the WFDB record to be written (without any file
        extensions). Must not contain any "." since this would indicate an
        EDF file which is not compatible at this point.
    sig_units : list
        A list of strings specifying the units of each signal channel. Used
        with `sig_name` to form y labels, if `ylabel` is not set. This
        parameter is required for plotting ECG grids.

    """
    # Get record attributes
    if record:
        if record.e_p_signal is not None:
            signal = record.e_p_signal
            n_sig = len(signal)
            physical = True
        elif record.e_d_signal is not None:
            signal = record.e_d_signal
            n_sig = len(signal)
            physical = False
        elif record.p_signal is not None:
            signal = record.p_signal
            n_sig = signal.shape[1]
            physical = True
        elif record.d_signal is not None:
            signal = record.d_signal
            n_sig = signal.shape[1]
            physical = False
        else:
            raise ValueError("The record has no signal to plot")

        fs = record.fs
        sig_name = [str(s) for s in record.sig_name]
        if physical:
            sig_units = [str(s) for s in record.units]
        else:
            sig_units = ["adu"] * n_sig
        record_name = "Record: %s" % record.record_name
        ylabel = ["/".join(pair) for pair in zip(sig_name, sig_units)]
    else:
        signal = fs = ylabel = record_name = sig_units = None

    # Get annotation attributes
    if annotation:
        # Get channels
        ann_chans = set(annotation.chan)
        n_ann_chans = max(ann_chans) + 1

        # Indices for each channel
        chan_inds = n_ann_chans * [np.empty(0, dtype="int")]

        for chan in ann_chans:
            chan_inds[chan] = np.where(annotation.chan == chan)[0]

        ann_samp = [annotation.sample[ci] for ci in chan_inds]

        if plot_sym:
            ann_sym = n_ann_chans * [None]
            for ch in ann_chans:
                ann_sym[ch] = [annotation.symbol[ci] for ci in chan_inds[ch]]
        else:
            ann_sym = None

        # Try to get fs from annotation if not already in record
        if fs is None:
            fs = annotation.fs

        record_name = record_name or annotation.record_name
    else:
        ann_samp = None
        ann_sym = None

    # Cleaning: remove empty channels and set labels and styles.

    # Wrangle together the signal and annotation channels if necessary
    if record and annotation:
        # There may be instances in which the annotation `chan`
        # attribute has non-overlapping channels with the signal.
        # In this case, omit empty middle channels. This function should
        # already process labels and arrangements before passing into
        # `plot_items`
        sig_chans = set(range(n_sig))
        all_chans = sorted(sig_chans.union(ann_chans))

        # Need to update ylabels and annotation values
        if sig_chans != all_chans:
            compact_ann_samp = []
            if plot_sym:
                compact_ann_sym = []
            else:
                compact_ann_sym = None
            ylabel = []
            for ch in all_chans:  # ie. 0, 1, 9
                if ch in ann_chans:
                    compact_ann_samp.append(ann_samp[ch])
                    if plot_sym:
                        compact_ann_sym.append(ann_sym[ch])
                if ch in sig_chans:
                    ylabel.append("".join([sig_name[ch], sig_units[ch]]))
                else:
                    ylabel.append("ch_%d/NU" % ch)
            ann_samp = compact_ann_samp
            ann_sym = compact_ann_sym
        # Signals encompass annotations
        else:
            ylabel = ["/".join(pair) for pair in zip(sig_name, sig_units)]

    # Remove any empty middle channels from annotations
    elif annotation:
        ann_samp = [a for a in ann_samp if a.size]
        if ann_sym is not None:
            ann_sym = [a for a in ann_sym if a]
        ylabel = ["ch_%d/NU" % ch for ch in ann_chans]

    return signal, ann_samp, ann_sym, fs, ylabel, record_name, sig_units


def plot_all_records(directory=""):
    """
    Plot all WFDB records in a directory (by finding header files), one at
    a time, until the 'enter' key is pressed.

    Parameters
    ----------
    directory : str, optional
        The directory in which to search for WFDB records. Defaults to
        current working directory.

    Returns
    -------
    N/A

    """
    directory = directory or os.getcwd()

    headers = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    headers = [f for f in headers if f.endswith(".hea")]

    records = [h.split(".hea")[0] for h in headers]
    records.sort()

    for record_name in records:
        record = rdrecord(os.path.join(directory, record_name))

        plot_wfdb(record, title="Record - %s" % record.record_name)
        input("Press enter to continue...")
