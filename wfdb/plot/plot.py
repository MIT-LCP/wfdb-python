import matplotlib.pyplot as plt
import numpy as np
import os

from ..io.record import Record, rdrecord
from ..io._header import float_types
from ..io._signal import downround, upround
from ..io.annotation import Annotation


def plot_items(signal=None, ann_samp=None, ann_sym=None, fs=None,
               time_units='samples', sig_name=None, sig_units=None,
               ylabel=None, title=None, sig_style=[''], ann_style=['r*'],
               ecg_grids=[], figsize=None, return_fig=False):
    """
    Subplot individual channels of signals and/or annotations.

    Parameters
    ----------
    signal : 1d or 2d numpy array, optional
        The uniformly sampled signal to be plotted. If signal.ndim is 1, it is
        assumed to be a one channel signal. If it is 2, axes 0 and 1, must
        represent time and channel number respectively.
    ann_samp: list, optional
        A list of annotation locations to plot, with each list item
        corresponding to a different channel. List items may be:
        - 1d numpy array, with values representing sample indices
        - list, with values representing sample indices
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
    fs : int or float, optional
        The sampling frequency of the signals and/or annotations. Used to
        calculate time intervals if `time_units` is not 'samples'. Also
        required for plotting ecg grids.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    sig_name : list, optional
        A list of strings specifying the signal names. Used with `sig_units`
        to form y labels, if `ylabel` is not set.
    sig_units : list, optional
        A list of strings specifying the units of each signal channel. Used
        with `sig_name` to form y labels, if `ylabel` is not set. This
        parameter is required for plotting ecg grids.
    ylabel : list, optional
        A list of strings specifying the final y labels. If this option is
        present, `sig_name` and `sig_units` will not be used for labels.
    title : str, optional
        The title of the graph.
    sig_style : list, optional
        A list of strings, specifying the style of the matplotlib plot for each
        signal channel. If the list has a length of 1, the style will be used
        for all channels.
    ann_style : list, optional
        A list of strings, specifying the style of the matplotlib plot for each
        annotation channel. If the list has a length of 1, the style will be
        used for all channels.
    ecg_grids : list, optional
        A list of integers specifying channels in which to plot ecg grids. May
        also be set to 'all' for all channels. Major grids at 0.5mV, and minor
        grids at 0.125mV. All channels to be plotted with grids must have
        `sig_units` equal to 'uV', 'mV', or 'V'.
    figsize : tuple, optional
        Tuple pair specifying the width, and height of the figure. It is the
        'figsize' argument passed into matplotlib.pyplot's `figure` function.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.

    Returns
    -------
    figure : matplotlib figure, optional
        The matplotlib figure generated. Only returned if the 'return_fig'
        parameter is set to True.

    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/100', sampto=3000)
    >>> ann = wfdb.rdann('sample-data/100', 'atr', sampto=3000)

    >>> wfdb.plot_items(signal=record.p_signal,
                        annotation=[ann.sample, ann.sample],
                        title='MIT-BIH Record 100', time_units='seconds',
                        figsize=(10,4), ecg_grids='all')

    """

    # Figure out number of subplots required
    sig_len, n_sig, n_annot, n_subplots = get_plot_dims(signal, ann_samp)

    # Create figure
    fig, axes = create_figure(n_subplots, figsize)

    if signal is not None:
        plot_signal(signal, sig_len, n_sig, fs, time_units, sig_style, axes)

    if ann_samp is not None:
        plot_annotation(ann_samp, n_annot, ann_sym, signal, n_sig, fs,
                        time_units, ann_style, axes)

    if ecg_grids:
        plot_ecg_grids(ecg_grids, fs, sig_units, time_units, axes)

    # Add title and axis labels.
    label_figure(axes, n_subplots, time_units, sig_name, sig_units, ylabel,
                 title)

    plt.show(fig)

    if return_fig:
        return fig

def get_plot_dims(signal, ann_samp):
    "Figure out the number of plot channels"
    if signal is not None:
        if signal.ndim == 1:
            sig_len = len(signal)
            n_sig = 1
        else:
            sig_len = signal.shape[0]
            n_sig = signal.shape[1]
    else:
        sig_len = 0
        n_sig = 0

    if ann_samp is not None:
        n_annot = len(ann_samp)
    else:
        n_annot = 0

    return sig_len, n_sig, n_annot, max(n_sig, n_annot)


def create_figure(n_subplots, figsize):
    "Create the plot figure and subplot axes"
    fig = plt.figure(figsize=figsize)
    axes = []

    for i in range(n_subplots):
        axes.append(fig.add_subplot(n_subplots, 1, i+1))

    return fig, axes


def plot_signal(signal, sig_len, n_sig, fs, time_units, sig_style, axes):
    "Plot signal channels"

    # Extend signal style if necesary
    if len(sig_style) == 1:
        sig_style = n_sig * sig_style

    # Figure out time indices
    if time_units == 'samples':
        t = np.linspace(0, sig_len-1, sig_len)
    else:
        downsample_factor = {'seconds':fs, 'minutes':fs * 60,
                             'hours':fs * 3600}
        t = np.linspace(0, sig_len-1, sig_len) / downsample_factor[time_units]

    # Plot the signals
    if signal.ndim == 1:
        axes[0].plot(t, signal, sig_style[0], zorder=3)
    else:
        for ch in range(n_sig):
            axes[ch].plot(t, signal[:,ch], sig_style[ch], zorder=3)


def plot_annotation(ann_samp, n_annot, ann_sym, signal, n_sig, fs, time_units,
                    ann_style, axes):
    "Plot annotations, possibly overlaid on signals"
    # Extend annotation style if necesary
    if len(ann_style) == 1:
        ann_style = n_annot * ann_style

    # Figure out downsample factor for time indices
    if time_units == 'samples':
        downsample_factor = 1
    else:
        downsample_factor = {'seconds':float(fs), 'minutes':float(fs)*60,
                             'hours':float(fs)*3600}[time_units]

    # Plot the annotations
    for ch in range(n_annot):
        if ann_samp[ch] is not None:
            # Figure out the y values to plot on a channel basis

            # 1 dimensional signals
            if n_sig > ch:
                if signal.ndim == 1:
                    y = signal[ann_samp[ch]]
                else:
                    y = signal[ann_samp[ch], ch]
            else:
                y = np.zeros(len(ann_samp[ch]))

            axes[ch].plot(ann_samp[ch] / downsample_factor, y, ann_style[ch])

            # Plot the annotation symbols if any
            if ann_sym is not None and ann_sym[ch] is not None:
                for i, s in enumerate(ann_sym[ch]):
                    axes[ch].annotate(s, (ann_samp[ch][i] / downsample_factor,
                                          y[i]))


def plot_ecg_grids(ecg_grids, fs, units, time_units, axes):
    "Add ecg grids to the axes"
    if ecg_grids == 'all':
        ecg_grids = range(0, len(axes))


    for ch in ecg_grids:
        # Get the initial plot limits
        auto_xlims = axes[ch].get_xlim()
        auto_ylims= axes[ch].get_ylim()

        (major_ticks_x, minor_ticks_x, major_ticks_y,
            minor_ticks_y) = calc_ecg_grids(auto_ylims[0], auto_ylims[1],
                                            units[ch], fs, auto_xlims[1],
                                            time_units)

        min_x, max_x = np.min(minor_ticks_x), np.max(minor_ticks_x)
        min_y, max_y = np.min(minor_ticks_y), np.max(minor_ticks_y)

        for tick in minor_ticks_x:
            axes[ch].plot([tick, tick], [min_y,  max_y], c='#ededed',
                          marker='|', zorder=1)
        for tick in major_ticks_x:
            axes[ch].plot([tick, tick], [min_y, max_y], c='#bababa',
                          marker='|', zorder=2)
        for tick in minor_ticks_y:
            axes[ch].plot([min_x, max_x], [tick, tick], c='#ededed',
                          marker='_', zorder=1)
        for tick in major_ticks_y:
            axes[ch].plot([min_x, max_x], [tick, tick], c='#bababa',
                          marker='_', zorder=2)

        # Plotting the lines changes the graph. Set the limits back
        axes[ch].set_xlim(auto_xlims)
        axes[ch].set_ylim(auto_ylims)

def calc_ecg_grids(minsig, maxsig, sig_units, fs, maxt, time_units):
    """
    Calculate tick intervals for ecg grids

    - 5mm 0.2s major grids, 0.04s minor grids
    - 0.5mV major grids, 0.125 minor grids

    10 mm is equal to 1mV in voltage.
    """
    # Get the grid interval of the x axis
    if time_units == 'samples':
        majorx = 0.2 * fs
        minorx = 0.04 * fs
    elif time_units == 'seconds':
        majorx = 0.2
        minorx = 0.04
    elif time_units == 'minutes':
        majorx = 0.2 / 60
        minorx = 0.04/60
    elif time_units == 'hours':
        majorx = 0.2 / 3600
        minorx = 0.04 / 3600

    # Get the grid interval of the y axis
    if sig_units.lower()=='uv':
        majory = 500
        minory = 125
    elif sig_units.lower()=='mv':
        majory = 0.5
        minory = 0.125
    elif sig_units.lower()=='v':
        majory = 0.0005
        minory = 0.000125
    else:
        raise ValueError('Signal units must be uV, mV, or V to plot ECG grids.')

    major_ticks_x = np.arange(0, upround(maxt, majorx) + 0.0001, majorx)
    minor_ticks_x = np.arange(0, upround(maxt, majorx) + 0.0001, minorx)

    major_ticks_y = np.arange(downround(minsig, majory),
                              upround(maxsig, majory) + 0.0001, majory)
    minor_ticks_y = np.arange(downround(minsig, majory),
                              upround(maxsig, majory) + 0.0001, minory)

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)


def label_figure(axes, n_subplots, time_units, sig_name, sig_units, ylabel,
                 title):
    "Add title, and axes labels"
    if title:
        axes[0].set_title(title)

    # Determine y label
    # Explicit labels take precedence if present. Otherwise, construct labels
    # using signal names and units
    if not ylabel:
        ylabel = []
        # Set default channel and signal names if needed
        if not sig_name:
            sig_name = ['ch_'+str(i) for i in range(n_subplots)]
        if not sig_units:
            sig_units = n_subplots * ['NU']

        ylabel = ['/'.join(pair) for pair in zip(sig_name, sig_units)]

    for ch in range(n_subplots):
        axes[ch].set_ylabel(ylabel[ch])

    axes[-1].set_xlabel('/'.join(['time', time_units[:-1]]))


def plot_wfdb(record=None, annotation=None, plot_sym=False,
              time_units='samples', title=None, sig_style=[''],
              ann_style=['r*'], ecg_grids=[], figsize=None, return_fig=False):
    """
    Subplot individual channels of a wfdb record and/or annotation.

    This function implements the base functionality of the `plot_items`
    function, while allowing direct input of wfdb objects.

    If the record object is input, the function will extract from it:
      - signal values, from the `p_signal` (priority) or `d_signal` attribute
      - sampling frequency, from the `fs` attribute
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
    record : wfdb Record, optional
        The Record object to be plotted
    annotation : wfdb Annotation, optional
        The Annotation object to be plotted
    plot_sym : bool, optional
        Whether to plot the annotation symbols on the graph.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    title : str, optional
        The title of the graph.
    sig_style : list, optional
        A list of strings, specifying the style of the matplotlib plot for each
        signal channel. If the list has a length of 1, the style will be used
        for all channels.
    ann_style : list, optional
        A list of strings, specifying the style of the matplotlib plot for each
        annotation channel. If the list has a length of 1, the style will be
        used for all channels.
    ecg_grids : list, optional
        A list of integers specifying channels in which to plot ecg grids. May
        also be set to 'all' for all channels. Major grids at 0.5mV, and minor
        grids at 0.125mV. All channels to be plotted with grids must have
        `sig_units` equal to 'uV', 'mV', or 'V'.
    figsize : tuple, optional
        Tuple pair specifying the width, and height of the figure. It is the
        'figsize' argument passed into matplotlib.pyplot's `figure` function.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.

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
    (signal, ann_samp, ann_sym, fs, sig_name,
        sig_units) = get_wfdb_plot_items(record=record, annotation=annotation,
                                         plot_sym=plot_sym)

    return plot_items(signal=signal, ann_samp=ann_samp, ann_sym=ann_sym, fs=fs,
                      time_units=time_units, sig_name=sig_name,
                      sig_units=sig_units, title=title, sig_style=sig_style,
                      ann_style=ann_style, ecg_grids=ecg_grids,
                      figsize=figsize, return_fig=return_fig)


def get_wfdb_plot_items(record, annotation, plot_sym):
    """
    Get items to plot from wfdb objects
    """
    # Get record attributes
    if record:
        if record.p_signal is not None:
            signal = record.p_signal
        elif record.d_signal is not None:
            signal = record.d_signal
        else:
            raise ValueError('The record has no signal to plot')

        fs = record.fs
        sig_name = record.sig_name
        sig_units = record.units
    else:
        signal, fs, sig_name, sig_units = 4 * [None]

    # Get annotation attributes
    if annotation:
        # Get channels
        all_chans = set(annotation.chan)

        n_chans = max(all_chans) + 1

        # Just one channel. Place content in one list index.
        # if len(all_chans) == 1:
        #     ann_samp = annotation.chan[0]*[None] + [annotation.sample]
        #     if plot_sym:
        #         ann_sym = annotation.chan[0]*[None] + [annotation.symbol]
        #     else:
        #         ann_sym = None
        # # Split annotations by channel
        # else:

        # Indices for each channel
        chan_inds = n_chans * [np.empty(0)]

        for chan in all_chans:
            chan_inds[chan] = np.where(annotation.chan == chan)[0]

        ann_samp = [annotation.sample[ci] for ci in chan_inds]

        if plot_sym:
            ann_sym = n_chans * [None]
            for ch in all_chans:
                ann_sym[ch] = [annotation.symbol[ci] for ci in chan_inds[ch]]
        else:
            ann_sym = None

        # Try to get fs from annotation if not already in record
        if fs is None:
            fs = annotation.fs
    else:
        ann_samp = None
        ann_sym = None

    return signal, ann_samp, ann_sym, fs, sig_name, sig_units


def plot_all_records(directory=''):
    """
    Plot all wfdb records in a directory (by finding header files), one at
    a time, until the 'enter' key is pressed.

    Parameters
    ----------
    directory : str, optional
        The directory in which to search for WFDB records. Defaults to
        current working directory.

    """
    directory = directory or os.getcwd()

    headers = [f for f in os.listdir(directory) if os.path.isfile(
        os.path.join(directory, f))]
    headers = [f for f in headers if f.endswith('.hea')]

    records = [h.split('.hea')[0] for h in headers]
    records.sort()

    for record_name in records:
        record = rdrecord(os.path.join(directory, record_name))

        plot_wfdb(record, title='Record - %s' % record.record_name)
        input('Press enter to continue...')
