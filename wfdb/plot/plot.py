import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

from wfdb.io.record import Record, rdrecord
from wfdb.io._header import float_types
from wfdb.io._signal import downround, upround
from wfdb.io.annotation import Annotation


def plot_items(signal=None, ann_samp=None, ann_sym=None, fs=None,
               time_units='samples', sig_name=None, sig_units=None,
               xlabel=None, ylabel=None, title=None, sig_style=[''],
               ann_style=['r*'], ecg_grids=[], figsize=None,
               return_fig=False, return_fig_axes=False):
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
    figsize : tuple, optional
        Tuple pair specifying the width, and height of the figure. It is the
        'figsize' argument passed into matplotlib.pyplot's `figure` function.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.

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
    # First, make sure that xlabel and ylabel inputs are valid
    if xlabel:
        if len(xlabel) != signal.shape[1]:
            raise Exception('The length of the xlabel must be the same as the '
                            'signal: {} values'.format(signal.shape[1]))

    if ylabel:
        if len(ylabel) != n_subplots:
            raise Exception('The length of the ylabel must be the same as the '
                            'signal: {} values'.format(n_subplots))

    label_figure(axes, n_subplots, time_units, sig_name, sig_units,
                 xlabel, ylabel, title)

    if return_fig:
        return fig

    if return_fig_axes:
        return fig, axes

    plt.show()


def get_plot_dims(signal, ann_samp):
    """
    Figure out the number of plot channels.

    Parameters
    ----------
    signal : 1d or 2d numpy array, optional
        The uniformly sampled signal to be plotted. If signal.ndim is 1, it is
        assumed to be a one channel signal. If it is 2, axes 0 and 1, must
        represent time and channel number respectively.
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
        The signal length (per channel) of the dat file.
    n_sig : int
        The number of signals contained in the dat file.
    n_annot : int
        The number of annotations contained in the dat file.
    int
        The max between number of signals and annotations.

    """
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
    """
    Create the plot figure and subplot axes.

    Parameters
    ----------
    n_subplots : int
        The number of subplots to generate.
    figsize : tuple
        The figure's width, height in inches. 

    Returns
    -------
    fig : matplotlib plot object
        The entire figure that will hold each subplot.
    axes : list
        The information needed for each subplot.

    """
    fig = plt.figure(figsize=figsize)
    axes = []

    for i in range(n_subplots):
        axes.append(fig.add_subplot(n_subplots, 1, i+1))

    return fig, axes


def plot_signal(signal, sig_len, n_sig, fs, time_units, sig_style, axes):
    """
    Plot signal channels.

    Parameters
    ----------
    signal : ndarray
        Tranformed expanded signal into uniform signal.
    sig_len : int
        The signal length (per channel) of the dat file.
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

    Returns
    -------
    N/A

    """
    # Extend signal style if necessary
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
    signal : ndarray
        Tranformed expanded signal into uniform signal.
    sig_len : int
        The signal length (per channel) of the dat file.
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

    Returns
    -------
    N/A

    """
    # Extend annotation style if necessary
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
        if ann_samp[ch] is not None and len(ann_samp[ch]):
            # Figure out the y values to plot on a channel basis

            # 1 dimensional signals
            try:
                if n_sig > ch:
                    if signal.ndim == 1:
                        y = signal[ann_samp[ch]]
                    else:
                        y = signal[ann_samp[ch], ch]
                else:
                    y = np.zeros(len(ann_samp[ch]))
            except IndexError:
                raise Exception('IndexError: try setting shift_samps=True in ' 
                                'the "rdann" function?')

            axes[ch].plot(ann_samp[ch] / downsample_factor, y, ann_style[ch])

            # Plot the annotation symbols if any
            if ann_sym is not None and ann_sym[ch] is not None:
                for i, s in enumerate(ann_sym[ch]):
                    axes[ch].annotate(s, (ann_samp[ch][i] / downsample_factor,
                                          y[i]))


def plot_ecg_grids(ecg_grids, fs, units, time_units, axes):
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

    Returns
    -------
    N/A

    """
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
        The sampling frequency of the record.
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


def label_figure(axes, n_subplots, time_units, sig_name, sig_units,
                 xlabel, ylabel, title):
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
        axes[-1].set_xlabel('/'.join(['time', time_units[:-1]]))
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
            sig_name = ['ch_'+str(i) for i in range(n_subplots)]
        if not sig_units:
            sig_units = n_subplots * ['NU']

        ylabel = ['/'.join(pair) for pair in zip(sig_name, sig_units)]

        # If there are annotations with channels outside of signal range
        # put placeholders
        n_missing_labels = n_subplots - len(ylabel)
        if n_missing_labels:
            ylabel = ylabel + ['ch_%d/NU' % i for i in range(len(ylabel),
                                                             n_subplots)]

    for ch in range(n_subplots):
        axes[ch].set_ylabel(ylabel[ch])


def plot_wfdb(record=None, annotation=None, plot_sym=False,
              time_units='samples', title=None, sig_style=[''],
              ann_style=['r*'], ecg_grids=[], figsize=None, return_fig=False):
    """
    Subplot individual channels of a WFDB record and/or annotation.

    This function implements the base functionality of the `plot_items`
    function, while allowing direct input of WFDB objects.

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
    (signal, ann_samp, ann_sym, fs,
        ylabel, record_name, sig_units) = get_wfdb_plot_items(record=record,
                                                              annotation=annotation,
                                                              plot_sym=plot_sym)

    return plot_items(signal=signal, ann_samp=ann_samp, ann_sym=ann_sym, fs=fs,
                      time_units=time_units, ylabel=ylabel,
                      title=(title or record_name),
                      sig_style=sig_style, sig_units=sig_units,
                      ann_style=ann_style, ecg_grids=ecg_grids,
                      figsize=figsize, return_fig=return_fig)


def get_wfdb_plot_items(record, annotation, plot_sym):
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
        if record.p_signal is not None:
            signal = record.p_signal
        elif record.d_signal is not None:
            signal = record.d_signal
        else:
            raise ValueError('The record has no signal to plot')

        fs = record.fs
        sig_name = [str(s) for s in record.sig_name]
        sig_units = [str(s) for s in record.units]
        record_name = 'Record: %s' % record.record_name
        ylabel = ['/'.join(pair) for pair in zip(sig_name, sig_units)]
    else:
        signal = fs = ylabel = record_name = sig_units = None

    # Get annotation attributes
    if annotation:
        # Get channels
        ann_chans = set(annotation.chan)
        n_ann_chans = max(ann_chans) + 1

        # Indices for each channel
        chan_inds = n_ann_chans * [np.empty(0, dtype='int')]

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
        sig_chans = set(range(signal.shape[1]))
        all_chans = sorted(sig_chans.union(ann_chans))

        # Need to update ylabels and annotation values
        if sig_chans != all_chans:
            compact_ann_samp = []
            if plot_sym:
                compact_ann_sym = []
            else:
                compact_ann_sym = None
            ylabel = []
            for ch in all_chans: # ie. 0, 1, 9
                if ch in ann_chans:
                    compact_ann_samp.append(ann_samp[ch])
                    if plot_sym:
                        compact_ann_sym.append(ann_sym[ch])
                if ch in sig_chans:
                    ylabel.append(''.join([sig_name[ch], sig_units[ch]]))
                else:
                    ylabel.append('ch_%d/NU' % ch)
            ann_samp = compact_ann_samp
            ann_sym = compact_ann_sym
        # Signals encompass annotations
        else:
            ylabel = ['/'.join(pair) for pair in zip(sig_name, sig_units)]

    # Remove any empty middle channels from annotations
    elif annotation:
        ann_samp = [a for a in ann_samp if a.size]
        if ann_sym is not None:
            ann_sym = [a for a in ann_sym if a]
        ylabel = ['ch_%d/NU' % ch for ch in ann_chans]

    return signal, ann_samp, ann_sym, fs, ylabel, record_name, sig_units


def plot_all_records(directory=''):
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

    headers = [f for f in os.listdir(directory) if os.path.isfile(
        os.path.join(directory, f))]
    headers = [f for f in headers if f.endswith('.hea')]

    records = [h.split('.hea')[0] for h in headers]
    records.sort()

    for record_name in records:
        record = rdrecord(os.path.join(directory, record_name))

        plot_wfdb(record, title='Record - %s' % record.record_name)
        input('Press enter to continue...')
