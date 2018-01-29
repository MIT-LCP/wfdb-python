import matplotlib.pyplot as plt
import numpy as np
import os

from ..io.record import Record, rdrecord
from ..io._header import float_types
from ..io._signal import downround, upround
from ..io.annotation import Annotation


import pdb


def plot_items(signal=None, ann_samp=None, ann_sym=None, fs=None,
               time_units='samples', sig_units=None, chan_name=None,
               title=None, sig_style=[''], ann_style=['r*'], ecg_grids=[],
               figsize=None, return_fig=False):
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
    sig_units : list, optional
        A list of strings specifying the units of each signal channel. This
        parameter is required for plotting ecg grids.
    chan_name : list, optional
        A list of strings specifying the channel names. 
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

    wfdb.plot_items(signal=record.p_signal,
                    annotation=[ann.sample, ann.sample],
                    title='Record 100 from MIT-BIH Arrhythmia Database',
                    time_units='seconds', figsize=(10,4), ecg_grids='all')

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
    label_figure(axes, n_subplots, sig_units, time_units, chan_name, title)
    
    plt.show(fig)

    if return_fig:
        return fig

def get_plot_dims(signal, ann_samp):
    "Figure out the number of signal/annotation channels"
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
        downsample_factor = {'seconds':fs, 'minutes':fs*60, 'hours':fs*3600}
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
        downsample_factor = {'seconds':float(fs), 'minutes':float(fs)*60, 'hours':float(fs)*3600}[time_units]
    
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
                    axes[ch].annotate(s, (ann_samp[ch][i] / downsample_factor, y[i]))


def plot_ecg_grids(ecg_grids, fs, units, time_units, axes):
    "Add ecg grids to the axes"
    if ecg_grids == 'all':
        ecg_grids = range(0, len(axes))
    
    
    for ch in ecg_grids:
        # Get the initial plot limits
        auto_xlims = axes[ch].get_xlim()
        auto_ylims= axes[ch].get_ylim()
        
        major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y = calc_ecg_grids(
                auto_ylims[0], auto_ylims[1], units[ch], fs, auto_xlims[1], time_units)
        
        min_x, max_x = np.min(minor_ticks_x), np.max(minor_ticks_x)
        min_y, max_y = np.min(minor_ticks_y), np.max(minor_ticks_y)
            
        for tick in minor_ticks_x:
            axes[ch].plot([tick, tick], [min_y,  max_y], c='#ededed', marker='|', zorder=1)
        for tick in major_ticks_x:
            axes[ch].plot([tick, tick], [min_y, max_y], c='#bababa', marker='|', zorder=2)
        for tick in minor_ticks_y:
            axes[ch].plot([min_x, max_x], [tick, tick], c='#ededed', marker='_', zorder=1)
        for tick in major_ticks_y:
            axes[ch].plot([min_x, max_x], [tick, tick], c='#bababa', marker='_', zorder=2)

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

    major_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory) + 0.0001, majory)
    minor_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory) + 0.0001, minory)

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)


def label_figure(axes, n_subplots, sig_units, time_units, chan_name, title):
    "Add title, and axes labels"
    if title:
        axes[0].set_title(title)
    
    # Set default channel and signal names if needed
    if not chan_name:
        chan_name = ['ch_'+str(i) for i in range(n_subplots)]
    if not sig_units:
        sig_units = n_subplots * ['NU']
    
    for ch in range(n_subplots):
        axes[ch].set_ylabel('/'.join([chan_name[ch], sig_units[ch]]))
    
    axes[-1].set_xlabel('/'.join(['time', time_units[:-1]]))
    
    

def plot_wfdb(record=None, annotation=None, time_units='samples',
              plot_physical=True, plot_ann_sym=False, 


    title=None, sig_style=[''], ann_style=['r*'],
    



            figsize=None,
            return_fig=False, ecg_grids=[]):
    """ 
    Subplot individual channels of a wfdb record and/or annotation.
    
    This function implements the base functionality of the `plot_items`
    function, while allowing direct input of wfdb objects.

    Parameters
    ----------
    record : wfdb Record
        The Record object, whose p_signal or d_signal attribute is to be
        plotted.
    title : str, optional
        The title of the graph.
    annotation : list, or wfdb Annotation, optional
        One of the following:
        1. Same as the `annotation` argument of the `plot_items` function. A
           list of annotation locations to plot, with each list item
           corresponding to a different channel. List items may be:
           - 1d numpy array, with values representing sample indices
           - list, with values representing sample indices
           - None. For channels in which nothing is to be plotted.
           If `signal` is defined, the annotation locations will be overlaid on
           the signals, with the list index corresponding to the signal
           channel. The length of `annotation` does not have to match the
           number of channels of `signal`.
        2. A wfdb Annotation object. The `chan` attribute decides the channel
           number of each sample.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'. If this option is not 'samples', either the `record` or
        `annotation` argument must be a valid `fs` attribute.
    sig_style : str, or list, optional
        String, or list of strings, specifying the styling of the matplotlib
        plot for the signals. If it is a string, each channel will have the same
        style. If it is a list, each channel's style will correspond to the list
        element. ie. sig_style=['r','b','k']
    ann_style : str, or list, optional
        String, or list of strings, specifying the styling of the matplotlib
        plot for the annotation. If it is a string, each channel will have the
        same style. If it is a list, each channel's style will correspond to the
        list element.
    plot_ann_sym : bool, optional
        Whether to plot the annotation symbols at their locations.
    figsize : tuple, optional
        Tuple pair specifying the width, and height of the figure. It is the
        'figsize' argument passed into matplotlib.pyplot's `figure` function.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.
    ecg_grids : list, optional
        List of integers specifying channels in which to plot ecg grids. May be
        set to [] for no channels, or 'all' for all channels. Major grids at
        0.5mV, and minor grids at 0.125mV. All channels to be plotted with grids
        must have units equal to 'uV', 'mV', or 'V'.
    
    Returns
    -------
    figure : matplotlib figure, optional
        The matplotlib figure generated. Only returned if the 'return_fig'
        option is set to True.

    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/100', sampto=3000)
    >>> annotation = wfdb.rdann('sample-data/100', 'atr', sampto=3000)

    wfdb.plot_record(record, annotation=annotation, title='Record 100 from MIT-BIH Arrhythmia Database', 
                 time_units='seconds', figsize=(10,4), ecg_grids='all')

    """



def plot_all_records(directory=os.getcwd()):
    """
    Plot all wfdb records in a directory (by finding header files), one at
    a time, until the 'enter' key is pressed.

    Parameters
    ----------
    directory : str, optional
        The directory in which to search for WFDB records.
    
    """
    file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file_list = [f for f in file_list if f.endswith('.hea')]
    record_list = [f.split('.hea')[0] for f in file_list]
    record_list.sort()

    for record_name in record_list:
        record = records.rdrecord(record_name)

        plot_wfdb(record, title='Record: %s' % record.recordname)
        input('Press enter to continue...')
