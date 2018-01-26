import matplotlib.pyplot as plt
import numpy as np
import os

from ..io.record import Record, rdrecord
from ..io._header import float_types
from ..io._signal import downround, upround
from ..io.annotation import Annotation


def plot_items(signal=None, annotation=None, fs=None, sig_units=None,
               time_units='samples', chan_name=None, title=None,
               sig_style=[''], ann_style=['r*'], ecg_grids=[], figsize=None,
               return_fig=False):
    """ 
    Subplot individual channels of signals and/or annotations.

    Parameters
    ----------
    signal : 1d or 2d numpy array, optional
        The uniformly sampled signal to be plotted. If signal.ndim is 1, it is
        assumed to be a one channel signal. If it is 2, axes 0 and 1, must
        represent time and channel number respectively.
    annotation: list, optional
        A list of annotation locations to plot, with each list item
        corresponding to a different channel. List items may be:
        - 1d numpy array, with values representing sample indices
        - list, with values representing sample indices
        - None. For channels in which nothing is to be plotted.
        If `signal` is defined, the annotation locations will be overlaid on
        the signals, with the list index corresponding to the signal channel.
        The length of `annotation` does not have to match the number of
        channels of `signal`.
    fs : int or float, optional
        The sampling frequency of the signals and/or annotations. Used to
        calculate time intervals if `time_units` is not 'samples'.
    sig_units : list, optional
        List of strings specifying the units of each signal channel.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    chan_name : list, optional
        A list of strings, representing the channel names. This 
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
        List of integers specifying channels in which to plot ecg grids. May be
        set to [] for no channels, or 'all' for all channels. Major grids at
        0.5mV, and minor grids at 0.125mV. All channels to be plotted with
        grids must have units equal to 'uV', 'mV', or 'V'.
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
    
    Notes
    -----
    At least one of `signal` or `annotation` must be defined, or there will be
    nothing to plot.
    
    Examples
    --------
    >>> record = wfdb.rdrecord('sample-data/100', sampto=3000)
    >>> annotation = wfdb.rdann('sample-data/100', 'atr', sampto=3000)

    wfdb.plot_record(record, annotation=annotation, title='Record 100 from MIT-BIH Arrhythmia Database', 
                 time_units='seconds', figsize=(10,4), ecg_grids='all')


    Add ann_sym option?

    """

    # Figure out number of subplots required

    # Create figure
    
    n_sig, n_annot, n_subplots = get_plot_dims(signal, annotation)
    
    fig, axes = create_figure(n_subplots, figsize)

    if signal:
        plot_signal(signal, sig_len, n_sig, fs, time_units, sig_style, axes)

    if annotation:
        plot_annotation(annotation, n_annot, signal, n_sig, fs, time_units, ann_style, axes)

    if ecg_grids:
        plot_ecg_grids(ecg_grids, fs, units, time_units, axes)

    # Add title and axis labels.
    label_figure()
    
    plt.show(fig)

    if return_fig:
        return figure

def get_plot_dims(signal, annotation):
    "Figure out the number of signal/annotation channels"
    if signal is not None:
        if sig.ndim == 1:
            sig_len = len(signal)
            n_sig = 1
        else:
            sig_len = signal.shape[0]
            n_sig = signal.shape[1]
    else:
        sig_len = 0
        n_sig = 0
        
    if annotation:
        n_annot = len(annotation)
    else:
        n_annot = 0

    return sig_len, n_sig, n_annot, max(n_sig, n_annot)


def create_figure(n_subplots, figsize)
    "Create the plot figure and subplot axes"
    axes = []

    for i in range(n_subplots):
        ax.append(fig.add_subplot(n_subplots, 1, i+1))
    
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
        axes[0].plot(t, signal, sig_style[ch], zorder=3)
    else:
        for ch in range(n_sig):
            axes[ch].plot(t, signal[:,ch], sig_style[ch], zorder=3) 
    
    
def plot_annotation(annotation, n_annot, signal, n_sig, fs, time_units, ann_style, axes):
    "Plot annotations, possibly overlaid on signals"
    
    # Figure out downsample factor for time indices
    if time_units == 'samples':
        downsample_factor = 1
    else:
        downsample_factor = {'seconds':float(fs), 'minutes':float(fs)*60, 'hours':float(fs)*3600}[time_units]
    
    # Plot the annotations    
    for ch in range(n_annot):
        if annotation[ch] is not None: 
            # Figure out the y values to plot on a channel basis
            if n_sig > ch:
                y = signal[annotation[ch], ch]
            else:
                y = np.zeros(len(annotation[ch]))
            
            axes[ch].plot(annotation[ch] / downsample_factor, ann_style[ch])


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

def calc_ecg_grids(minsig, maxsig, units, fs, maxt, time_units):
    """
    Calculate tick intervals for ecg grids
    
    - 5mm 0.2s major grids, 0.04s minor grids
    - 0.5mV major grids, 0.125 minor grids 
    
    10 mm is equal to 1mV in voltage.
    """
    # Get the grid interval of the x axis
    if time_units == 'samples':
        majorx = 0.2*fs
        minorx = 0.04*fs
    elif time_units == 'seconds':
        majorx = 0.2
        minorx = 0.04
    elif time_units == 'minutes':
        majorx = 0.2/60
        minorx = 0.04/60
    elif time_units == 'hours':
        majorx = 0.2/3600
        minorx = 0.04/3600

    # Get the grid interval of the y axis
    if units.lower()=='uv':
        majory = 500
        minory = 125
    elif units.lower()=='mv':
        majory = 0.5
        minory = 0.125
    elif units.lower()=='v':
        majory = 0.0005
        minory = 0.000125
    else:
        raise ValueError('Signal units must be uV, mV, or V to plot ECG grids.')

    major_ticks_x = np.arange(0, _upround(maxt, majorx)+0.0001, majorx)
    minor_ticks_x = np.arange(0, _upround(maxt, majorx)+0.0001, minorx)

    major_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory)+0.0001, majory)
    minor_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory)+0.0001, minory)

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)


def label_figure(n_subplots, sig_units, time_units, chan_name, title):
    
    if title:
        axes[0].set_title(title)
    
    if not chan_name:
        chan_name = ['ch_'+str(i) for i in range(n_subplots)]
    
    if not sig_units:
        sig_units = n_subplots * ['NU']
    
    for ch in range(n_subplots):
        
        pass
        
        
    

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


    a, b, c = extract_wfdb_items()

    # Check the validity of items used to make the plot
    # Return the x axis time values to plot for the record (and annotation if any)
    t, tann, annplot = check_plot_items(record, title, annotation, time_units, sig_style, ann_style)

    siglen, n_sig = record.p_signal.shape
    
    # Expand list styles
    if isinstance(sig_style, str):
        sig_style = [sig_style]*record.n_sig
    else:
        if len(sig_style) < record.n_sig:
            sig_style = sig_style+['']*(record.n_sig-len(sig_style))
    if isinstance(ann_style, str):
        ann_style = [ann_style]*record.n_sig
    else:
        if len(ann_style) < record.n_sig:
            ann_style = ann_style+['r*']*(record.n_sig-len(ann_style))

    # Expand ecg grid channels
    if ecg_grids == 'all':
        ecg_grids = range(0, record.n_sig)

    # Create the plot  
    fig=plt.figure(figsize=figsize)
    
    for ch in range(n_sig):
        # Plot signal channel
        ax = fig.add_subplot(n_sig, 1, ch+1)
        ax.plot(t, record.p_signal[:,ch], sig_style[ch], zorder=3) 
        
        if (title is not None) and (ch==0):
            plt.title(title)
            
        # Plot annotation if specified
        if annplot[ch] is not None:
            ax.plot(tann[ch], record.p_signal[annplot[ch], ch], ann_style[ch])
            # Plot the annotation symbols if specified
            if plot_ann_sym:
                for i, s in enumerate(annotation.symbol):
                    ax.annotate(s, (tann[ch][i], record.p_signal[annplot[ch], ch][i]))

        # Axis Labels
        if time_units == 'samples':
            plt.xlabel('index/sample')
        else:
            plt.xlabel('time/'+time_units[:-1])
            
        if record.sig_name[ch] is not None:
            chanlabel=record.sig_name[ch]
        else:
            chanlabel='channel'
        if record.units[ch] is not None:
            unitlabel=record.units[ch]
        else:
            unitlabel='NU'
        plt.ylabel(chanlabel+"/"+unitlabel)

        # Show standard ecg grids if specified.
        if ch in ecg_grids:
            
            auto_xlims = ax.get_xlim()
            auto_ylims= ax.get_ylim()

            major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y = calc_ecg_grids(
                auto_ylims[0], auto_ylims[1], record.units[ch], record.fs, auto_xlims[1], time_units)

            min_x, max_x = np.min(minor_ticks_x), np.max(minor_ticks_x)
            min_y, max_y = np.min(minor_ticks_y), np.max(minor_ticks_y)

            for tick in minor_ticks_x:
                ax.plot([tick, tick], [min_y,  max_y], c='#ededed', marker='|', zorder=1)
            for tick in major_ticks_x:
                ax.plot([tick, tick], [min_y, max_y], c='#bababa', marker='|', zorder=2)
            for tick in minor_ticks_y:
                ax.plot([min_x, max_x], [tick, tick], c='#ededed', marker='_', zorder=1)
            for tick in major_ticks_y:
                ax.plot([min_x, max_x], [tick, tick], c='#bababa', marker='_', zorder=2)

            # Plotting the lines changes the graph. Set the limits back
            ax.set_xlim(auto_xlims)
            ax.set_ylim(auto_ylims)

    plt.show(fig)
    
    # Return the figure if requested
    if return_fig:
        return fig

# Calculate tick intervals for ecg grids
def calc_ecg_grids(minsig, maxsig, units, fs, maxt, time_units):

    # 5mm 0.2s major grids, 0.04s minor grids
    # 0.5mV major grids, 0.125 minor grids 
    # 10 mm is equal to 1mV in voltage.
    
    # Get the grid interval of the x axis
    if time_units == 'samples':
        majorx = 0.2*fs
        minorx = 0.04*fs
    elif time_units == 'seconds':
        majorx = 0.2
        minorx = 0.04
    elif time_units == 'minutes':
        majorx = 0.2/60
        minorx = 0.04/60
    elif time_units == 'hours':
        majorx = 0.2/3600
        minorx = 0.04/3600

    # Get the grid interval of the y axis
    if units.lower()=='uv':
        majory = 500
        minory = 125
    elif units.lower()=='mv':
        majory = 0.5
        minory = 0.125
    elif units.lower()=='v':
        majory = 0.0005
        minory = 0.000125
    else:
        raise ValueError('Signal units must be uV, mV, or V to plot the ECG grid.')


    major_ticks_x = np.arange(0, _upround(maxt, majorx)+0.0001, majorx)
    minor_ticks_x = np.arange(0, _upround(maxt, majorx)+0.0001, minorx)

    major_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory)+0.0001, majory)
    minor_ticks_y = np.arange(downround(minsig, majory), upround(maxsig, majory)+0.0001, minory)

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)


def check_plot_items(record, title, annotation, time_units, sig_style, ann_style):
    """
    Check the validity of items used to make the plot
    Return the x axis time values to plot for the record (and time and values for annotation if any)
    """

    # signals
    if not isinstance(record, Record):
        raise TypeError("The 'record' argument must be a valid wfdb.Record object")
    if not isinstance(record.p_signal, np.ndarray) or record.p_signal.ndim != 2:
        raise TypeError("The plotted signal 'record.p_signal' must be a 2d numpy array")
    
    siglen, n_sig = record.p_signal.shape

    # fs and time_units
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if time_units not in allowedtimes:
        raise ValueError("The 'time_units' field must be one of the following: ", allowedtimes)
    # Get x axis values. fs must be valid when plotting time
    if time_units == 'samples':
        t = np.linspace(0, siglen-1, siglen)
    else:
        if not isinstance(record.fs, float_types):
            raise TypeError("The 'fs' field must be a number")
        
        if time_units == 'seconds':
            t = np.linspace(0, siglen-1, siglen)/record.fs
        elif time_units == 'minutes':
            t = np.linspace(0, siglen-1, siglen)/record.fs/60
        else:
            t = np.linspace(0, siglen-1, siglen)/record.fs/3600
    
    # units
    if record.units is None:
        record.units = ['NU']*n_sig
    else:
        if not isinstance(record.units, list) or len(record.units)!= n_sig:
            raise ValueError("The 'units' parameter must be a list of strings with length equal to the number of signal channels")
        for ch in range(n_sig):
            if record.units[ch] is None:
                record.units[ch] = 'NU'

    # sig_name
    if record.sig_name is None:
        record.sig_name = ['ch'+str(ch) for ch in range(1, n_sig+1)] 
    else:
        if not isinstance(record.sig_name, list) or len(record.sig_name)!= n_sig:
            raise ValueError("The 'sig_name' parameter must be a list of strings, with length equal to the number of signal channels")
    
    # title
    if title is not None and not isinstance(title, str):
        raise TypeError("The 'title' field must be a string")
    
    # signal line style
    if isinstance(sig_style, str):
        pass
    elif isinstance(sig_style, list):
        if len(sig_style) > record.n_sig:
            raise ValueError("The 'sig_style' list cannot have more elements than the number of record channels")
    else:
        raise TypeError("The 'sig_style' field must be a string or a list of strings")

    # annotation plot style
    if isinstance(ann_style, str):
        pass
    elif isinstance(ann_style, list):
        if len(ann_style) > record.n_sig:
            raise ValueError("The 'ann_style' list cannot have more elements than the number of record channels")
    else:
        raise TypeError("The 'ann_style' field must be a string or a list of strings")


    # Annotations if any
    if annotation is not None:

        # The output list of numpy arrays (or Nones) to plot
        annplot = [None]*record.n_sig

        # Move single channel annotation.to channel 0
        if isinstance(annotation, Annotation):
            annplot[0] = annotation.sample
        elif isinstance(annotation, np.ndarray):
            annplot[0] = annotation
        # Ready list.
        elif isinstance(annotation, list):
            if len(annotation) > record.n_sig:
                raise ValueError("The number of annotation series to plot cannot be more than the number of channels")
            if len(annotation) < record.n_sig:
                annotation = annotation+[None]*(record.n_sig-len(annotation))
            # Check elements. Copy over to new list.
            for ch in range(record.n_sig):
                if isinstance(annotation[ch], Annotation):
                    annplot[ch] = annotation[ch].sample
                elif isinstance(annotation[ch], np.ndarray):
                    annplot[ch] = annotation[ch]
                elif annotation[ch] is None:
                    pass
                else:
                    raise TypeError("The 'annotation' argument must be a wfdb.Annotation object, a numpy array, None, or a list of these data types")
        else:
            raise TypeError("The 'annotation' argument must be a wfdb.Annotation object, a numpy array, None, or a list of these data types")
        
        # The annotation locations to plot
        tann = [None]*record.n_sig

        for ch in range(record.n_sig):
            if annplot[ch] is None:
                continue
            if time_units == 'samples':
                tann[ch] = annplot[ch]
            elif time_units == 'seconds':
                tann[ch] = annplot[ch]/float(record.fs)
            elif time_units == 'minutes':
                tann[ch] = annplot[ch]/float(record.fs)/60
            else:
                tann[ch] = annplot[ch]/float(record.fs)/3600
    else:
        tann = None
        annplot = [None]*record.n_sig

    # tann is the sample values to plot for each annotation series
    return (t, tann, annplot)


def plot_annotation(annotation, title=None, time_units='samples',
                    return_fig=False): 
    """
    Plot sample locations of an Annotation object.
    
    Parameters
    ----------
    annotation : wfdb Annotation 
        Annotation object. The sample attribute locations will be plotted.
    title : str, optional
        The title of the graph.
    time_units : str, optional
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    return_fig : bool, optional
        Whether the figure is to be returned as an output argument.
    
    Returns
    -------
    figure : matplotlib figure, optional
        The matplotlib figure generated. Only returned if the 'return_fig'
        option is set to True.

    Notes
    -----
    The `plot_record` function has more features, and is useful for plotting
    annotation.on top of signal waveforms.

    Examples
    --------
    >>> annotation = wfdb.rdann('sample-data/100', 'atr', sampfrom=100000,
                                sampto=110000)
    >>> annotation.fs = 360
    >>> wfdb.plot_annotation(annotation, time_units='minutes')

    """
    # Check the validity of items used to make the plot
    # Get the x axis annotation values to plot
    plotvals = checkannplotitems(annotation, title, time_units)
    
    # Create the plot
    fig=plt.figure()
    
    plt.plot(plotvals, np.zeros(len(plotvals)), 'r+')
    
    if title is not None:
        plt.title(title)
        
    # Axis Labels
    if time_units == 'samples':
        plt.xlabel('index/sample')
    else:
        plt.xlabel('time/'+time_units[:-1])

    plt.show(fig)
    
    # Return the figure if requested
    if return_fig:
        return fig


# Check the validity of items used to make the annotation plot
def checkannplotitems(annotation, title, time_units):
    
    # signals
    if not isinstance(annotation, Annotation):
        raise TypeError("The 'annotation' field must be a 'wfdb.Annotation' object")

    # fs and time_units
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if time_units not in allowedtimes:
        raise ValueError("The 'time_units' field must be one of the following: ", allowedtimes)

    # fs must be valid when plotting time
    if time_units != 'samples':
        if not isinstance(annotation.fs, float_types):
            raise Exception("In order to plot time units, the Annotation object must have a valid 'fs' attribute")

    # Get x axis values to plot
    if time_units == 'samples':
        plotvals = annotation.sample
    elif time_units == 'seconds':
        plotvals = annotation.sample/float(annotation.fs)
    elif time_units == 'minutes':
        plotvals = annotation.sample/float(annotation.fs*60)
    elif time_units == 'hours':
        plotvals = annotation.sample/float(annotation.fs*3600)

    # title
    if title is not None and not isinstance(title, str):
        raise TypeError("The 'title' field must be a string")
    
    return plotvals


def plot_all_records(directory=os.getcwd()):
    """
    Plot all wfdb records in a directory (by finding header files), one at
    a time, until the 'enter' key is pressed.

    Parameters
    ----------
    directory : str, optional
        The directory in which to search for WFDB records.
    
    """
    filelist = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    filelist = [f for f in filelist if f.endswith('.hea')]
    recordlist = [f.split('.hea')[0] for f in filelist]
    recordlist.sort()

    for record_name in recordlist:
        record = records.rdrecord(record_name)

        plot_record(record, title='Record: %s' % record.recordname)
        input('Press enter to continue...')
