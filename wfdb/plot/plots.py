import numpy as np
import matplotlib.pyplot as plt
from ..readwrite import records
from ..readwrite import _headers
from ..readwrite import _signals
from ..readwrite import annotations

# Plot a WFDB Record's signals
# Optionally, overlay annotation locations
def plotrec(record=None, title = None, annotation = None, annch = [0], timeunits='samples', figsize=None, returnfig = False, ecggrids=[]): 
    """ Subplot and label each channel of a WFDB Record.
    Optionally, subplot annotation locations over selected channels.
    
    Usage: 
    plotrec(record=None, title = None, annotation = None, annch = [0], timeunits='samples', returnfig=False)
    
    Input arguments:
    - record (required): A wfdb Record object. The p_signals attribute will be plotted.
    - title (default=None): A string containing the title of the graph.
    - annotation (default=None): An Annotation object. The annsamp attribute locations will be overlaid on the signal.
    - annch (default=[0]): A list of channels on which to plot the annotation samples.
    - timeunits (default='samples'): String specifying the x axis unit. 
      Allowed options are: 'samples', 'seconds', 'minutes', and 'hours'.
    - figsize (default=None): Tuple pair specifying the width, and height of the figure. Same as the 'figsize' argument
      passed into matplotlib.pyplot's figure() function.
    - returnfig (default=False): Specifies whether the figure is to be returned as an output argument
    - ecggrids (default=[]): List of integers specifying channels in which to plot ecg grids. May be set to [] for
      no channels, or 'all' for all channels. Major grids at 0.5mV, and minor grids at 0.125mV. All channels to be 
      plotted with grids must have units equal to 'uV', 'mV', or 'V'.
    
    Output argument:
    - figure: The matplotlib figure generated. Only returned if the 'returnfig' option is set to True.

    Example Usage:
    import wfdb
    record = wfdb.rdsamp('sampledata/100', sampto = 3000)
    annotation = wfdb.rdann('sampledata/100', 'atr', sampto = 3000)

    wfdb.plotrec(record, annotation = annotation, title='Record 100 from MIT-BIH Arrhythmia Database', 
                 timeunits = 'seconds', figsize = (10,4), ecggrids = 'all')
    """

    # Check the validity of items used to make the plot
    # Return the x axis time values to plot for the record (and annotation if any)
    t, tann = checkplotitems(record, title, annotation, annch, timeunits)
    
    siglen, nsig = record.p_signals.shape
    
    # Expand ecg grid channels
    if ecggrids == 'all':
        ecggrids = range(0, record.nsig)

    # Create the plot  
    fig=plt.figure(figsize=figsize)
    
    for ch in range(nsig):
        # Plot signal channel
        ax = fig.add_subplot(nsig, 1, ch+1)
        ax.plot(t, record.p_signals[:,ch], zorder=3) 
        
        if (title is not None) and (ch==0):
            plt.title(title)
            
        # Plot annotation if specified
        if annotation is not None and ch in annch:
            ax.plot(tann, record.p_signals[annotation.annsamp, ch], 'r+')

        # Axis Labels
        if timeunits == 'samples':
            plt.xlabel('index/sample')
        else:
            plt.xlabel('time/'+timeunits[:-1])
            
        if record.signame[ch] is not None:
            chanlabel=record.signame[ch]
        else:
            chanlabel='channel'
        if record.units[ch] is not None:
            unitlabel=record.units[ch]
        else:
            unitlabel='NU'
        plt.ylabel(chanlabel+"/"+unitlabel)

        # Show standard ecg grids if specified.
        if ch in ecggrids:
            
            auto_xlims = ax.get_xlim()
            auto_ylims= ax.get_ylim()

            major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y = calc_ecg_grids(
                auto_ylims[0], auto_ylims[1], record.units[ch], record.fs, auto_xlims[1], timeunits)

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
    if returnfig:
        return fig

# Calculate tick intervals for ecg grids
def calc_ecg_grids(minsig, maxsig, units, fs, maxt, timeunits):

    # 5mm 0.2s major grids, 0.04s minor grids
    # 0.5mV major grids, 0.125 minor grids 
    # 10 mm is equal to 1mV in voltage.
    
    # Get the grid interval of the x axis
    if timeunits == 'samples':
        majorx = 0.2*fs
        minorx = 0.04*fs
    elif timeunits == 'seconds':
        majorx = 0.2
        minorx = 0.04
    elif timeunits == 'minutes':
        majorx = 0.2/60
        minorx = 0.04/60
    elif timeunits == 'hours':
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


    major_ticks_x = np.arange(0, _signals.upround(maxt, majorx)+0.0001, majorx)
    minor_ticks_x = np.arange(0, _signals.upround(maxt, majorx)+0.0001, minorx)

    major_ticks_y = np.arange(_signals.downround(minsig, majory), _signals.upround(maxsig, majory)+0.0001, majory)
    minor_ticks_y = np.arange(_signals.downround(minsig, majory), _signals.upround(maxsig, majory)+0.0001, minory)

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)

# Check the validity of items used to make the plot
# Return the x axis time values to plot for the record (and annotation if any)
def checkplotitems(record, title, annotation, annch, timeunits):
    
    # signals
    if type(record) != records.Record:
        raise TypeError("The 'record' argument must be a valid wfdb.Record object")
    if type(record.p_signals) != np.ndarray or record.p_signals.ndim != 2:
        raise TypeError("The plotted signal 'record.p_signals' must be a 2d numpy array")
    
    siglen, nsig = record.p_signals.shape

    # fs and timeunits
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if timeunits not in allowedtimes:
        raise ValueError("The 'timeunits' field must be one of the following: ", allowedtimes)
    # Get x axis values. fs must be valid when plotting time
    if timeunits == 'samples':
        t = np.linspace(0, siglen-1, siglen)
    else:
        if type(record.fs) not in _headers.floattypes:
            raise TypeError("The 'fs' field must be a number")
        
        if timeunits == 'seconds':
            t = np.linspace(0, siglen-1, siglen)/record.fs
        elif timeunits == 'minutes':
            t = np.linspace(0, siglen-1, siglen)/record.fs/60
        else:
            t = np.linspace(0, siglen-1, siglen)/record.fs/3600
    
    # units
    if record.units is None:
        record.units = ['NU']*nsig
    else:
        if type(record.units) != list or len(record.units)!= nsig:
            raise ValueError("The 'units' parameter must be a list of strings with length equal to the number of signal channels")
        for ch in range(nsig):
            if record.units[ch] is None:
                record.units[ch] = 'NU'

    # signame
    if record.signame is None:
        record.signame = ['ch'+str(ch) for ch in range(1, nsig+1)] 
    else:
        if type(record.signame) != list or len(record.signame)!= nsig:
            raise ValueError("The 'signame' parameter must be a list of strings, with length equal to the number of signal channels")
                
    # title
    if title is not None and type(title) != str:
        raise TypeError("The 'title' field must be a string")
    
    # Annotations if any
    if annotation is not None:
        if type(annotation) != annotations.Annotation:
            raise TypeError("The 'annotation' argument must be a valid wfdb.Annotation object")
        if type(annch)!= list:
            raise TypeError("The 'annch' argument must be a list of integers")
        if min(annch)<0 or max(annch)>nsig:
            raise ValueError("The elements of 'annch' must be between 0 and the number of record channels")

        # The annotation locations to plot   
        if timeunits == 'samples':
            tann = annotation.annsamp
        elif timeunits == 'seconds':
            tann = annotation.annsamp/record.fs
        elif timeunits == 'minutes':
            tann = annotation.annsamp/record.fs/60
        else:
            tann = annotation.annsamp/record.fs/3600
    else:
        tann = None

    return (t, tann)



# Plot the sample locations of a WFDB annotation on a new figure
def plotann(annotation, title = None, timeunits = 'samples', returnfig = False): 
    """ Plot sample locations of an Annotation object.
    
    Usage: plotann(annotation, title = None, timeunits = 'samples', returnfig = False)
    
    Input arguments:
    - annotation (required): An Annotation object. The annsamp attribute locations will be overlaid on the signal.
    - title (default=None): A string containing the title of the graph.
    - timeunits (default='samples'): String specifying the x axis unit. 
      Allowed options are: 'samples', 'seconds', 'minutes', and 'hours'.
    - returnfig (default=False): Specifies whether the figure is to be returned as an output argument
    
    Output argument:
    - figure: The matplotlib figure generated. Only returned if the 'returnfig' option is set to True.

    Note: The plotrec function is useful for plotting annotations on top of signal waveforms.

    Example Usage:
    import wfdb
    annotation = wfdb.rdann('sampledata/100', 'atr', sampfrom = 100000, sampto = 110000)
    annotation.fs = 360
    wfdb.plotann(annotation, timeunits = 'minutes')
    """

    # Check the validity of items used to make the plot
    # Get the x axis annotation values to plot
    plotvals = checkannplotitems(annotation, title, timeunits)
    
    # Create the plot
    fig=plt.figure()
    
    plt.plot(plotvals, np.zeros(len(plotvals)), 'r+')
    
    if title is not None:
        plt.title(title)
        
    # Axis Labels
    if timeunits == 'samples':
        plt.xlabel('index/sample')
    else:
        plt.xlabel('time/'+timeunits[:-1])

    plt.show(fig)
    
    # Return the figure if requested
    if returnfig:
        return fig

# Check the validity of items used to make the annotation plot
def checkannplotitems(annotation, title, timeunits):
    
    # signals
    if type(annotation)!= annotations.Annotation:
        raise TypeError("The 'annotation' field must be a 'wfdb.Annotation' object")

    # fs and timeunits
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if timeunits not in allowedtimes:
        raise ValueError("The 'timeunits' field must be one of the following: ", allowedtimes)

    # fs must be valid when plotting time
    if timeunits != 'samples':
        if type(annotation.fs) not in _headers.floattypes:
            raise Exception("In order to plot time units, the Annotation object must have a valid 'fs' attribute")

    # Get x axis values to plot
    if timeunits == 'samples':
        plotvals = annotation.annsamp
    elif timeunits == 'seconds':
        plotvals = annotation.annsamp/annotation.fs
    elif timeunits == 'minutes':
        plotvals = annotation.annsamp/(annotation.fs*60)
    elif timeunits == 'hours':
        plotvals = annotation.annsamp/(annotation.fs*3600)

    # title
    if title is not None and type(title) != str:
        raise TypeError("The 'title' field must be a string")
    
    return plotvals

