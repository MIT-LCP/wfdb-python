import numpy as np
import sys
import matplotlib.pyplot as plt
from . import records
from . import _headers
from . import annotations

# Plot a WFDB Record's signals
# Optionally, overlay annotation locations
def plotrec(record=None, title = None, annotation = None, annch = [0], timeunits='samples', returnfig = False): 
    
    # Check the validity of items used to make the plot
    # Return the x axis time values to plot for the record (and annotation if any)
    t, tann = checkplotitems(record, title, annotation, annch, timeunits)
    
    siglen, nsig = record.p_signals.shape
    
    # Create the plot  
    fig=plt.figure()
    
    for ch in range(nsig):
        # Plot signal channel
        ax = fig.add_subplot(nsig, 1, ch+1)
        ax.plot(t, record.p_signals[:,ch]) 
        
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
        
    plt.show(fig)
    
    # Return the figure if requested
    if returnfig:
        return fig

# Check the validity of items used to make the plot
# Return the x axis time values to plot for the record (and annotation if any)
def checkplotitems(record, title, annotation, annch, timeunits):
    
    # signals
    if type(record) != records.Record:
        sys.exit("The 'record' argument must be a valid wfdb.Record object")
    if type(record.p_signals) != np.ndarray or record.p_signals.ndim != 2:
        sys.exit("The plotted signal 'record.p_signals' must be a 2d numpy array")
    
    siglen, nsig = record.p_signals.shape

    # fs and timeunits
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if timeunits not in allowedtimes:
        print("The 'timeunits' field must be one of the following: ", allowedtimes)
        sys.exit()
    # Get x axis values. fs must be valid when plotting time
    if timeunits == 'samples':
        t = np.linspace(0, siglen-1, siglen)
    else:
        if type(record.fs) not in _headers.floattypes:
            sys.exit("The 'fs' field must be a number")
        
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
            sys.exit("The 'units' parameter must be a list of strings with length equal to the number of signal channels")
        for ch in range(nsig):
            if record.units[ch] is None:
                record.units[ch] = 'NU'

    # signame
    if record.signame is None:
        record.signame = ['ch'+str(ch) for ch in range(1, nsig+1)] 
    else:
        if type(record.signame) != list or len(record.signame)!= nsig:
            sys.exit("The 'signame' parameter must be a list of strings with length equal to the number of signal channels")
                
    # title
    if title is not None and type(title) != str:
        sys.exit("The 'title' field must be a string")
    
    # Annotations if any
    if annotation is not None:
        if type(annotation) != annotations.Annotation:
            sys.exit("The 'annotation' argument must be a valid wfdb.Annotation object")
        if type(annch)!= list:
            sys.exit("The 'annch' argument must be a list of integers")
        if min(annch)<0 or max(annch)>nsig:
            sys.exit("The elements of 'annch' must be between 0 and the number of record channels")

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
        sys.exit("The 'annotation' field must be a 'wfdb.Annotation' object")

    # fs and timeunits
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if timeunits not in allowedtimes:
        print("The 'timeunits' field must be one of the following: ", allowedtimes)
        sys.exit()

    # fs must be valid when plotting time
    if timeunits != 'samples':
        if type(annotation.fs) not in _headers.floattypes:
            sys.exit("In order to plot time units, the Annotation object must have a valid 'fs' attribute")

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
        sys.exit("The 'title' field must be a string")
    
    return plotvals
