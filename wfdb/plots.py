import numpy as np
import sys
import matplotlib.pyplot as plt
from . import records
from . import _headers
from . import annotations


# Plot a WFDB record's signals
def plotrec(record=None, signals = None, fields = None, title = None, timeunits='samples', returnfig = False): 
    
    # Figure out which arguments to use to plot
    signals, fields = getplotitems(record, signals, fields)
    
    # Check the validity of items used to make the plot
    # Return any ammendments and the x axis time/sample value
    fields, t = checkplotitems(signals, fields, title, timeunits)
    
    siglen, nsig = signals.shape
    
    # Create the plot  
    fig=plt.figure()
    
    for ch in range(nsig):
        # Plot signal channel
        plt.subplot(100*nsig+11+ch)
        plt.plot(t, signals[:,ch]) 
        
        if (title is not None) and (ch==0):
            plt.title(title)
            
        # Axis Labels
        if timeunits == 'samples':
            plt.xlabel('index/sample')
        else:
            plt.xlabel('time/'+timeunits[:-1])
            
        if fields["signame"][ch] is not None:
            chanlabel=fields["signame"][ch]
        else:
            chanlabel='channel'
        if fields["units"][ch] is not None:
            unitlabel=fields["units"][ch]
        else:
            unitlabel='NU'
        plt.ylabel(chanlabel+"/"+unitlabel)
        
    plt.show(fig)
    
    # Return the figure if requested
    if returnfig:
        return fig

#  Figure out which arguments to use to plot
def getplotitems(record, signals, fields):
    
    if record is None:
        if signals is None:
            sys.exit('Either record or signals must be present. No input signal to plot.')
        # Use the signals array
    else:
        # Use the record object
        if signals is None:
            # If it is a MultiRecord, convert it into single
            if type(record) == records.MultiRecord:
                record = record.multi_to_single()
            
            # Need to ensure p_signals is present
            if record.p_signals is None:
                sys.exit('The p_signals field must be present in the record object to plot')
                
            signals = record.p_signals
            fields = {}
            for field in ['fs', 'units', 'signame']:
                fields[field] = getattr(record, field)
        # Both are present
        else:
            sys.exit("Only one of 'record' or 'signals' can be input. Cannot plot both items.")
            
    return signals, fields
    
def checkplotitems(signals, fields, title, timeunits):
    
    # signals
    if type(signals) != np.ndarray or signals.ndim != 2:
        sys.exit("The plotted signal must be a 2d numpy array")
    
    siglen, nsig = signals.shape
    
    # fs and timeunits
    allowedtimes = ['samples', 'seconds', 'minutes', 'hours']
    if timeunits not in allowedtimes:
        print("The 'timeunits' field must be one of the following: ", allowedtimes)
        sys.exit()
    # Get x axis values. fs must be valid when plotting time
    if timeunits == 'samples':
        t = np.linspace(0, siglen-1, siglen)
    else:
        if type(fields['fs']) not in _headers.floattypes:
            sys.exit("The 'fs' field must be a number")
        
        if timeunits == 'seconds':
            t = np.linspace(0, siglen-1, siglen)/fs
        elif timeunits == 'minutes':
            t = np.linspace(0, siglen-1, siglen)/fs/60
        else:
            t = np.linspace(0, siglen-1, siglen)/fs/3600
    
    # units
    if fields['units'] is None:
        fields['units'] = ['NU']*nsig
    else:
        if type(fields['units']) != list or len(fields['units'])!= nsig:
            sys.exit("The 'units' parameter must be a list of strings with length equal to the number of signal channels")
        for ch in range(nsig):
            if fields['units'][ch] is None:
                fields['units'][ch] = 'NU'
    # signame
    if fields['signame'] is None:
        fields['signame'] = ['ch'+str(ch) for ch in range(1, nsig+1)] 
    else:
        if type(fields['signame']) != list or len(fields['signame'])!= nsig:
            sys.exit("The 'signame' parameter must be a list of strings with length equal to the number of signal channels")
                
    # title
    if title is not None and type(title) != str:
        sys.exit("The 'title' field must be a string")
    
    return fields, t


# Plot the sample locations of a WFDB annotation
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






def plotreco(sig, fields, annsamp=None, annch=[0], title=None, plottime=1): 
    """ Subplot and label each channel of a WFDB signal. Also subplot annotation locations on selected channels if present.  
    
    Usage: plotsigs(sig, fields, annsamp=[], annch=[0], title=[], plottime=1): 
    
    Input arguments: 
    - sig (required): An nxm numpy array containing the signal to be plotted - first output argument of rdsamp 
    - fields (required): A dictionary of metadata about the record - second output argument of rdsamp
    - annsamp (optional): A 1d numpy array of annotation locations to be plotted on top of selected channels - first output argument of readannot.rdann().
    - annch (default=[0]): A list of channels on which to plot the annotations. 
    - title (optional): A string containing the title of the graph.
    - plottime (default=1) - Flag that specifies whether to plot the x axis as time (1) or samples (0). Defaults to samples if the input fields dictionary does not contain a value for fs.
    
    """
    
    if len(fields)==3: # Multi-segment variable layout. Get the layout header fields.
        fields=fields[1]
    elif len(fields)==2: # Multi-segment fixed layout. Get the header fields of the first segment. 
        fields=fields[1][0]
    
    if (not fields["fs"])|(plottime==0): # x axis is index
        plottime=0
        t=np.array(range(0,sig.shape[0]))
        if annsamp!=[]:
            annplott=annsamp
    else: # x axis is time
        t=np.array(range(0,sig.shape[0]))/fields["fs"]
        if annsamp!=[]:
            annplott=annsamp/fields["fs"]
        
    f1=plt.figure()
    for ch in range(0, sig.shape[1]):
        plt.subplot(100*sig.shape[1]+11+ch)
        plt.plot(t, sig[:,ch]) # Plot signal channel
        if (annsamp!=[]) & (ch in annch): # If there are annotations to plot and the channel is specified
            plt.plot(annplott, sig[annsamp, ch], 'r+') # Plot annotations
        
        if (title!=[])&(ch==0):
            plt.title(title)
            
        # Axis Labels
        if plottime:
            plt.xlabel('time/s')
        else:
            plt.xlabel('index/sample')
        if fields["signame"][ch]:
            chanlabel=fields["signame"][ch]
        else:
            chanlabel='channel'
        if fields["units"][ch]:
            unitlabel=fields["units"][ch]
        else:
            unitlabel='NU'
        plt.ylabel(chanlabel+"/"+unitlabel)
        
    plt.show(f1)