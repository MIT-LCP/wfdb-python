import numpy as np
import matplotlib.pyplot as plt


def plotrec(sig, fields, annsamp=None, annch=[0], title=None, plottime=1): 
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

def plotann():
    print('on it')