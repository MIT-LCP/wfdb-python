import numpy as np
from wfdb import readsignal
import matplotlib.pyplot as plt

# Plot signals and labels output by rdsamp. Also plot annotation locations. 
def plotsigs(sig, fields, annsamp=[], annch=[0], title=[], plottime=1): 
    
    if (not fields["fs"])|(plottime==0): # x axis is index
        plottime=0
        t=np.array(range(0,sig.shape[0]))
        if annsamp!=[]:
            annplott=annsamp
    else: # x axis is time
        t=np.array(range(0,sig.shape[0]))/fields["fs"]
        if annsamp!=[]>0:
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
       