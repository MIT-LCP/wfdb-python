import numpy as np
from wfdb import readsignal
import matplotlib.pyplot as plt

# Plot signals and labels output by rdsamp
def plotsigs(sig, fields, plottime=1):
    if (not fields["fs"])|(plottime==0):
        plottime=0
        t=np.array(range(0,sig.shape[0]))
    else:
        t=np.array(range(0,sig.shape[0]))/fields["fs"]
        
    plt.figure(1)
    for ch in range(0, sig.shape[1]):
        plt.subplot(100*sig.shape[1]+11+ch)
        plt.plot(t, sig[:,ch])
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
    plt.show()