import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt
import pdb


class Conf(object):
    """
    Initial configuration object
    """
    def __init__(self, hr=75, qrs_width=0.1):
        self.hr = hr
        self.qrs_width = qrs_width


class CQRS(object):


    def __init__(self, sig, fs, conf):
        self.sig = sig
        self.fs = fs

        self.sig_len = len(sig)

        # These values are in samples
        self.rr = 60 * fs / conf.hr
        self.qrs_width = conf.qrs_width * fs




    def bandpass(self):
        """
        Apply a bandpass filter onto the signal, and save the filtered signal.
        """
        b, a = signal.butter(2, [5*2/self.fs, 20*2/self.fs], 'pass')
        self.sig_f = signal.filtfilt(b, a, self.sig, axis=0)


    def mwi(self):
        """
        Apply moving wave integration with a ricker (Mexican hat) wavelet onto
        the filtered signal, and save the square of the integrated signal.
        
        The width of the hat is equal to the qrs width
        """
        b = signal.ricker(self.qrs_width, 4)
        self.sig_i = signal.filtfilt(b, [1], self.sig_f, axis=0) ** 2


    def learn_params(self):
        """
        Learn the following:
        - rr interval
        - 

        """
        # Find the dominant peaks of the signal
        dominant_peaks = self.sig_i[find_dominant_peaks(self.sig_i,
                                             int(self.qrs_width / 2))]
        # Cluster and use the largest cluster mean as expected param
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dominant_peaks)

        # The mean value of the top cluster
        top_cluster_ind = np.argmax(kmeans.cluster_centers_)

        self.qrs_thresh = max(kmeans.cluster_centers_)
        # The standard deviation of the top cluster
        self.qrs_dev = np.std(dominant_peaks[np.where(kmeans.labels_ == top_cluster_ind)])

        pdb.set_trace()

    def detect(self, sampfrom=0, sampto='end'):
        """
        Detect qrs locations between two samples
        """
        if sampto == 'end':
            sampto = self.sig_len

        self.qrs_inds = []
        self.backsearch_qrs_inds = []

        self.bandpass()
        self.mwi()

        self.learn_params()

        return

        for i in range(self.sig_len):


            if self.is_peak(i):
                pass



        self.qrs_inds = np.array(self.qrs_inds)




def find_dominant_peaks(sig, radius):
    """
    Find all dominant peaks in a signal.
    A sample is a dominant peak if it is the largest value within the
    <radius> samples on its left and right.
    """
    peak_inds = []

    i = 0
    while i < radius + 1:
        if sig[i] == max(sig[:i + radius]):
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    while i < len(sig):
        if sig[i] == max(sig[i - radius:i + radius]):
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    while i < len(sig):
        if sig[i] == max(sig[i - radius:]):
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    return(np.array(peak_inds))




def cqrs_detect(sig, fs, conf=Conf()):

    cqrs = CQRS(sig, fs, conf)

    cqrs.detect()

    plt.plot(cqrs.sig_f, 'b')
    plt.plot(cqrs.sig_i, 'r')
    plt.show()

    #return cqrs.annotation
