import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt
import pdb


class Conf(object):
    """
    Initial configuration object
    """
    def __init__(self, hr=75, hr_max=200, hr_min=25, qrs_width=0.1):
        """
        Parameters
        ----------
        hr : int or float, optional
            Heart rate in beats per minute
        qrs_width : int or float, optional
            Expected qrs width in seconds
        hr_max : int or float, optional
            Hard maximum heart rate between two beats, in beats per minute
        hr_min : int or float, optional
            Hard minimum heart rate between two beats, in beats per minute
        """

        self.hr = hr
        self.hr_max = hr_max
        self.hr_min = hr_min
        self.qrs_width = qrs_width


class CQRS(object):


    def __init__(self, sig, fs, conf=Conf()):
        self.sig = sig
        self.fs = fs

        self.sig_len = len(sig)

        # These values are in samples
        self.rr = 60 * self.fs / conf.hr
        self.rr_max = 60 * self.fs / conf.hr_min
        self.rr_min = 60 * self.fs / conf.hr_max

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
        - qrs threshold

        """
        # Find the dominant peaks of the signal. Store indices.
        self.dominant_peak_inds = find_dominant_peaks(self.sig_i,
                                                  int(self.qrs_width / 2))
        dominant_peaks = self.sig_i[self.dominant_peak_inds]

        # Cluster the peaks. The peaks in the largest amplitude cluster are
        # treated as qrs peaks. Use these to determine qrs detection thresholds
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dominant_peaks)
        top_cluster_ind = np.argmax(kmeans.cluster_centers_)
        
        self.qrs_peak_inds = self.dominant_peak_inds[np.where(kmeans.labels_ == top_cluster_ind)]
        qrs_peaks = self.sig_i[self.qrs_peak_inds]

        self.qrs_thresh = np.mean(qrs_peaks) / 3
        self.qrs_thresh_min = self.qrs_thresh / 3

        return
        
        

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



    return cqrs.qrs_inds
