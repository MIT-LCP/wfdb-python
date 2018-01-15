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
    """
    Challenges to overcome:
    - Must be able to return zero annotations when there are none
    - Must be able to detect few beats
    - Not classify large t-waves as qrs
    - Inverted qrs

    Record 117 is a good challenge. Big twave, weird qrs
    """


    def __init__(self, sig, fs, conf=Conf()):
        self.sig = sig
        self.fs = fs

        self.sig_len = len(sig)

        # These values are in samples
        self.rr = 60 * self.fs / conf.hr
        self.rr_max = 60 * self.fs / conf.hr_min
        self.rr_min = 60 * self.fs / conf.hr_max

        self.qrs_width = conf.qrs_width * fs




    def bandpass(self, fc_low=5, fc_high=20):
        """
        Apply a bandpass filter onto the signal, and save the filtered signal.
        """
        b, a = signal.butter(2, [fc_low * 2 / self.fs, fc_high * 2 / self.fs],
                             'pass')
        self.sig_f = signal.filtfilt(b, a, self.sig, axis=0)

        # Save the passband gain
        self.filter_gain = get_filter_gain(b, a, np.mean(fc_low, fc_high), fs)



    def mwi(self):
        """
        Apply moving wave integration with a ricker (Mexican hat) wavelet onto
        the filtered signal, and save the square of the integrated signal.
        
        The width of the hat is equal to the qrs width
        """
        b = signal.ricker(self.qrs_width, 4)
        self.sig_i = signal.filtfilt(b, [1], self.sig_f, axis=0) ** 2

        # Save the mwi gain
        self.mwi_gain = get_filter_gain(b, [1], np.mean(fc_low, fc_high), fs)

    def learn_params(self, n_calib_beats=8):
        """
        Find a number of beats using cross correlation to determine qrs
        detection thresholds.

        Parameters
        ----------
        n_calib_beats : int, optional
            Number of calibration beats to detect for learning

        Learn the following:
        - qrs threshold

        """

        # Find the dominant peaks of the signal. Store indices.
        self.dominant_peak_inds = find_dominant_peaks(self.sig_i,
                                                  int(self.qrs_width / 2))
        #dominant_peaks = self.sig_i[self.dominant_peak_inds]

        # Find n qrs peaks and store their amplitudes
        qrs_peak_amps = []

        peak_num = 0
        while len(qrs_peak_amps) < n_calib_beats
            i = dominant_peak_inds[peak_num]
            peak_num += 1



        self.qrs_thresh = np.mean(qrs_peak_amps)



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


def get_filter_gain(b, a, f_gain, fs):
    """
    Given filter coefficients, return the gain at a particular frequency
    """
    # Save the passband gain
    w, h = signal.freqz(b, a)
    w_gain = f_gain * 2 * np.pi / fs
    ind = np.where(h >= w_band)[0][0]
    gain = abs(h[ind])
    return gain






def cqrs_detect(sig, fs, conf=Conf()):

    cqrs = CQRS(sig, fs, conf)

    cqrs.detect()



    return cqrs.qrs_inds
