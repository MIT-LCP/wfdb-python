import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt
import pdb


class Conf(object):
    """
    Initial signal configuration object
    """
    def __init__(self, hr_init=75, hr_max=200, hr_min=25, qrs_width=0.1,
                 qrs_thr_init=0.13, qrs_thr_min=0):
        """
        Parameters
        ----------
        hr : int or float, optional
            Heart rate in beats per minute. Wait... what is this for?
            Setting initial hr and rr values.
        hr_max : int or float, optional
            Hard maximum heart rate between two beats, in beats per minute
        hr_min : int or float, optional
            Hard minimum heart rate between two beats, in beats per minute
        qrs_width : int or float, optional
            Expected qrs width in seconds
        qrs_thr_init : int or float, optional
            Initial qrs detection threshold. If learning=True and beats are
            detected, this will be overwritten.
        qrs_thr_min : int or float or string, optional
            Hard minimum detection threshold of qrs wave. Leave as 0 for no
            minimum.
        
        Should RT values be specified?
        """
        if hr_min < 0:
            raise ValueError("'hr_min' must be <= 0")

        if not hr_min < hr_init < hr_max:
            raise ValueError("'hr_min' < 'hr_init' < 'hr_max' must be True")

        if qrs_thr_init < qrs_thr_min:
            raise ValueError("qrs_thr_min must be <= qrs_thr_init")

        self.hr_init = hr_init
        self.hr_max = hr_max
        self.hr_min = hr_min
        self.qrs_width = qrs_width
        self.qrs_thr_init = qrs_thr_init
        self.qrs_thr_min = qrs_thr_min


class CQRS(object):
    """
    Challenges to overcome:
    - Must be able to return zero annotations when there are none
    - Must be able to detect few beats
    - Not classify large t-waves as qrs
    - Inverted qrs

    Record 117 is a good challenge. Big twave, weird qrs
    

    Process:
    - Bandpass
    - MWI
    - Learning:
      - Find prominent peaks (using qrs radius)
      - Find N beats. Use ricker wavelet convoluted with filtered signal.
      - Use beats to initialize qrs threshold. Other peak threshold is
        set to half of that. Set min peak and qrs threshold? There is a problem
        with setting this hard limit. gqrs has a hard
        limit, but pantompkins doesn't. Solution: Give options
          1. Specify limit. Can set to 0 if don't want.
          2. Set limit from peaks detected while learning. Else 0.
        During learning beat detection, confirm the set is roughly the same size.
        Reject 'beats' detected that are too far off in amplitude.
    """


    def __init__(self, sig, fs, conf=Conf()):
        self.sig = sig
        self.fs = fs
        self.sig_len = len(sig)



    def set_conf(self):
        """
        Set configuration parameters from the Conf object into the CQRS detector
        object.

        Time values are in samples, amplitude values are in mV.
        """
        self.rr = 60 * self.fs / self.conf.hr
        self.rr_max = 60 * self.fs / self.conf.hr_min
        self.rr_min = 60 * self.fs / self.conf.hr_max
        self.qrs_width = self.conf.qrs_width * self.fs

        self.qrs_thr_init = self.conf.qrs_thr_init
        self.qrs_thr_min = self.conf.qrs_thr_min

        


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


    def learn_init_params(self, n_calib_beats=8):
        """
        Find a number of consecutive beats using cross correlation to determine
        qrs detection thresholds.

        If the system fails to find enough beats, the default parameters will be
        used instead.

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

        # Go through the dominant peaks and find the qrs peaks
        qrs_peak_amps = []

        for peak_num in range(len(self.dominant_peak_inds)):
            i = self.dominant_peak_inds[peak_num]


            if len(qrs_peak_amps) == n_calib_beats:
                break

        
        if len(qrs_peak_amps) == n_calib_beats:
            self.qrs_thr = max(np.mean(qrs_peak_amps) / 2, self.qrs_thr_min)

        # Failed to find enough calibration beats. Use default values.
        else:
            self.set_init_params


        



        return

    def set_init_params(self):
        self.qrs_thr = self.qrs_thr_init
        self.recent_qrs_inds = 

        self.recent_rr = 
        
        

    def detect(self, sampfrom=0, sampto='end', learn=True):
        """
        Detect qrs locations between two samples
        """
        if sampfrom < 0:
            raise ValueError("'sampfrom' cannot be negative")
        if sampto == 'end':
            sampto = self.sig_len
        elif sampto > self.sig_len:
            raise ValueError("'sampto' cannot exceed the signal length")

        # Detected qrs indices
        self.qrs_inds = []
        # qrs indices found via backsearch
        self.backsearch_qrs_inds = []

        # Get signal configuration fields from Conf object
        self.set_conf()
        # Bandpass filter the signal
        self.bandpass()
        # Compute moving wave integration of filtered signal
        self.mwi()

        # Learn parameters
        if learn:
            self.learn_initial_params()
        else:
            self.set_initial_params()






        return

        for i in range(self.sig_len):


            if self.is_peak(i):
                pass



        self.qrs_inds = np.array(self.qrs_inds)




def find_dominant_peaks(sig, radius):
    """
    Find all dominant peaks in a signal. A sample is a dominant peak if it is
    the largest value within the <radius> samples on its left and right.

    In cases where it shares the max value with nearby samples, the earliest
    sample is classified as the dominant peak.

    TODO: Fix flat mountain scenarios.
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






def cqrs_detect(sig, fs, conf=Conf(), learn=True):

    cqrs = CQRS(sig, fs, conf)

    cqrs.detect()



    return cqrs.qrs_inds
