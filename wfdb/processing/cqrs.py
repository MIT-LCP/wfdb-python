import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import pdb

from .peaks import find_local_peaks


class Conf(object):
    """
    Initial signal configuration object
    """
    def __init__(self, hr_init=75, hr_max=200, hr_min=25, qrs_width=0.1,
                 qrs_thr_init=0.13, qrs_thr_min=0, ref_period=0.2):
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
            Expected qrs width in seconds. Also acts as a refractory period.
        qrs_thr_init : int or float, optional
            Initial qrs detection threshold. Use when learning is False, or
            learning fails.
        qrs_thr_min : int or float or string, optional
            Hard minimum detection threshold of qrs wave. Leave as 0 for no
            minimum.
        ref_period : int or float, optional
            The qrs refractory period. The minimum . Why not just use hr_min?


        
        Should RT values be specified?

        rr = 60 / hr

        We have a refractory period (200ms in PT)
        and a t-wave curiosity period (360ms in PT)

        

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
        self.qrs_radius = int(self.qrs_width / 2)
        self.qrs_thr_init = qrs_thr_init
        self.qrs_thr_min = qrs_thr_min


class CQRS(object):
    """
    Challenges to overcome:
    - Must be able to return zero annotations when there are none
    - Must be able to detect few beats
    - Not classify large t-waves as qrs
    - Inverted qrs
    - rr history must be cleansed when there is a large gap? or keep rr history
      but discard recent qrs history.
    - Want to find peaks near beginning of record

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


    - Detection
        - found qrs. 
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

        # Should these all be integers?
        self.rr_init = 60 * self.fs / self.conf.hr_init
        self.rr_max = 60 * self.fs / self.conf.hr_min
        self.rr_min = 60 * self.fs / self.conf.hr_max
        
        self.qrs_width = int(self.conf.qrs_width * self.fs)
        self.qrs_radius = int(self.conf.qrs_radius * self.fs)

        self.qrs_thr_init = self.conf.qrs_thr_init
        self.qrs_thr_min = self.conf.qrs_thr_min




    def bandpass(self, fc_low=5, fc_high=20):
        """
        Apply a bandpass filter onto the signal, and save the filtered signal.
        """
        b, a = signal.butter(2, [fc_low * 2 / self.fs, fc_high * 2 / self.fs],
                             'pass')
        self.sig_f = signal.filtfilt(b, a, self.sig[sampfrom:sampto], axis=0)

        # Save the passband gain (x2 due to double filtering)
        self.filter_gain = (get_filter_gain(b, a, np.mean(fc_low, fc_high), fs)
                            * 2)

    def mwi(self):
        """
        Apply moving wave integration with a ricker (Mexican hat) wavelet onto
        the filtered signal, and save the square of the integrated signal.
        
        The width of the hat is equal to the qrs width

        Also find all local peaks in the mwi signal.
        """
        b = signal.ricker(self.qrs_width, 4)
        self.sig_i = signal.filtfilt(b, [1], self.sig_f, axis=0) ** 2

        # Save the mwi gain  (x2 due to double filtering) and the total gain
        # from raw to mwi
        self.mwi_gain = (get_filter_gain(b, [1], np.mean(fc_low, fc_high), fs)
                         * 2)
        self.transform_gain = self.filter_gain * self.mwi_gain

        self.peak_inds_i = find_local_peaks(self.sig_i)


    def learn_init_params(self, n_calib_beats=8):
        """
        Find a number of consecutive beats using cross correlation to determine
        qrs detection thresholds.

        If the system fails to find enough beats, the default parameters will be
        used instead.

        Beats are classified as 


        Parameters
        ----------
        n_calib_beats : int, optional
            Number of calibration beats to detect for learning

        Learn the following:
        - qrs threshold

        """
        if self.verbose:
            print('Learning initial signal parameters...')

        # Find the local peaks of the signal.
        peak_inds_f = find_local_peaks(self.sig_f,
                                             int(self.qrs_width / 2))

        last_qrs_ind = -self.rr_max
        qrs_inds = []
        qrs_amps = []
        noise_amps = []

        qrs_radius = int(self.qrs_width / 2)
        ricker_wavelet = signal.ricker(qrs_width, 4).reshape(-1,1)

        # Go through the peaks and find qrs peaks and noise peaks.
        for peak_num in range(
                np.where(peak_inds_f > self.qrs_width)[0][0],
                len(peak_inds_f)):

            i = peak_inds_f[peak_num]

            # Calculate cross-correlation between the filtered signal segment
            # and a ricker wavelet
            sig_segment = normalize((self.sig_f[i - qrs_radius:i + qrs_radius,
                                     0]**2).reshape(-1, 1), axis=0)

            xcorr = np.correlate(sig_segment[:, 0], b[:,0])

            # Classify as qrs if xcorr is large enough
            if xcorr > 0.6 and i-last_qrs_ind > rr_min:
                last_qrs_ind = i
                qrs_inds.append(i)
                qrs_amps.append(sig_i[i])
            else:
                noise_amps.append(sig_i[i])

            if len(qrs_inds) == n_calib_beats:
                break

        # Found enough calibration beats to initialize parameters
        if len(qrs_inds) == n_calib_beats:

            if self.verbose:
                print('Found %d beats during learning.' % n_calib_beats
                      + ' Initializing using learned parameters')

            # QRS amplitude is most important.
            qrs_amp = np.mean(qrs_amps)

            # Set noise amplitude if found
            if noise_amps:
                noise_amp = np.mean(noise_amps)
            else: 
                # Set default of 1/10 of qrs amplitude
                noise_amp = qrs_amp / 10

            # Get rr intervals of consecutive beats, if any.
            rr_intervals = np.diff(qrs_inds)
            rr_intervals = rr_intervals[rr_intervals < self.rr_max]
            if rr_intervals:
                rr_recent = np.mean(rr_intervals)
            else:
                rr_recent = self.rr_init

            # If an early qrs was detected, set last_qrs_ind so that it can be
            # picked up.
            last_qrs_ind = min(0, rr_recent)

            self.set_init_params(qrs_amp_recent=qrs_amp,
                                 noise_amp_recent=noise_amp,
                                 rr_recent=rr_recent,
                                 last_qrs_ind=last_qrs_ind)

        # Failed to find enough calibration beats. Use default values.
        else:
            if self.verbose:
                print('Failed to find %d beats during learning.'
                      % n_calib_beats)
            
            self.set_init_params()

        return

    def set_init_params(self, qrs_amp_recent, noise_amp_recent, rr_recent,
                        last_qrs_ind):
        """
        Set initial online parameters
        """
        self.qrs_amp_recent = qrs_amp_recent
        self.noise_amp_recent = noise_amp_recent
        self.qrs_thr = max(0.25*self.qrs_amp_recent
                           + 0.75*self.noise_amp_recent,
                           self.qrs_thr_min * self.transform_gain)
        self.rr_recent = self.rr_recent
        self.last_qrs_ind = last_qrs_ind


    def set_default_init_params(self):
        """
        Set initial online parameters using default values

        Steady state equation is qrs_thr = 0.25*qrs_amp + 0.75*noise_amp
        
        Estimate that qrs amp is 10x noise amp, which is equivalent to 
        qrs_thr = 0.325*qrs_amp which seems reasonable.
        """
        # self.set_init_params(qrs_amp_recent=self.qrs_thr_init / 0.325,
        #                      noise_amp_recent=self.qrs_thr_init / 3.25,
        #                      rr_recent=self.rr_init)

        if self.verbose:
            print('Initializing using default parameters')
        # Multiply the specified ecg thresholds by the filter and mwi gain
        # factors
        qrs_thr_init = self.conf.qrs_thr_init
        qrs_thr_min = self.conf.qrs_thr_min


        qrs_amp_recent = MATH
        noise_amp = MATH
        rr_recent = STUFF
        last_qrs_ind = 0

        self.set_init_params(qrs_amp_recent=qrs_amp,
                             noise_amp_recent=noise_amp,
                             rr_recent=rr_recent,
                             last_qrs_ind=last_qrs_ind)

    def is_qrs(self, peak_num, backsearch=False):
        """
        Check whether a peak is a qrs complex.

        - Must come after refractory period
        - Must pass qrs threshold
        - Must not be a t-wave. Check if close to previous qrs.
        """
        i = self.peak_inds_i[peak_num]
        if backsearch:
            qrs_thr = self.qrs_thr / 2
        else:
            qrs_thr = self.qrs_thr
        
        if (i-self.last_qrs_ind > self.ref_period
           and self.sig_i[i] > qrs_thr):
            if i-self.last_qrs_ind < self.t_check_interval:
                if self.is_twave(peak_num):
                    return False
            return True
        
        return False


    def update_qrs(self, peak_num, backsearch=False):
        """
        Update live qrs parameters

        Parameters
        ----------
        peak_num : int
            
        """

        i = self.peak_inds_i[peak_num]

        self.qrs_inds.append(i)
        self.last_qrs_ind = i
        self.last_qrs_peak_num = self.peak_num

        self.qrs_amp_recent = (0.875*self.qrs_amp_recent
                               + 0.125*self.sig_i[i])
        self.qrs_thr = max((0.25*self.qrs_amp_recent
                            + 0.75*self.noise_amp_recent), self.qrs_thr_min)

        rr_new = self.i - self.last_qrs_ind
        # Update recent rr if the beat is consecutive
        if rr_new < rr_max:
            self.rr_recent = 0.875*self.rr_recent + 0.125*rr_new

        if backsearch:
            self.backsearch_qrs_inds.append(i)

    def is_twave(self, peak_num):
        """
        Check whether a segment is a t-wave

        Compare the maximum gradient of the filtered signal segment with that
        of the previous qrs segment
        """
        i = self.peak_inds_i[peak_num]

        # Get half the qrs width of the signal to the left
        sig_segment = normalize((self.sig_f[i - qrs_radius:
                                            i, 0]**2).reshape(-1, 1),
                                axis=0)

        last_qrs_segment = self.sig_f[self.last_qrs_ind - self.qrs_radius:
                                      self.last_qrs_ind]

        segment_slope = np.diff(sig_segment)
        last_qrs_slope = np.diff(last_qrs_segment)

        # Should we be using absolute values?
        if max(segment_slope) < 0.5*max(abs(last_qrs_slope)):
            return True
        else:
            return False

    def update_noise(self, peak_num):
        """
        Update live noise parameters
        """
        i = self.peak_inds_i[peak_num]
        self.noise_amp_recent = 0.875*self.noise_amp_recent
                                + 0.125*self.sig_i[i]


    def require_backsearch(self):
        """
        Determine whether a backsearch should be performed on prior peaks
        """
        next_peak_ind = self.peak_inds_i[self.peak_num + 1]
        
        if next_peak_ind-self.last_qrs_ind > self.rr_recent*1.66:
            return True
        else:
            return False

    def backsearch(self):
        """
        Inspect previous peaks for qrs using a lower threshold
        """
        for peak_num in range(self.last_qrs_peak_num + 1, self.peak_num + 1):
            if self.is_qrs(peak_num=peak_num, backsearch=True):
                self.update_qrs_params(peak_num=peak_num, backsearch=True)

            # No need to update noise parameters if it was classified as noise.
            # It would have already been updated.


    def detect(self, sampfrom=0, sampto='end', learn=True, verbose=True):
        """
        Detect qrs locations between two samples
        """
        if sampfrom < 0:
            raise ValueError("'sampfrom' cannot be negative")
        self.sampfrom = sampfrom
        if sampto == 'end':
            self.sampto = self.sig_len
        elif sampto > self.sig_len:
            raise ValueError("'sampto' cannot exceed the signal length")

        # Length of the signal to perform detection on
        detect_len = self.sampto - self.sampfrom

        self.verbose = verbose

        # Get/set signal configuration fields from Conf object
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

        if self.verbose:
            print('Running QRS detection...')

        # Detected qrs indices
        self.qrs_inds = []
        # qrs indices found via backsearch
        self.backsearch_qrs_inds = []
        
        # Iterate through mwi signal peak indices
        for self.peak_num in range(len(self.peak_inds_i)):

            if self.is_qrs(self.peak_num):
                self.update_qrs(self.peak_num)
            else:
                self.update_noise(self.peak_num)

            # Before continuing to the next peak, do backsearch if necessary
            while self.require_backsearch():
                self.backsearch()

        # Detected indices are relative to starting sample
        if self.qrs_inds:
            self.qrs_inds = np.array(self.qrs_inds) + sampfrom

        if self.verbose:
            print('QRS detection complete.')

        return self.qrs_inds



def update_buffer(buffer, value):
    """
    Append a value to the end of a buffer, and remove its first element.
    """
    buffer.append(value)
    del(buffer[0])


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



def cqrs_detect(sig, fs, conf=Conf(), learn=True, verbose=True):

    cqrs = CQRS(sig, fs, conf)

    cqrs.detect()


    return cqrs.qrs_inds
