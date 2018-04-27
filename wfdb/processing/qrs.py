import copy
import numpy as np
from scipy import signal
from sklearn.preprocessing import normalize

from .basic import get_filter_gain
from .peaks import find_local_peaks
from ..io.record import Record


class XQRS(object):
    """
    The qrs detector class for the xqrs algorithm.

    The `XQRS.Conf` class is the configuration class that stores initial
    parameters for the detection.

    The `XQRS.detect` method runs the detection algorithm.

    The process works as follows:

    - Load the signal and configuration parameters.
    - Bandpass filter the signal between 5 and 20 Hz, to get the
      filtered signal.
    - Apply moving wave integration (mwi) with a ricker
      (Mexican hat) wavelet onto the filtered signal, and save the
      square of the integrated signal.
    - Conduct learning if specified, to initialize running
      parameters of noise and qrs amplitudes, the qrs detection
      threshold, and recent rr intervals. If learning is unspecified
      or fails, use default parameters.
    - Run the main detection. Iterate through the local maxima of
      the mwi signal. For each local maxima:

      - Check if it is a qrs complex. To be classified as a qrs,
        it must come after the refractory period, cross the qrs
        detection threshold, and not be classified as a t-wave
        if it comes close enough to the previous qrs. If
        successfully classified, update running detection
        threshold and heart rate parameters.
      - If not a qrs, classify it as a noise peak and update
        running parameters.
      - Before continuing to the next local maxima, if no qrs
        was detected within 1.66 times the recent rr interval,
        perform backsearch qrs detection. This checks previous
        peaks using a lower qrs detection threshold.

    Examples
    --------
    >>> import wfdb
    >>> from wfdb import processing

    >>> sig, fields = wfdb.rdsamp('sample-data/100', channels=[0])
    >>> xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
    >>> xqrs.detect()

    >>> wfdb.plot_items(signal=sig, ann_samp=[xqrs.qrs_inds])

    """

    def __init__(self, sig, fs, conf=None):
        if sig.ndim != 1:
            raise ValueError('sig must be a 1d numpy array')
        self.sig = sig
        self.fs = fs
        self.sig_len = len(sig)
        self.conf = conf or XQRS.Conf()
        self._set_conf()

    class Conf(object):
        """
        Initial signal configuration object for this qrs detector
        """
        def __init__(self, hr_init=75, hr_max=200, hr_min=25, qrs_width=0.1,
                     qrs_thr_init=0.13, qrs_thr_min=0, ref_period=0.2,
                     t_inspect_period=0.36):
            """
            Parameters
            ----------
            hr_init : int or float, optional
                Initial heart rate in beats per minute. Used for
            hr_max : int or float, optional
                Hard maximum heart rate between two beats, in beats per
                minute. Used for refractory period.
            hr_min : int or float, optional
                Hard minimum heart rate between two beats, in beats per
                minute. Used for calculating recent rr intervals.
            qrs_width : int or float, optional
                Expected qrs width in seconds. Used for filter widths
                indirect refractory period.
            qrs_thr_init : int or float, optional
                Initial qrs detection threshold in mV. Use when learning
                is False, or learning fails.
            qrs_thr_min : int or float or string, optional
                Hard minimum detection threshold of qrs wave. Leave as 0
                for no minimum.
            ref_period : int or float, optional
                The qrs refractory period.
            t_inspect_period : int or float, optional
                The period below which a potential qrs complex is
                inspected to see if it is a t wave.

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
            self.qrs_radius = self.qrs_width / 2
            self.qrs_thr_init = qrs_thr_init
            self.qrs_thr_min = qrs_thr_min
            self.ref_period = ref_period
            self.t_inspect_period = t_inspect_period

    def _set_conf(self):
        """
        Set configuration parameters from the Conf object into the detector
        object.

        Time values are converted to samples, and amplitude values are in mV.
        """
        self.rr_init = 60 * self.fs / self.conf.hr_init
        self.rr_max = 60 * self.fs / self.conf.hr_min
        self.rr_min = 60 * self.fs / self.conf.hr_max

        # Note: if qrs_width is odd, qrs_width == qrs_radius*2 + 1
        self.qrs_width = int(self.conf.qrs_width * self.fs)
        self.qrs_radius = int(self.conf.qrs_radius * self.fs)

        self.qrs_thr_init = self.conf.qrs_thr_init
        self.qrs_thr_min = self.conf.qrs_thr_min

        self.ref_period = int(self.conf.ref_period * self.fs)
        self.t_inspect_period = int(self.conf.t_inspect_period * self.fs)


    def _bandpass(self, fc_low=5, fc_high=20):
        """
        Apply a bandpass filter onto the signal, and save the filtered signal.
        """
        self.fc_low = fc_low
        self.fc_high = fc_high

        b, a = signal.butter(2, [float(fc_low) * 2 / self.fs,
                                 float(fc_high) * 2 / self.fs], 'pass')
        self.sig_f = signal.filtfilt(b, a, self.sig[self.sampfrom:self.sampto],
                                     axis=0)
        # Save the passband gain (x2 due to double filtering)
        self.filter_gain = get_filter_gain(b, a, np.mean([fc_low, fc_high]),
                                           self.fs) * 2


    def _mwi(self):
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
        self.mwi_gain = get_filter_gain(b, [1],
                         np.mean([self.fc_low, self.fc_high]), self.fs) * 2
        self.transform_gain = self.filter_gain * self.mwi_gain
        self.peak_inds_i = find_local_peaks(self.sig_i, radius=self.qrs_radius)
        self.n_peaks_i = len(self.peak_inds_i)

    def _learn_init_params(self, n_calib_beats=8):
        """
        Find a number of consecutive beats and use them to initialize:
        - recent qrs amplitude
        - recent noise amplitude
        - recent rr interval
        - qrs detection threshold

        The learning works as follows:
        - Find all local maxima (largest sample within `qrs_radius`
          samples) of the filtered signal.
        - Inspect the local maxima until `n_calib_beats` beats are
          found:
          - Calculate the cross-correlation between a ricker wavelet of
            length `qrs_width`, and the filtered signal segment centered
            around the local maximum.
          - If the cross-correlation exceeds 0.6, classify it as a beat.
        - Use the beats to initialize the previously described
          parameters.
        - If the system fails to find enough beats, the default
          parameters will be used instead. See the docstring of
          `XQRS._set_default_init_params` for detauls.

        Parameters
        ----------
        n_calib_beats : int, optional
            Number of calibration beats to detect for learning


        """
        if self.verbose:
            print('Learning initial signal parameters...')

        last_qrs_ind = -self.rr_max
        qrs_inds = []
        qrs_amps = []
        noise_amps = []

        ricker_wavelet = signal.ricker(self.qrs_radius * 2, 4).reshape(-1,1)

        # Find the local peaks of the signal.
        peak_inds_f = find_local_peaks(self.sig_f, self.qrs_radius)

        peak_inds_f_r = np.where(peak_inds_f > self.qrs_width)[0]
        peak_inds_f_l = np.where(peak_inds_f <= self.sig_len - self.qrs_width)[0]

        # Skip if no peaks in range
        if (not peak_inds_f.size or not peak_inds_f_r.size
                                 or not peak_inds_f_l.size):
            if self.verbose:
                print('Failed to find %d beats during learning.'
                      % n_calib_beats)
            self._set_default_init_params()
            return

        # Go through the peaks and find qrs peaks and noise peaks.
        # only inspect peaks with at least qrs_radius around either side
        for peak_num in range(peak_inds_f_r[0], peak_inds_f_l[0]):
            i = peak_inds_f[peak_num]
            # Calculate cross-correlation between the filtered signal
            # segment and a ricker wavelet

            # Question: should the signal be squared? Case for inverse qrs
            # complexes
            sig_segment = normalize((self.sig_f[i - self.qrs_radius:
                                                i + self.qrs_radius]).reshape(-1, 1), axis=0)

            xcorr = np.correlate(sig_segment[:, 0], ricker_wavelet[:,0])

            # Classify as qrs if xcorr is large enough
            if xcorr > 0.6 and i-last_qrs_ind > self.rr_min:
                last_qrs_ind = i
                qrs_inds.append(i)
                qrs_amps.append(self.sig_i[i])
            else:
                noise_amps.append(self.sig_i[i])

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
            if rr_intervals.any():
                rr_recent = np.mean(rr_intervals)
            else:
                rr_recent = self.rr_init

            # If an early qrs was detected, set last_qrs_ind so that it can be
            # picked up.
            last_qrs_ind = min(0, qrs_inds[0] - self.rr_min - 1)

            self._set_init_params(qrs_amp_recent=qrs_amp,
                                  noise_amp_recent=noise_amp,
                                  rr_recent=rr_recent,
                                  last_qrs_ind=last_qrs_ind)

        # Failed to find enough calibration beats. Use default values.
        else:
            if self.verbose:
                print('Failed to find %d beats during learning.'
                      % n_calib_beats)

            self._set_default_init_params()


    def _set_init_params(self, qrs_amp_recent, noise_amp_recent, rr_recent,
                         last_qrs_ind):
        """
        Set initial online parameters
        """
        self.qrs_amp_recent = qrs_amp_recent
        self.noise_amp_recent = noise_amp_recent
        # What happens if qrs_thr is calculated to be less than the explicit
        # min threshold? Should print warning?
        self.qrs_thr = max(0.25*self.qrs_amp_recent
                           + 0.75*self.noise_amp_recent,
                           self.qrs_thr_min * self.transform_gain)
        self.rr_recent = rr_recent
        self.last_qrs_ind = last_qrs_ind

        # No qrs detected initially
        self.last_qrs_peak_num = None


    def _set_default_init_params(self):
        """
        Set initial running parameters using default values.

        The steady state equation is:
          `qrs_thr = 0.25*qrs_amp + 0.75*noise_amp`

        Estimate that qrs amp is 10x noise amp, giving:
          `qrs_thr = 0.325 * qrs_amp or 13/40 * qrs_amp`

        """
        if self.verbose:
            print('Initializing using default parameters')
        # Multiply the specified ecg thresholds by the filter and mwi gain
        # factors
        qrs_thr_init = self.qrs_thr_init * self.transform_gain
        qrs_thr_min = self.qrs_thr_min * self.transform_gain

        qrs_amp = 27/40 * qrs_thr_init
        noise_amp = qrs_amp / 10
        rr_recent = self.rr_init
        last_qrs_ind = 0

        self._set_init_params(qrs_amp_recent=qrs_amp,
                              noise_amp_recent=noise_amp,
                              rr_recent=rr_recent,
                              last_qrs_ind=last_qrs_ind)

    def _is_qrs(self, peak_num, backsearch=False):
        """
        Check whether a peak is a qrs complex. It is classified as qrs
        if it:
        - Comes after the refractory period
        - Passes qrs threshold
        - Is not a t-wave (check it if the peak is close to the previous
          qrs).

        Parameters
        ----------
        peak_num : int
            The peak number of the mwi signal to be inspected
        backsearch: bool, optional
            Whether the peak is being inspected during backsearch

        """
        i = self.peak_inds_i[peak_num]
        if backsearch:
            qrs_thr = self.qrs_thr / 2
        else:
            qrs_thr = self.qrs_thr

        if (i-self.last_qrs_ind > self.ref_period
           and self.sig_i[i] > qrs_thr):
            if i-self.last_qrs_ind < self.t_inspect_period:
                if self._is_twave(peak_num):
                    return False
            return True

        return False


    def _update_qrs(self, peak_num, backsearch=False):
        """
        Update live qrs parameters. Adjust the recent rr-intervals and
        qrs amplitudes, and the qrs threshold.

        Parameters
        ----------
        peak_num : int
            The peak number of the mwi signal where the qrs is detected
        backsearch: bool, optional
            Whether the qrs was found via backsearch
        """

        i = self.peak_inds_i[peak_num]

        # Update recent rr if the beat is consecutive (do this before
        # updating self.last_qrs_ind)
        rr_new = i - self.last_qrs_ind
        if rr_new < self.rr_max:
            self.rr_recent = 0.875*self.rr_recent + 0.125*rr_new

        self.qrs_inds.append(i)
        self.last_qrs_ind = i
        # Peak number corresponding to last qrs
        self.last_qrs_peak_num = self.peak_num

        # qrs recent amplitude is adjusted twice as quickly if the peak
        # was found via backsearch
        if backsearch:
            self.backsearch_qrs_inds.append(i)
            self.qrs_amp_recent = (0.75*self.qrs_amp_recent
                                   + 0.25*self.sig_i[i])
        else:
            self.qrs_amp_recent = (0.875*self.qrs_amp_recent
                                   + 0.125*self.sig_i[i])

        self.qrs_thr = max((0.25*self.qrs_amp_recent
                            + 0.75*self.noise_amp_recent), self.qrs_thr_min)

        return


    def _is_twave(self, peak_num):
        """
        Check whether a segment is a t-wave. Compare the maximum gradient of
        the filtered signal segment with that of the previous qrs segment.

        Parameters
        ----------
        peak_num : int
            The peak number of the mwi signal where the qrs is detected

        """
        i = self.peak_inds_i[peak_num]

        # Due to initialization parameters, last_qrs_ind may be negative.
        # No way to check in this instance.
        if self.last_qrs_ind - self.qrs_radius < 0:
            return False

        # Get half the qrs width of the signal to the left.
        # Should this be squared?
        sig_segment = normalize((self.sig_f[i - self.qrs_radius:i]
                                 ).reshape(-1, 1), axis=0)
        last_qrs_segment = self.sig_f[self.last_qrs_ind - self.qrs_radius:
                                      self.last_qrs_ind]

        segment_slope = np.diff(sig_segment)
        last_qrs_slope = np.diff(last_qrs_segment)

        # Should we be using absolute values?
        if max(segment_slope) < 0.5*max(abs(last_qrs_slope)):
            return True
        else:
            return False

    def _update_noise(self, peak_num):
        """
        Update live noise parameters
        """
        i = self.peak_inds_i[peak_num]
        self.noise_amp_recent = (0.875*self.noise_amp_recent
                                 + 0.125*self.sig_i[i])
        return

    def _require_backsearch(self):
        """
        Determine whether a backsearch should be performed on prior peaks
        """
        if self.peak_num == self.n_peaks_i-1:
            # If we just return false, we may miss a chance to backsearch.
            # Update this?
            return False

        next_peak_ind = self.peak_inds_i[self.peak_num + 1]

        if next_peak_ind-self.last_qrs_ind > self.rr_recent*1.66:
            return True
        else:
            return False

    def _backsearch(self):
        """
        Inspect previous peaks from the last detected qrs peak (if any),
        using a lower threshold

        """
        if self.last_qrs_peak_num is not None:
            for peak_num in range(self.last_qrs_peak_num + 1, self.peak_num + 1):
                if self._is_qrs(peak_num=peak_num, backsearch=True):
                    self._update_qrs(peak_num=peak_num, backsearch=True)
                # No need to update noise parameters if it was classified as
                # noise. It would have already been updated.

    def _run_detection(self):
        """
        Run the qrs detection after all signals and parameters have been
        configured and set.

        """
        if self.verbose:
            print('Running QRS detection...')

        # Detected qrs indices
        self.qrs_inds = []
        # qrs indices found via backsearch
        self.backsearch_qrs_inds = []

        # Iterate through mwi signal peak indices
        for self.peak_num in range(self.n_peaks_i):
            if self._is_qrs(self.peak_num):
                self._update_qrs(self.peak_num)
            else:
                self._update_noise(self.peak_num)

            # Before continuing to the next peak, do backsearch if
            # necessary
            if self._require_backsearch():
                self._backsearch()

        # Detected indices are relative to starting sample
        if self.qrs_inds:
            self.qrs_inds = np.array(self.qrs_inds) + self.sampfrom
        else:
            self.qrs_inds = np.array(self.qrs_inds)

        if self.verbose:
            print('QRS detection complete.')


    def detect(self, sampfrom=0, sampto='end', learn=True, verbose=True):
        """
        Detect qrs locations between two samples.

        Parameters
        ----------
        sampfrom : int, optional
            The starting sample number to run the detection on.
        sampto : int, optional
            The final sample number to run the detection on. Set as
            'end' to run on the entire signal.
        learn : bool, optional
            Whether to apply learning on the signal before running the
            main detection. If learning fails or is not conducted, the
            default configuration parameters will be used to initialize
            these variables. See the `XQRS._learn_init_params` docstring
            for details.
        verbose : bool, optional
            Whether to display the stages and outcomes of the detection
            process.

        """
        if sampfrom < 0:
            raise ValueError("'sampfrom' cannot be negative")
        self.sampfrom = sampfrom

        if sampto == 'end':
            sampto = self.sig_len
        elif sampto > self.sig_len:
            raise ValueError("'sampto' cannot exceed the signal length")
        self.sampto = sampto
        self.verbose = verbose

        # Don't attempt to run on a flat signal
        if np.max(self.sig) == np.min(self.sig):
            self.qrs_inds = np.empty(0)
            if self.verbose:
                print('Flat signal. Detection skipped.')
            return

        # Get/set signal configuration fields from Conf object
        self._set_conf()
        # Bandpass filter the signal
        self._bandpass()
        # Compute moving wave integration of filtered signal
        self._mwi()

        # Initialize the running parameters
        if learn:
            self._learn_init_params()
        else:
            self._set_default_init_params()

        # Run the detection
        self._run_detection()


def xqrs_detect(sig, fs, sampfrom=0, sampto='end', conf=None,
                learn=True, verbose=True):
    """
    Run the 'xqrs' qrs detection algorithm on a signal. See the
    docstring of the XQRS class for algorithm details.

    Parameters
    ----------
    sig : numpy array
        The input ecg signal to apply the qrs detection on.
    fs : int or float
        The sampling frequency of the input signal.
    sampfrom : int, optional
        The starting sample number to run the detection on.
    sampto :
        The final sample number to run the detection on. Set as 'end' to
        run on the entire signal.
    conf : XQRS.Conf object, optional
        The configuration object specifying signal configuration
        parameters. See the docstring of the XQRS.Conf class.
    learn : bool, optional
        Whether to apply learning on the signal before running the main
        detection. If learning fails or is not conducted, the default
        configuration parameters will be used to initialize these
        variables.
    verbose : bool, optional
        Whether to display the stages and outcomes of the detection
        process.

    Returns
    -------
    qrs_inds : numpy array
        The indices of the detected qrs complexes

    Examples
    --------
    >>> import wfdb
    >>> from wfdb import processing

    >>> sig, fields = wfdb.rdsamp('sample-data/100', channels=[0])
    >>> qrs_inds = processing.xqrs_detect(sig=sig[:,0], fs=fields['fs'])

    """
    xqrs = XQRS(sig=sig, fs=fs, conf=conf)
    xqrs.detect(sampfrom=sampfrom, sampto=sampto, verbose=verbose)
    return xqrs.qrs_inds


def time_to_sample_number(seconds, frequency):
    return seconds * frequency + 0.5

class GQRS(object):
    """
    GQRS detection class
    """
    class Conf(object):
        """
        Initial signal configuration object for this qrs detector

        """
        def __init__(self, fs, adc_gain, hr=75,
                     RRdelta=0.2, RRmin=0.28, RRmax=2.4,
                     QS=0.07, QT=0.35,
                     RTmin=0.25, RTmax=0.33,
                     QRSa=750, QRSamin=130,
                     thresh=1.0):
            self.fs = fs

            self.sps = int(time_to_sample_number(1, fs))
            self.spm = int(time_to_sample_number(60, fs))

            self.hr = hr
            self.RR = 60.0 / self.hr
            self.RRdelta = RRdelta
            self.RRmin = RRmin
            self.RRmax = RRmax
            self.QS = QS
            self.QT = QT
            self.RTmin = RTmin
            self.RTmax = RTmax
            self.QRSa = QRSa
            self.QRSamin = QRSamin
            self.thresh = thresh

            self._NORMAL = 1  # normal beat
            self._ARFCT = 16  # isolated QRS-like artifact
            self._NOTE = 22  # comment annotation
            self._TWAVE = 27  # T-wave peak
            self._NPEAKS = 64  # number of peaks buffered (per signal)
            self._BUFLN = 32768  # must be a power of 2, see qf()

            self.rrmean = int(self.RR * self.sps)
            self.rrdev = int(self.RRdelta * self.sps)
            self.rrmin = int(self.RRmin * self.sps)
            self.rrmax = int(self.RRmax * self.sps)

            self.rrinc = int(self.rrmean / 40)
            if self.rrinc < 1:
                self.rrinc = 1

            self.dt = int(self.QS * self.sps / 4)
            if self.dt < 1:
                raise Exception('Sampling rate is too low. Unable to use signal.')

            self.rtmin = int(self.RTmin * self.sps)
            self.rtmean = int(0.75 * self.QT * self.sps)
            self.rtmax = int(self.RTmax * self.sps)

            dv = adc_gain * self.QRSamin * 0.001
            self.pthr = int((self.thresh * dv * dv) / 6)
            self.qthr = self.pthr << 1
            self.pthmin = self.pthr >> 2
            self.qthmin = int((self.pthmin << 2) / 3)
            self.tamean = self.qthr  # initial value for mean T-wave amplitude

            # Filter constants and thresholds.
            self.dt2 = 2 * self.dt
            self.dt3 = 3 * self.dt
            self.dt4 = 4 * self.dt

            self.smdt = self.dt
            self.v1norm = self.smdt * self.dt * 64

            self.smt = 0
            self.smt0 = 0 + self.smdt


    class Peak(object):
        def __init__(self, peak_time, peak_amp, peak_type):
            self.time = peak_time
            self.amp = peak_amp
            self.type = peak_type
            self.next_peak = None
            self.prev_peak = None


    class Annotation(object):
        def __init__(self, ann_time, ann_type, ann_subtype, ann_num):
            self.time = ann_time
            self.type = ann_type
            self.subtype = ann_subtype
            self.num = ann_num


    def putann(self, annotation):
        self.annotations.append(copy.deepcopy(annotation))

    def detect(self, x, conf, adc_zero):
        """
        Run detection. x is digital signal
        """
        self.c = conf
        self.annotations = []
        self.sample_valid = False

        if len(x) < 1:
            return []

        self.x = x
        self.adc_zero = adc_zero

        self.qfv = np.zeros((self.c._BUFLN), dtype="int64")
        self.smv = np.zeros((self.c._BUFLN), dtype="int64")
        self.v1 = 0

        t0 = 0
        self.tf = len(x) - 1
        self.t = 0 - self.c.dt4

        self.annot = GQRS.Annotation(0, "NOTE", 0, 0)

        # Cicular buffer of Peaks
        first_peak = GQRS.Peak(0, 0, 0)
        tmp = first_peak
        for _ in range(1, self.c._NPEAKS):
            tmp.next_peak = GQRS.Peak(0, 0, 0)
            tmp.next_peak.prev_peak = tmp
            tmp = tmp.next_peak
        tmp.next_peak = first_peak
        first_peak.prev_peak = tmp
        self.current_peak = first_peak

        if self.c.spm > self.c._BUFLN:
            if self.tf - t0 > self.c._BUFLN:
                tf_learn = t0 + self.c._BUFLN - self.c.dt4
            else:
                tf_learn = self.tf - self.c.dt4
        else:
            if self.tf - t0 > self.c.spm:
                tf_learn = t0 + self.c.spm - self.c.dt4
            else:
                tf_learn = self.tf - self.c.dt4

        self.countdown = -1
        self.state = "LEARNING"
        self.gqrs(t0, tf_learn)

        self.rewind_gqrs()

        self.state = "RUNNING"
        self.t = t0 - self.c.dt4
        self.gqrs(t0, self.tf)

        return self.annotations

    def rewind_gqrs(self):
        self.countdown = -1
        self.at(self.t)
        self.annot.time = 0
        self.annot.type = "NORMAL"
        self.annot.subtype = 0
        self.annot.num = 0
        p = self.current_peak
        for _ in range(self.c._NPEAKS):
            p.time = 0
            p.type = 0
            p.amp = 0
            p = p.next_peak

    def at(self, t):
        if t < 0:
            self.sample_valid = True
            return self.x[0]
        if t > len(self.x) - 1:
            self.sample_valid = False
            return self.x[-1]
        self.sample_valid = True
        return self.x[t]

    def smv_at(self, t):
        return self.smv[t & (self.c._BUFLN - 1)]

    def smv_put(self, t, v):
        self.smv[t & (self.c._BUFLN - 1)] = v

    def qfv_at(self, t):
        return self.qfv[t & (self.c._BUFLN - 1)]

    def qfv_put(self, t, v):
        self.qfv[t & (self.c._BUFLN - 1)] = v

    def sm(self, at_t):
        # implements a trapezoidal low pass (smoothing) filter
        # (with a gain of 4*smdt) applied to input signal sig
        # before the QRS matched filter qf().
        # Before attempting to 'rewind' by more than BUFLN-smdt
        # samples, reset smt and smt0.

        # Calculate samp values from self.smt to at_t.
        smt = self.c.smt
        smdt = int(self.c.smdt)

        v = 0
        while at_t > smt:
            smt += 1
            # from dt+1 onwards
            if smt > int(self.c.smt0):
                tmp = int(self.smv_at(smt - 1) + \
                             self.at(smt + smdt) + self.at(smt + smdt - 1) - \
                             self.at(smt - smdt) - self.at(smt - smdt - 1))
                self.smv_put(smt, tmp)
                self.SIG_SMOOTH.append(tmp)
            # from 1 to dt. 0 is never calculated.
            else:
                v = int(self.at(smt))
                for j in range(1, smdt):
                    smtpj = self.at(smt + j)
                    smtlj = self.at(smt - j)
                    v += int(smtpj + smtlj)
                self.smv_put(smt, (v << 1) + self.at(smt + j+1) + self.at(smt - j-1) - \
                             self.adc_zero * (smdt << 2))

                self.SIG_SMOOTH.append((v << 1) + self.at(smt + j+1) + self.at(smt - j-1) - \
                             self.adc_zero * (smdt << 2))
        self.c.smt = smt

        return self.smv_at(at_t)

    def qf(self):
        # evaluate the QRS detector filter for the next sample

        # do this first, to ensure that all of the other smoothed values needed below are in the buffer
        dv2 = self.sm(self.t + self.c.dt4)
        dv2 -= self.smv_at(self.t - self.c.dt4)
        dv1 = int(self.smv_at(self.t + self.c.dt) - self.smv_at(self.t - self.c.dt))
        dv = dv1 << 1
        dv -= int(self.smv_at(self.t + self.c.dt2) - self.smv_at(self.t - self.c.dt2))
        dv = dv << 1
        dv += dv1
        dv -= int(self.smv_at(self.t + self.c.dt3) - self.smv_at(self.t - self.c.dt3))
        dv = dv << 1
        dv += dv2
        self.v1 += dv
        v0 = int(self.v1 / self.c.v1norm)
        self.qfv_put(self.t, v0 * v0)
        self.SIG_QRS.append(v0 ** 2)

    def gqrs(self, from_sample, to_sample):
        q0 = None
        q1 = 0
        q2 = 0
        rr = None
        rrd = None
        rt = None
        rtd = None
        rtdmin = None

        p = None  # (Peak)
        q = None  # (Peak)
        r = None  # (Peak)
        tw = None  # (Peak)

        last_peak = from_sample
        last_qrs = from_sample

        self.SIG_SMOOTH = []
        self.SIG_QRS = []

        def add_peak(peak_time, peak_amp, type):
            p = self.current_peak.next_peak
            p.time = peak_time
            p.amp = peak_amp
            p.type = type
            self.current_peak = p
            p.next_peak.amp = 0

        def peaktype(p):
            # peaktype() returns 1 if p is the most prominent peak in its neighborhood, 2
            # otherwise.  The neighborhood consists of all other peaks within rrmin.
            # Normally, "most prominent" is equivalent to "largest in amplitude", but this
            # is not always true.  For example, consider three consecutive peaks a, b, c
            # such that a and b share a neighborhood, b and c share a neighborhood, but a
            # and c do not; and suppose that amp(a) > amp(b) > amp(c).  In this case, if
            # there are no other peaks, a is the most prominent peak in the (a, b)
            # neighborhood.  Since b is thus identified as a non-prominent peak, c becomes
            # the most prominent peak in the (b, c) neighborhood.  This is necessary to
            # permit detection of low-amplitude beats that closely precede or follow beats
            # with large secondary peaks (as, for example, in R-on-T PVCs).
            if p.type:
                return p.type
            else:
                a = p.amp
                t0 = p.time - self.c.rrmin
                t1 = p.time + self.c.rrmin

                if t0 < 0:
                    t0 = 0

                pp = p.prev_peak
                while t0 < pp.time and pp.time < pp.next_peak.time:
                    if pp.amp == 0:
                        break
                    if a < pp.amp and peaktype(pp) == 1:
                        p.type = 2
                        return p.type
                    # end:
                    pp = pp.prev_peak

                pp = p.next_peak
                while pp.time < t1 and pp.time > pp.prev_peak.time:
                    if pp.amp == 0:
                        break
                    if a < pp.amp and peaktype(pp) == 1:
                        p.type = 2
                        return p.type
                    # end:
                    pp = pp.next_peak

                p.type = 1
                return p.type

        def find_missing(r, p):
            if r is None or p is None:
                return None

            minrrerr = p.time - r.time

            s = None
            q = r.next_peak
            while q.time < p.time:
                if peaktype(q) == 1:
                    rrtmp = q.time - r.time
                    rrerr = rrtmp - self.c.rrmean
                    if rrerr < 0:
                        rrerr = -rrerr
                    if rrerr < minrrerr:
                        minrrerr = rrerr
                        s = q
                # end:
                q = q.next_peak

            return s

        r = None
        next_minute = 0
        minutes = 0
        while self.t <= to_sample + self.c.sps:
            if self.countdown < 0:
                if self.sample_valid:
                    self.qf()
                else:
                    self.countdown = int(time_to_sample_number(1, self.c.fs))
                    self.state = "CLEANUP"
            else:
                self.countdown -= 1
                if self.countdown < 0:
                    break

            q0 = self.qfv_at(self.t)
            q1 = self.qfv_at(self.t - 1)
            q2 = self.qfv_at(self.t - 2)
            # state == RUNNING only
            if q1 > self.c.pthr and q2 < q1 and q1 >= q0 and self.t > self.c.dt4:
                add_peak(self.t - 1, q1, 0)
                last_peak = self.t - 1
                p = self.current_peak.next_peak
                while p.time < self.t - self.c.rtmax:
                    if p.time >= self.annot.time + self.c.rrmin and peaktype(p) == 1:
                        if p.amp > self.c.qthr:
                            rr = p.time - self.annot.time
                            q = find_missing(r, p)
                            if rr > self.c.rrmean + 2 * self.c.rrdev and \
                               rr > 2 * (self.c.rrmean - self.c.rrdev) and \
                               q is not None:
                                p = q
                                rr = p.time - self.annot.time
                                self.annot.subtype = 1
                            rrd = rr - self.c.rrmean
                            if rrd < 0:
                                rrd = -rrd
                            self.c.rrdev += (rrd - self.c.rrdev) >> 3
                            if rrd > self.c.rrinc:
                                rrd = self.c.rrinc
                            if rr > self.c.rrmean:
                                self.c.rrmean += rrd
                            else:
                                self.c.rrmean -= rrd
                            if p.amp > self.c.qthr * 4:
                                self.c.qthr += 1
                            elif p.amp < self.c.qthr:
                                self.c.qthr -= 1
                            if self.c.qthr > self.c.pthr * 20:
                                self.c.qthr = self.c.pthr * 20
                            last_qrs = p.time

                            if self.state == "RUNNING":
                                self.annot.time = p.time - self.c.dt2
                                self.annot.type = "NORMAL"
                                qsize = int(p.amp * 10.0 / self.c.qthr)
                                if qsize > 127:
                                    qsize = 127
                                self.annot.num = qsize
                                self.putann(self.annot)
                                self.annot.time += self.c.dt2

                            # look for this beat's T-wave
                            tw = None
                            rtdmin = self.c.rtmean
                            q = p.next_peak
                            while q.time > self.annot.time:
                                rt = q.time - self.annot.time - self.c.dt2
                                if rt < self.c.rtmin:
                                    # end:
                                    q = q.next_peak
                                    continue
                                if rt > self.c.rtmax:
                                    break
                                rtd = rt - self.c.rtmean
                                if rtd < 0:
                                    rtd = -rtd
                                if rtd < rtdmin:
                                    rtdmin = rtd
                                    tw = q
                                # end:
                                q = q.next_peak
                            if tw is not None:
                                tmp_time = tw.time - self.c.dt2
                                tann = GQRS.Annotation(tmp_time, "TWAVE",
                                                  1 if tmp_time > self.annot.time + self.c.rtmean else 0,
                                                  rtdmin)
                                # if self.state == "RUNNING":
                                #     self.putann(tann)
                                rt = tann.time - self.annot.time
                                self.c.rtmean += (rt - self.c.rtmean) >> 4
                                if self.c.rtmean > self.c.rtmax:
                                    self.c.rtmean = self.c.rtmax
                                elif self.c.rtmean < self.c.rtmin:
                                    self.c.rtmean = self.c.rrmin
                                tw.type = 2  # mark T-wave as secondary
                            r = p
                            q = None
                            self.annot.subtype = 0
                        elif self.t - last_qrs > self.c.rrmax and self.c.qthr > self.c.qthmin:
                            self.c.qthr -= (self.c.qthr >> 4)
                    # end:
                    p = p.next_peak
            elif self.t - last_peak > self.c.rrmax and self.c.pthr > self.c.pthmin:
                self.c.pthr -= (self.c.pthr >> 4)

            self.t += 1
            if self.t >= next_minute:
                next_minute += self.c.spm
                minutes += 1
                if minutes >= 60:
                    minutes = 0

        if self.state == "LEARNING":
            return

        # Mark the last beat or two.
        p = self.current_peak.next_peak
        while p.time < p.next_peak.time:
            if p.time >= self.annot.time + self.c.rrmin and p.time < self.tf and peaktype(p) == 1:
                self.annot.type = "NORMAL"
                self.annot.time = p.time
                self.putann(self.annot)
            # end:
            p = p.next_peak


def gqrs_detect(sig=None, fs=None, d_sig=None, adc_gain=None, adc_zero=None,
                threshold=1.0, hr=75, RRdelta=0.2, RRmin=0.28, RRmax=2.4,
                QS=0.07, QT=0.35, RTmin=0.25, RTmax=0.33,
                QRSa=750, QRSamin=130):
    """
    Detect qrs locations in a single channel ecg. Functionally, a direct port
    of the gqrs algorithm from the original wfdb package. Accepts either a
    physical signal, or a digital signal with known adc_gain and adc_zero.

    See the notes below for a summary of the program. This algorithm is not
    being developed/supported.

    Parameters
    ----------
    sig : 1d numpy array, optional
        The input physical signal. The detection algorithm which replicates
        the original, works using digital samples, and this physical option is
        provided as a convenient interface. If this is the specified input
        signal, automatic adc is performed using 24 bit precision, to obtain
        the `d_sig`, `adc_gain`, and `adc_zero` parameters. There may be minor
        differences in detection results (ie. an occasional 1 sample
        difference) between using `sig` and `d_sig`. To replicate the exact
        output of the original gqrs algorithm, use the `d_sig` argument
        instead.
    fs : int, or float
        The sampling frequency of the signal.
    d_sig : 1d numpy array, optional
        The input digital signal. If this is the specified input signal rather
        than `sig`, the `adc_gain` and `adc_zero` parameters must be specified.
    adc_gain : int, or float, optional
        The analogue to digital gain of the signal (the number of adus per
        physical unit).
    adc_zero: int, optional
        The value produced by the ADC given a 0 volt input.
    threshold : int, or float, optional
        The relative amplitude detection threshold. Used to initialize the peak
        and qrs detection threshold.
    hr : int, or float, optional
        Typical heart rate, in beats per minute.
    RRdelta : int or float, optional
        Typical difference between successive RR intervals in seconds.
    RRmin : int or float, optional
        Minimum RR interval ("refractory period"), in seconds.
    RRmax : int or float, optional
        Maximum RR interval, in seconds. Thresholds will be adjusted if no
        peaks are detected within this interval.
    QS : int or float, optional
        Typical QRS duration, in seconds.
    QT : int or float, optional
        Typical QT interval, in seconds.
    RTmin : int or float, optional
        Minimum interval between R and T peaks, in seconds.
    RTmax : int or float, optional
        Maximum interval between R and T peaks, in seconds.
    QRSa : int or float, optional
        Typical QRS peak-to-peak amplitude, in microvolts.
    QRSamin : int or float, optional
        Minimum QRS peak-to-peak amplitude, in microvolts.

    Returns
    -------
    qrs_locs : numpy array
        Detected qrs locations


    Notes
    -----
    This function should not be used for signals with fs <= 50Hz

    The algorithm theoretically works as follows:

    - Load in configuration parameters. They are used to set/initialize the:

      * allowed rr interval limits (fixed)
      * initial recent rr interval (running)
      * qrs width, used for detection filter widths (fixed)
      * allowed rt interval limits (fixed)
      * initial recent rt interval (running)
      * initial peak amplitude detection threshold (running)
      * initial qrs amplitude detection threshold (running)
      * `Note`: this algorithm does not normalize signal amplitudes, and
        hence is highly dependent on configuration amplitude parameters.

    - Apply trapezoid low-pass filtering to the signal
    - Convolve a QRS matched filter with the filtered signal
    - Run the learning phase using a calculated signal length: detect qrs and
      non-qrs peaks as in the main detection phase, without saving the qrs
      locations. During this phase, running parameters of recent intervals
      and peak/qrs thresholds are adjusted.
    - Run the detection::
        if a sample is bigger than its immediate neighbors and larger
        than the peak detection threshold, it is a peak.
            if it is further than RRmin from the previous qrs, and is a
            *primary peak.
                if it is further than 2 standard deviations from the
                previous qrs, do a backsearch for a missed low amplitude
                beat
                    return the primary peak between the current sample
                    and the previous qrs if any.
                if it surpasses the qrs threshold, it is a qrs complex
                    save the qrs location.
                    update running rr and qrs amplitude parameters.
                    look for the qrs complex's t-wave and mark it if
                    found.
        else if it is not a peak
            lower the peak detection threshold if the last peak found
            was more than RRmax ago, and not already at its minimum.

    *A peak is secondary if there is a larger peak within its neighborhood
    (time +- rrmin), or if it has been identified as a T-wave associated with a
    previous primary peak. A peak is primary if it is largest in its neighborhood,
    or if the only larger peaks are secondary.

    The above describes how the algorithm should theoretically work, but there
    are bugs which make the program contradict certain parts of its supposed
    logic. A list of issues from the original c, code and hence this python
    implementation can be found here:

    https://github.com/bemoody/wfdb/issues/17

    gqrs will not be supported/developed in this library.

    Examples
    --------
    >>> import numpy as np
    >>> import wfdb
    >>> from wfdb import processing

    >>> # Detect using a physical input signal
    >>> record = wfdb.rdrecord('sample-data/100', channels=[0])
    >>> qrs_locs = processing.gqrs_detect(record.p_signal[:,0], fs=record.fs)

    >>> # Detect using a digital input signal
    >>> record_2 = wfdb.rdrecord('sample-data/100', channels=[0], physical=False)
    >>> qrs_locs_2 = processing.gqrs_detect(d_sig=record_2.d_signal[:,0],
                                            fs=record_2.fs,
                                            adc_gain=record_2.adc_gain[0],
                                            adc_zero=record_2.adc_zero[0])

    """
    # Perform adc if input signal is physical
    if sig is not None:
        record = Record(p_signal=sig.reshape([-1,1]), fmt=['24'])
        record.set_d_features(do_adc=True)
        d_sig = record.d_signal[:,0]
        adc_zero = 0
        adc_gain = record.adc_gain[0]

    conf = GQRS.Conf(fs=fs, adc_gain=adc_gain, hr=hr, RRdelta=RRdelta, RRmin=RRmin,
                RRmax=RRmax, QS=QS, QT=QT, RTmin=RTmin, RTmax=RTmax, QRSa=QRSa,
                QRSamin=QRSamin, thresh=threshold)
    gqrs = GQRS()

    annotations = gqrs.detect(x=d_sig, conf=conf, adc_zero=adc_zero)

    return np.array([a.time for a in annotations])
