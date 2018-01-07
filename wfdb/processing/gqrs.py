import numpy as np
import copy
"""

"""

# The buffer length. Must be a power of 2.
_BUFFER_LENGTH = 32768
# Number of peaks buffered per signal
_N_PEAKS_BUFFERED = 64

        # self._NORMAL = 1  # normal beat
        # self._ARFCT = 16  # isolated QRS-like artifact
        # self._NOTE = 22  # comment annotation
        # self._TWAVE = 27  # T-wave peak

def seconds_to_samples(seconds, fs):
    """
    Convert number of seconds to number of samples, given a sampling frequency

    But this should be a float...
    """
    return int(seconds * fs + 0.5)


class Conf(object):
    """
    ecg Configuration object. Stores static ish information about the input
    ecg.
    """
    def __init__(self,

                 hr=75,

                 rr_delta=0.2, rr_min=0.28, rr_max=2.4,

                 qs=0.07, qt=0.35,

                 rt_min=0.25, rt_max=0.33,

                 qrs_amp=750, qrs_amp_min=130,

                 thresh=1.0):
        """
        
        hr : int, or float, optional
            Typical heart rate, in beats per minute.
        rr_delta : int or float, optional
            Typical difference between successive rr intervals in seconds.
        rr_min : int or float, optional
            Minimum rr interval ("refractory period"), in seconds.
        rr_max : int or float, optional
            Maximum rr interval, in seconds. Thresholds will be adjusted if no
            peaks are detected within this interval.
        qs : int or float, optional
            Typical QRS duration, in seconds.
        qt : int or float, optional
            Typical qt interval, in seconds.
        rt_min : int or float, optional
            Minimum interval between R and T peaks, in seconds.
        rt_max : int or float, optional
            Maximum interval between R and T peaks, in seconds.
        qrs_amp : int or float, optional
            Typical QRS peak-to-peak amp, in microvolts.
        qrs_amp_min : int or float, optional
            Minimum QRS peak-to-peak amp, in microvolts.
        threshold : int, or float, optional
            The normalized detection threshold
        """

        # Static ish parameters

        self.hr = hr
        # Typical r-r interval in seconds
        self.rr = 60.0 / self.hr
        self.rr_delta = rr_delta
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.qs = qs
        self.qt = qt
        self.rt_min = rt_min
        self.rt_max = rt_max
        self.qrs_amp = qrs_amp
        self.qrs_amp_min = qrs_amp_min
        self.thresh = thresh




class Peak(object):
    """
    Contains information about a local maximum in the filtered signal
    """
    def __init__(self, sample, amp, peak_type):
        """
        type:
            1 = primary, 2 = secondary, 0 = undetermined

        A peak is secondary if there is a larger peak within +/- rr_min, or
        if it has been identified as a t-wave associated with a previous
        primary peak.

        A peak is primary if it is the largest within +/- rr_min, or if
        the only larger peaks are secondary.
        """
        self.sample = sample
        self.amp = amp
        self.peak_type = peak_type




class Annotation(object):
    def __init__(self, sample, symbol, subtype, num):
        self.sample = sample
        self.symbol = symbol
        self.subtype = subtype
        self.num = num


class GQRS(object):
    """
    Detector class. Stores updating information about the signal and its
    detections.
    """

    def __init__(self, sig, fs, conf):
        """
        sig : numpy array
            Signal
        fs : int, or float
            Sampling frequency of signal
        conf : Conf object
            Configuration object with parameters.
        """
        self.sig = sig
        self.fs = fs
        self.samps_per_sec = seconds_to_samples(1, fs)
        self.samps_per_min = 60 * self.samps_per_sec
        self.conf = conf

        if self.fs < 50:
            raise Warning('Sampling rate is lower than 50Hz recommended limit.')

        # List of annotations made
        self.annotations = []
        self.valid_sample = False

        # Smoothed signal
        self.sig_smoothed = np.zeros(_BUFFER_LENGTH, dtype='int64')
        # The qrs filtered signal
        self.sig_filtered = np.zeros(_BUFFER_LENGTH, dtype='int64')


        # Initialize adaptive parameters using a-priori parameters

        # Recent average r-r interval, in samples
        self.rr_mean = int(conf.rr * self.samps_per_sec)
        # Latest r-r interval, in samples
        self.rr_dev = int(conf.rr_delta * self.samps_per_sec)
        # Recent minimum r-r interval, in samples
        self.rr_min = int(conf.rr_min * self.samps_per_sec)
        # Recent maximum r-r interval, in samples
        self.rr_max = int(conf.rr_max * self.samps_per_sec)
        # Maximum incremental change in rr_mean
        self.rr_inc = int(self.rr_mean / 40)

        if self.rr_inc < 1:
            self.rr_inc = 1



        self.rt_min = int(conf.rt_min * self.samps_per_sec)
        self.rt_mean = int(0.75 * conf.qt * self.samps_per_sec)
        self.rt_max = int(self.rt_max * self.samps_per_sec)

        # What is this?
        self.dv = gain * self.qrs_amp_min * 0.001

        # Peak detection threshold
        self.peak_thresh = int((conf.thresh * self.dv**2) / 6)
        # QRS detection threshold
        self.qrs_thresh = self.peak_thresh << 1
        # Minimum peak detection threshold
        self.peak_thresh_min = self.peak_thresh >> 2
        # Minimum QRS detection threshold
        self.qrs_thresh_min = int((self.peak_thresh_min << 2) / 3)
        
        # T-wave amplitude
        self.t_amp_mean = self.qrs_thresh  


        # Filter constants and thresholds.
        # dt variables are measured in sample ntervals
        self.dt = int(self.qs * self.samps_per_sec / 4)
        if self.dt < 1:
            self.dt = 1

        self.dt2 = 2 * self.dt
        self.dt3 = 3 * self.dt
        self.dt4 = 4 * self.dt

        self.smooth_dt = self.dt
        # v1 is integral of dv in qrs_filter. v1norm is normalization for v1
        self.v1norm = self.smooth_dt * self.dt * 64

        # sample number for smoothed output
        self.smooth_samp = 0
        self.smooth_samp0 = 0 + self.smooth_dt





    def add_ann(self, annotation):
        """
        Add an annotation to the entire collection
        """
        self.annotations.append(copy.deepcopy(annotation))

    def get_peak_type(self):
        # returns 1 if p is the most prominent peak in its neighborhood, 2
        # otherwise.  The neighborhood consists of all other peaks within rr_min.
        # Normally, "most prominent" is equivalent to "largest in amp", but this
        # is not always true.  For example, consider three consecutive peaks a, b, c
        # such that a and b share a neighborhood, b and c share a neighborhood, but a
        # and c do not; and suppose that amp(a) > amp(b) > amp(c).  In this case, if
        # there are no other peaks, a is the most prominent peak in the (a, b)
        # neighborhood.  Since b is thus identified as a non-prominent peak, c becomes
        # the most prominent peak in the (b, c) neighborhood.  This is necessary to
        # permit detection of low-amp beats that closely precede or follow beats
        # with large secondary peaks (as, for example, in R-on-T PVCs).
        if self.type:
            return self.type
        else:
            amp = self.amp
            sampfrom = p.time - self.c.rr_min
            t1 = p.time + self.c.rr_min

            if sampfrom < 0:
                sampfrom = 0

            pp = p.prev_peak
            while sampfrom < pp.time and pp.time < pp.next_peak.time:
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



    def detect(self, sampfrom=0, sampto='end'):
        """
        Detect all qrs complexes. This is like void main.

        Can add sampfrom and sampto as input arguments
        """

        # Check shape of input array
        # Return if necessary




        # Integral of dv in qrs_filter
        self.v1 = 0

        if sampto == 'end':
            sampto = len(self.sig) - 1

        # The current sample number being analyzed
        self.sampnum = 0 - self.dt4



        # Initialize the circular peak of buffers
        self.peak_buffer = [Peak(0, 0, 0)] * _N_PEAKS_BUFFERED
        

        # Determine the length of the signal to learn from
  
        # If buffer can't store 1 min of samples (high fs)
        if self.samps_per_min > self.c._BUFLN:
            # Signal length is greater than buffer length. Use buffln samples.
            if sampto - sampfrom > self.c._BUFLN:
                sampto_learn = sampfrom + self.c._BUFLN - self.dt4
            else:
                # Use all samples to learn
                sampto_learn = sampto - self.dt4
        # 1 min fits into buffer (normal/low fs)
        else:
            # Signal is longer than a minute. Use a minute.
            if sampto - sampfrom > self.samps_per_min:
                sampto_learn = sampfrom + self.samps_per_min - self.dt4
            # Signal shorter than a minute. Use whole signal.
            else:
                sampto_learn = sampto - self.dt4


        # Learning
        self.gqrs_detect(sampfrom, sampto_learn)
        # Rewind
        self.rewind_gqrs()


        # Do detection
        # Reset sample number
        self.sampnum = sampfrom - self.dt4
        self.gqrs_detect(sampfrom, self.sampto)

        return self.annotations


    def rewind_gqrs(self):
        self.countdown = -1
        self.at(self.sampnum)
        self.annot.time = 0
        self.annot.type = "NORMAL"
        self.annot.subtype = 0
        self.annot.num = 0
        p = self.current_peak
        for _ in range(_N_PEAKS_BUFFERED):
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

    # Get the smoothed filter value
    def smv_at(self, t):
        return self.smv[t & (self.c._BUFLN - 1)]

    def smv_put(self, t, v):
        self.smv[t & (self.c._BUFLN - 1)] = v

    # Get the qrs filtered value
    def qfv_at(self, t):
        return self.qfv[t & (self.c._BUFLN - 1)]

    def qfv_put(self, t, v):
        self.qfv[t & (self.c._BUFLN - 1)] = v


    # smooth_filter is applied first, then qrs_filter
    def smooth_filter(self, at_t):
        """
        Implements a trapezoidal low pass (smoothing) filter (with a gain of
        4*smooth_dt) applied to input signal sig before the QRS matched filter
        qrs_filter().
        
        Before attempting to 'rewind' by more than BUFLN-smooth_dt
        samples, reset smooth_samp and smooth_samp0.
        """
        smooth_samp = self.c.smooth_samp
        smooth_dt = int(self.c.smooth_dt)

        v = 0
        while at_t > smooth_samp:
            smooth_samp += 1
            if smooth_samp > int(self.c.smooth_samp0):
                tmp = int(self.smv_at(smooth_samp - 1) + \
                             self.at(smooth_samp + smooth_dt) + self.at(smooth_samp + smooth_dt - 1) - \
                             self.at(smooth_samp - smooth_dt) - self.at(smooth_samp - smooth_dt - 1))
                self.smv_put(smooth_samp, tmp)
            else:
                v = int(self.at(smooth_samp))
                for j in range(1, smooth_dt):
                    smooth_samppj = self.at(smooth_samp + j)
                    smooth_samplj = self.at(smooth_samp - j)
                    v += int(smooth_samppj + smooth_samplj)
                self.smv_put(smooth_samp, (v << 1) + self.at(smooth_samp + j+1) + self.at(smooth_samp - j-1) - \
                             self.adc_zero * (smooth_dt << 2))
        self.c.smooth_samp = smooth_samp
        return self.smv_at(at_t)


    def qrs_filter(self):
        # evaluate the QRS detector filter for the next sample

        # do this first, to ensure that all of the other smoothed values needed below are in the buffer
        dv2 = self.smooth_filter(self.sampnum + self.dt4)
        dv2 -= self.smv_at(self.sampnum - self.dt4)
        dv1 = int(self.smv_at(self.sampnum + self.dt) - self.smv_at(self.sampnum - self.dt))
        dv = dv1 << 1
        dv -= int(self.smv_at(self.sampnum + self.dt2) - self.smv_at(self.sampnum - self.dt2))
        dv = dv << 1
        dv += dv1
        dv -= int(self.smv_at(self.sampnum + self.dt3) - self.smv_at(self.sampnum - self.dt3))
        dv = dv << 1
        dv += dv2
        self.v1 += dv
        v0 = int(self.v1 / self.c.v1norm)
        self.qfv_put(self.sampnum, v0 * v0)


    def add_peak(peak_time, peak_amp, type):
        p = self.current_peak.next_peak
        p.time = peak_time
        p.amp = peak_amp
        p.type = type
        self.current_peak = p
        p.next_peak.amp = 0


    def find_missing(r, p):
        if r is None or p is None:
            return None

        minrrerr = p.time - r.time

        s = None
        q = r.next_peak
        while q.time < p.time:
            if peaktype(q) == 1:
                rrtmp = q.time - r.time
                rrerr = rrtmp - self.c.rr_mean
                if rrerr < 0:
                    rrerr = -rrerr
                if rrerr < minrrerr:
                    minrrerr = rrerr
                    s = q
            # end:
            q = q.next_peak

        return s


    def gqrs_detect(self, sampfrom, sampto):
        """
        The big function
        """

        # What is this?
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

        last_peak = sampfrom
        last_qrs = sampfrom



        r = None

        next_minute = 0

        minutes = 0


        while self.sampnum <= sampto + self.samps_per_sec:
            if self.countdown < 0:
                if self.sample_valid:
                    self.qrs_filter()
                else:
                    self.countdown = int(seconds_to_samples(1, self.c.fs))
                    self.state = "CLEANUP"
            else:
                self.countdown -= 1
                if self.countdown < 0:
                    break

            q0 = self.qfv_at(self.sampnum)
            q1 = self.qfv_at(self.sampnum - 1)
            q2 = self.qfv_at(self.sampnum - 2)

            # state == RUNNING only
            if q1 > self.c.peak_thresh and q2 < q1 and q1 >= q0 and self.sampnum > self.dt4:
                add_peak(self.sampnum - 1, q1, 0)
                last_peak = self.sampnum - 1
                p = self.current_peak.next_peak
                while p.time < self.sampnum - self.c.rt_max:
                    if p.time >= self.annot.time + self.c.rr_min and peaktype(p) == 1:
                        if p.amp > self.c.qrs_thresh:
                            rr = p.time - self.annot.time
                            q = find_missing(r, p)
                            if rr > self.c.rr_mean + 2 * self.c.rr_dev and \
                               rr > 2 * (self.c.rr_mean - self.c.rr_dev) and \
                               q is not None:
                                p = q
                                rr = p.time - self.annot.time
                                self.annot.subtype = 1
                            rrd = rr - self.c.rr_mean
                            if rrd < 0:
                                rrd = -rrd
                            self.c.rr_dev += (rrd - self.c.rr_dev) >> 3
                            if rrd > self.c.rr_inc:
                                rrd = self.c.rr_inc
                            if rr > self.c.rr_mean:
                                self.c.rr_mean += rrd
                            else:
                                self.c.rr_mean -= rrd
                            if p.amp > self.c.qrs_thresh * 4:
                                self.c.qrs_thresh += 1
                            elif p.amp < self.c.qrs_thresh:
                                self.c.qrs_thresh -= 1
                            if self.c.qrs_thresh > self.c.peak_thresh * 20:
                                self.c.qrs_thresh = self.c.peak_thresh * 20
                            last_qrs = p.time

                            if self.state == "RUNNING":
                                self.annot.time = p.time - self.dt2
                                self.annot.type = "NORMAL"
                                qsize = int(p.amp * 10.0 / self.c.qrs_thresh)
                                if qsize > 127:
                                    qsize = 127
                                self.annot.num = qsize
                                self.add_ann(self.annot)
                                self.annot.time += self.dt2

                            # look for this beat's T-wave
                            tw = None
                            rtdmin = self.c.rt_mean
                            q = p.next_peak
                            while q.time > self.annot.time:
                                rt = q.time - self.annot.time - self.dt2
                                if rt < self.c.rt_min:
                                    # end:
                                    q = q.next_peak
                                    continue
                                if rt > self.c.rt_max:
                                    break
                                rtd = rt - self.c.rt_mean
                                if rtd < 0:
                                    rtd = -rtd
                                if rtd < rtdmin:
                                    rtdmin = rtd
                                    tw = q
                                # end:
                                q = q.next_peak
                            if tw is not None:
                                tmp_time = tw.time - self.dt2
                                tann = Annotation(tmp_time, "TWAVE",
                                                  1 if tmp_time > self.annot.time + self.c.rt_mean else 0, rtdmin)
                                # if self.state == "RUNNING":
                                #     self.add_ann(tann)
                                rt = tann.time - self.annot.time
                                self.c.rt_mean += (rt - self.c.rt_mean) >> 4
                                if self.c.rt_mean > self.c.rt_max:
                                    self.c.rt_mean = self.c.rt_max
                                elif self.c.rt_mean < self.c.rt_min:
                                    self.c.rt_mean = self.c.rr_min
                                tw.type = 2  # mark T-wave as secondary
                            r = p
                            q = None
                            self.annot.subtype = 0
                        elif self.sampnum - last_qrs > self.c.rr_max and self.c.qrs_thresh > self.c.qrs_thresh_min:
                            self.c.qrs_thresh -= (self.c.qrs_thresh >> 4)
                    # end:
                    p = p.next_peak
            elif self.sampnum - last_peak > self.c.rr_max and self.c.peak_thresh > self.c.peak_thresh_min:
                self.c.peak_thresh -= (self.c.peak_thresh >> 4)

            self.sampnum += 1
            if self.sampnum >= next_minute:
                next_minute += self.c.samps_per_min
                minutes += 1
                if minutes >= 60:
                    minutes = 0

        if self.state == "LEARNING":
            return

        # Mark the last beat or two.
        p = self.current_peak.next_peak
        while p.time < p.next_peak.time:
            if p.time >= self.annot.time + self.c.rr_min and p.time < self.sampto and peaktype(p) == 1:
                self.annot.type = "NORMAL"
                self.annot.time = p.time
                self.add_ann(self.annot)
            # end:
            p = p.next_peak





# The gateway function
def gqrs_detect(sig, fs, conf=Conf()):
    """
    Detect qrs locations in a single channel ecg.

    Functionally, a direct port of the gqrs algorithm from the original
    wfdb package. Therefore written to accept wfdb record fields.

    Parameters
    ----------
    sig : np array
        The digital signal.
    fs : int, or float
        The sampling frequency of the signal.

    
    Returns
    -------
    qrs_locs : np array
        Detected qrs locations


    Notes
    -----
    This function should not be used for signals with fs <= 50Hz

    """

    gqrs = GQRS(sig, fs, conf)
    annotations = gqrs.detect()

    return np.array([a.sample for a in annotations])
