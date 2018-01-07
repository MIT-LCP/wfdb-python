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
    Configuration object
    """
    def __init__(self, fs,

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
        self.fs = fs

        self.samps_per_sec = seconds_to_samples(1, fs)
        self.samps_per_min = 60 * self.samps_per_sec

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

        # Adaptive parameters
        # These are updated on the fly, tracking latest r-r stats
        self.rr_mean = int(self.rr * self.samps_per_sec)
        self.rr_dev = int(self.rr_delta * self.samps_per_sec)
        self.rr_min = int(self.rr_min * self.samps_per_sec)
        self.rr_max = int(self.rr_max * self.samps_per_sec)

        self.rr_inc = int(self.rr_mean / 40)

        if self.rr_inc < 1:
            self.rr_inc = 1

        self.dt = int(self.qs * self.samps_per_sec / 4)
        if self.dt < 1:
            self.dt = 1
            print("Warning: sampling rate may be too low!")




        self.rt_min = int(self.rt_min * self.samps_per_sec)
        self.rt_mean = int(0.75 * self.qt * self.samps_per_sec)
        self.rt_max = int(self.rt_max * self.samps_per_sec)




        dv = gain * self.qrs_amp_min * 0.001
        self.pthr = int((self.thresh * dv * dv) / 6)
        self.qthr = self.pthr << 1
        self.pthmin = self.pthr >> 2
        self.qthmin = int((self.pthmin << 2) / 3)
        self.tamean = self.qthr  # initial value for mean T-wave amp

        # Filter constants and thresholds.
        self.dt2 = 2 * self.dt
        self.dt3 = 3 * self.dt
        self.dt4 = 4 * self.dt

        self.smdt = self.dt
        self.v1norm = self.smdt * self.dt * 64

        self.smt = 0
        self.smt0 = 0 + self.smdt


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
    Detector class
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

        self.conf = conf

        # List of annotations made
        self.annotations = []
        self.valid_sample = False

        # Smoothed value
        self.smv = np.zeros(_BUFFER_LENGTH, dtype='int64')
        # The qrs filtered value
        self.qfv = np.zeros(_BUFFER_LENGTH, dtype='int64')



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

    def detect(self):
        """
        Detect all qrs complexes
        """

        # Check shape of input array
        if len(self.sig) < 1:
            return []

        if self.conf.fs < 50:
            raise Warning('Sampling rate is lower than 50Hz recommended threshold.')




        self.v1 = 0

        sampfrom = 0
        self.sampto = len(x) - 1
        self.t = 0 - self.c.dt4

        self.annot = Annotation(0, "NOTE", 0, 0)

        # Cicular buffer of Peaks
        first_peak = Peak(0, 0, 0)
        tmp = first_peak
        for _ in range(1, _N_PEAKS_BUFFERED):
            tmp.next_peak = Peak(0, 0, 0)
            tmp.next_peak.prev_peak = tmp
            tmp = tmp.next_peak
        tmp.next_peak = first_peak
        first_peak.prev_peak = tmp
        self.current_peak = first_peak

        if self.c.samps_per_min > self.c._BUFLN:
            if self.sampto - sampfrom > self.c._BUFLN:
                tf_learn = sampfrom + self.c._BUFLN - self.c.dt4
            else:
                tf_learn = self.sampto - self.c.dt4
        else:
            if self.sampto - sampfrom > self.c.samps_per_min:
                tf_learn = sampfrom + self.c.samps_per_min - self.c.dt4
            else:
                tf_learn = self.sampto - self.c.dt4

        self.countdown = -1
        self.state = "LEARNING"
        self.gqrs(sampfrom, tf_learn)

        self.rewind_gqrs()

        self.state = "RUNNING"
        self.t = sampfrom - self.c.dt4
        self.gqrs(sampfrom, self.sampto)

        return self.annotations

    def rewind_gqrs(self):
        self.countdown = -1
        self.at(self.t)
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

    def smv_at(self, t):
        return self.smv[t & (self.c._BUFLN - 1)]

    def smv_put(self, t, v):
        self.smv[t & (self.c._BUFLN - 1)] = v

    def qfv_at(self, t):
        return self.qfv[t & (self.c._BUFLN - 1)]

    def qfv_put(self, t, v):
        self.qfv[t & (self.c._BUFLN - 1)] = v


    # sm is applied first, then qf
    def sm(self, at_t):
        # implements a trapezoidal low pass (smoothing) filter
        # (with a gain of 4*smdt) applied to input signal sig
        # before the QRS matched filter qf().
        # Before attempting to 'rewind' by more than BUFLN-smdt
        # samples, reset smt and smt0.
        smt = self.c.smt
        smdt = int(self.c.smdt)

        v = 0
        while at_t > smt:
            smt += 1
            if smt > int(self.c.smt0):
                tmp = int(self.smv_at(smt - 1) + \
                             self.at(smt + smdt) + self.at(smt + smdt - 1) - \
                             self.at(smt - smdt) - self.at(smt - smdt - 1))
                self.smv_put(smt, tmp)
            else:
                v = int(self.at(smt))
                for j in range(1, smdt):
                    smtpj = self.at(smt + j)
                    smtlj = self.at(smt - j)
                    v += int(smtpj + smtlj)
                self.smv_put(smt, (v << 1) + self.at(smt + j+1) + self.at(smt - j-1) - \
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



    def gqrs(self, sampfrom, sampto):
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

        r = None
        next_minute = 0
        minutes = 0
        while self.t <= sampto + self.c.samps_per_sec:
            if self.countdown < 0:
                if self.sample_valid:
                    self.qf()
                else:
                    self.countdown = int(seconds_to_samples(1, self.c.fs))
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
                while p.time < self.t - self.c.rt_max:
                    if p.time >= self.annot.time + self.c.rr_min and peaktype(p) == 1:
                        if p.amp > self.c.qthr:
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
                                self.add_ann(self.annot)
                                self.annot.time += self.c.dt2

                            # look for this beat's T-wave
                            tw = None
                            rtdmin = self.c.rt_mean
                            q = p.next_peak
                            while q.time > self.annot.time:
                                rt = q.time - self.annot.time - self.c.dt2
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
                                tmp_time = tw.time - self.c.dt2
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
                        elif self.t - last_qrs > self.c.rr_max and self.c.qthr > self.c.qthmin:
                            self.c.qthr -= (self.c.qthr >> 4)
                    # end:
                    p = p.next_peak
            elif self.t - last_peak > self.c.rr_max and self.c.pthr > self.c.pthmin:
                self.c.pthr -= (self.c.pthr >> 4)

            self.t += 1
            if self.t >= next_minute:
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


def gqrs_detect(sig, fs, conf=Conf(fs)):
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
    conf = Conf(fs=fs, hr=hr,
                rr_delta=rr_delta, rr_min=rr_min, rr_max=rr_max,
                qs=qs, qt=qt,
                rt_min=rt_min, rt_max=rt_max,
                qrs_amp=qrs_amp, qrs_amp_min=qrs_amp_min,
                thresh=threshold)

    gqrs = GQRS()
    annotations = gqrs.detect(sig=sig, conf=conf, adc_zero=adc_zero)
    return np.array([a.sample for a in annotations])
