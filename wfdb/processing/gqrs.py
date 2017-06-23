import numpy
import copy

def time_to_sample_number(seconds, frequency):
    return seconds * frequency + 0.5


class Conf(object):
    def __init__(self, freq, gain,
                 hr=75,
                 RRdelta=0.2, RRmin=0.28, RRmax=2.4,
                 QS=0.07, QT=0.35,
                 RTmin=0.25, RTmax=0.33,
                 QRSa=750, QRSamin=130,
                 thresh=1.0):
        self.freq = freq

        self.sps = int(time_to_sample_number(1, freq))
        self.spm = int(time_to_sample_number(60, freq))

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
            self.dt = 1
            print("Warning: sampling rate may be too low!")

        self.rtmin = int(self.RTmin * self.sps)
        self.rtmean = int(0.75 * self.QT * self.sps)
        self.rtmax = int(self.RTmax * self.sps)

        dv = gain * self.QRSamin * 0.001
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


class GQRS(object):
    def putann(self, annotation):
        self.annotations.append(copy.deepcopy(annotation))

    def detect(self, x, conf, adczero):
        self.c = conf
        self.annotations = []
        self.sample_valid = False

        if len(x) < 1:
            return []

        self.x = x
        self.adczero = adczero

        self.qfv = numpy.zeros((self.c._BUFLN), dtype="int64")
        self.smv = numpy.zeros((self.c._BUFLN), dtype="int64")
        self.v1 = 0

        t0 = 0
        self.tf = len(x) - 1
        self.t = 0 - self.c.dt4

        self.annot = Annotation(0, "NOTE", 0, 0)

        # Cicular buffer of Peaks
        first_peak = Peak(0, 0, 0)
        tmp = first_peak
        for _ in range(1, self.c._NPEAKS):
            tmp.next_peak = Peak(0, 0, 0)
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

    def sm(self, at_t):  # CHECKED!
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
                             self.adczero * (smdt << 2))
        self.c.smt = smt
        return self.smv_at(at_t)

    def qf(self):  # CHECKED!
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
                    self.countdown = int(time_to_sample_number(1, self.c.freq))
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
                                tann = Annotation(tmp_time, "TWAVE",
                                                  1 if tmp_time > self.annot.time + self.c.rtmean else 0, rtdmin)
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


def gqrs_detect(x, frequency, adcgain, adczero, threshold=1.0,
                hr=75, RRdelta=0.2, RRmin=0.28, RRmax=2.4,
                QS=0.07, QT=0.35, RTmin=0.25, RTmax=0.33,
                QRSa=750, QRSamin=130):
    """
    A signal with a frequency of only 50Hz is not handled by the original algorithm,
    thus it is not recommended to use this algorithm for this case.

    * x: The signal as an array
    * frequency: The signal frequency
    * adcgain: The gain of the signal (the number of adus (q.v.) per physical unit)
    * adczero: The value produced by the ADC given a 0 volt input.
    * threshold: The threshold for detection
    * hr: Typical heart rate, in beats per minute
    * RRdelta: Typical difference between successive RR intervals in seconds
    * RRmin: Minimum RR interval ("refractory period"), in seconds
    * RRmax: Maximum RR interval, in seconds; thresholds will be adjusted if no peaks are detected within this interval
    * QS: Typical QRS duration, in seconds
    * QT: Typical QT interval, in seconds
    * RTmin: Minimum interval between R and T peaks, in seconds
    * RTmax: Maximum interval between R and T peaks, in seconds
    * QRSa: Typical QRS peak-to-peak amplitude, in microvolts
    * QRSamin: Minimum QRS peak-to-peak amplitude, in microvolts
    """
    conf = Conf(freq=frequency, gain=adcgain, hr=hr,
                RRdelta=RRdelta, RRmin=RRmin, RRmax=RRmax,
                QS=QS, QT=QT,
                RTmin=RTmin, RTmax=RTmax,
                QRSa=QRSa, QRSamin=QRSamin,
                thresh=threshold)
    gqrs = GQRS()
    annotations = gqrs.detect(x=x, conf=conf, adczero=adczero)
    return [a.time for a in annotations]
