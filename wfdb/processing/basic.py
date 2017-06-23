import numpy
from scipy import signal

from wfdb import Annotation


def resample_ann(tt, annsamp):
    # tt: numpy.array as returned by signal.resample
    # annsamp: numpy.array containing indexes of annotations (Annotation.annsamp)

    # Compute the new annotation indexes

    result = numpy.zeros(len(tt), dtype='bool')
    j = 0
    tprec = tt[j]
    for i, v in enumerate(annsamp):
        while True:
            d = False
            if j+1 == len(tt):
                result[j] = 1
                break
            tnow = tt[j+1]
            if tprec <= v and v <= tnow:
                if v-tprec < tnow-v:
                    result[j] = 1
                else:
                    result[j+1] = 1
                d = True
            j += 1
            tprec = tnow
            if d:
                break
    return numpy.where(result==1)[0].astype('int64')


def resample_sig(x, fs, fs_target):
    # x: a numpy.array containing the signal
    # fs: the current frequency
    # fs_target: the target frequency

    # Resample a signal

    t = numpy.arange(x.shape[0]).astype('float64')

    if fs == fs_target:
        return x, t

    new_length = int(x.shape[0]*fs_target/fs)
    xx, tt = signal.resample(x, num=new_length, t=t)
    assert xx.shape == tt.shape and xx.shape[0] == new_length
    assert numpy.all(numpy.diff(tt) > 0)
    return xx, tt


def resample_singlechan(x, ann, fs, fs_target):
    # x: a numpy.array containing the signal
    # ann: an Annotation object
    # fs: the current frequency
    # fs_target: the target frequency

    # Resample a single-channel signal with its annotations

    xx, tt = resample_sig(x, fs, fs_target)

    new_annsamp = resample_ann(tt, ann.annsamp)
    assert ann.annsamp.shape == new_annsamp.shape

    new_ann = Annotation(ann.recordname, ann.annotator, new_annsamp, ann.anntype, ann.num, ann.subtype, ann.chan, ann.aux, ann.fs)
    return xx, new_ann


def resample_multichan(xs, ann, fs, fs_target, resamp_ann_chan=0):
    # xs: a numpy.ndarray containing the signals as returned by wfdb.srdsamp
    # ann: an Annotation object
    # fs: the current frequency
    # fs_target: the target frequency
    # resample_ann_channel: the signal channel that is used to compute new annotation indexes

    # Resample multiple channels with their annotations

    assert resamp_ann_chan < xs.shape[1]

    lx = []
    lt = None
    for chan in range(xs.shape[1]):
        xx, tt = resample_sig(xs[:, chan], fs, fs_target)
        lx.append(xx)
        if chan == resamp_ann_chan:
            lt = tt

    new_annsamp = resample_ann(lt, ann.annsamp)
    assert ann.annsamp.shape == new_annsamp.shape

    new_ann = Annotation(ann.recordname, ann.annotator, new_annsamp, ann.anntype, ann.num, ann.subtype, ann.chan, ann.aux, ann.fs)
    return numpy.column_stack(lx), new_ann


def normalize(x, lb=0, ub=1):
    # lb: Lower bound
    # ub: Upper bound

    # Resizes a signal between the lower and upper bound

    mid = ub - (ub - lb) / 2
    min_v = numpy.min(x)
    max_v = numpy.max(x)
    mid_v =  max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return x * coef - (mid_v * coef) + mid


def smooth(x, window_size):
    box = numpy.ones(window_size)/window_size
    return numpy.convolve(x, box, mode='same')
