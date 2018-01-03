import numpy
from scipy import signal

from wfdb import Annotation


def resample_ann(resampled_t, ann_sample):
    """
    Compute the new annotation indices
    
    Parameters
    ----------
    resampled_t : numpy array
        Array of signal locations as returned by scipy.signal.resample
    ann_sample : numpy array
        Array of annotation locations
    
    Returns
    -------
    resampled_ann_sample : numpy array
        Array of resampled annotation locations

    """
    tmp = numpy.zeros(len(resampled_t), dtype='int16')
    j = 0
    tprec = resampled_t[j]
    for i, v in enumerate(ann_sample):
        while True:
            d = False
            if v < tprec:
                j -= 1
                tprec = resampled_t[j]
                
            if j+1 == len(resampled_t):
                tmp[j] += 1
                break
            
            tnow = resampled_t[j+1]
            if tprec <= v and v <= tnow:
                if v-tprec < tnow-v:
                    tmp[j] += 1
                else:
                    tmp[j+1] += 1
                d = True
            j += 1
            tprec = tnow
            if d:
                break
                
    idx = numpy.where(tmp>0)[0].astype('int64')
    res = []
    for i in idx:
        for j in range(tmp[i]):
            res.append(i)
    assert len(res) == len(ann_sample)

    return numpy.asarray(res, dtype='int64')


def resample_sig(x, fs, fs_target):
    """
    Resample a signal to a different frequency.
    
    Parameters
    ----------
    x : numpy array
        Array containing the signal
    fs : int, or float
        The original sampling frequency
    fs_target : int, or float
        The target frequency
    
    Returns
    -------
    resampled_x : numpy array
        Array of the resampled signal values
    resampled_t : numpy array
        Array of the resampled signal locations

    """

    t = numpy.arange(x.shape[0]).astype('float64')

    if fs == fs_target:
        return x, t

    new_length = int(x.shape[0]*fs_target/fs)
    resampled_x, resampled_t = signal.resample(x, num=new_length, t=t)
    assert resampled_x.shape == resampled_t.shape and resampled_x.shape[0] == new_length
    assert numpy.all(numpy.diff(resampled_t) > 0)
    
    return resampled_x, resampled_t


def resample_singlechan(x, ann, fs, fs_target):
    """
    Resample a single-channel signal with its annotations
    
    Parameters
    ----------
    x: numpy array
        The signal array
    ann : wfdb Annotation
        The wfdb annotation object
    fs : int, or float
        The original frequency
    fs_target : int, or float
        The target frequency

    Returns
    -------
    resampled_x : numpy array
        Array of the resampled signal values
    resampled_ann : wfdb Annotation
        Annotation containing resampled annotation locations

    """

    resampled_x, resampled_t = resample_sig(x, fs, fs_target)

    new_sample = resample_ann(resampled_t, ann.sample)
    assert ann.sample.shape == new_sample.shape

    resampled_ann = Annotation(ann.record_name, ann.extension, new_sample,
        ann.symbol, ann.num, ann.subtype, ann.chan, ann.aux_note, ann.fs)

    return resampled_x, resampled_ann


def resample_multichan(xs, ann, fs, fs_target, resamp_ann_chan=0):
    """
    Resample multiple channels with their annotations

    Parameters
    ----------
    xs: numpy array
        The signal array
    ann : wfdb Annotation
        The wfdb annotation object
    fs : int, or float
        The original frequency
    fs_target : int, or float
        The target frequency
    resample_ann_channel : int, optional
        The signal channel used to compute new annotation indices
    
    Returns
    -------
    resampled_xs : numpy array
        Array of the resampled signal values
    resampled_ann : wfdb Annotation
        Annotation containing resampled annotation locations

    """
    assert resamp_ann_chan < xs.shape[1]

    lx = []
    lt = None
    for chan in range(xs.shape[1]):
        resampled_x, resampled_t = resample_sig(xs[:, chan], fs, fs_target)
        lx.append(resampled_x)
        if chan == resamp_ann_chan:
            lt = resampled_t

    new_sample = resample_ann(lt, ann.sample)
    assert ann.sample.shape == new_sample.shape

    resampled_ann = Annotation(ann.record_name, ann.extension, new_sample, ann.symbol,
        ann.num, ann.subtype, ann.chan, ann.aux_note, ann.fs)

    return numpy.column_stack(lx), resampled_ann


def normalize(x, lb=0, ub=1):
    """
    Normalize a signal between the lower and upper bound
    
    Parameters
    ----------
    x : numpy array
        Original signal to be normalized
    lb : int, or float
        Lower bound
    ub : int, or float
        Upper bound

    Returns
    -------
    x_normalized : numpy array
        Normalized signal
    
    """

    mid = ub - (ub - lb) / 2
    min_v = numpy.min(x)
    max_v = numpy.max(x)
    mid_v =  max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return x * coef - (mid_v * coef) + mid


def smooth(x, window_size):
    box = numpy.ones(window_size)/window_size
    return numpy.convolve(x, box, mode='same')
