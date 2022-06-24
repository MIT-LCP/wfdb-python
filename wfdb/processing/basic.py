import numpy as np
import pandas as pd
from scipy import signal

from wfdb.io.annotation import Annotation


def resample_ann(ann_sample, fs, fs_target):
    """
    Compute the new annotation indices.

    Parameters
    ----------
    ann_sample : ndarray
        Array of annotation locations.
    fs : int
        The starting sampling frequency.
    fs_target : int
        The desired sampling frequency.

    Returns
    -------
    ndarray
        Array of resampled annotation locations.

    """
    ratio = fs_target / fs
    return (ratio * ann_sample).astype(np.int64)


def resample_sig(x, fs, fs_target):
    """
    Resample a signal to a different frequency.

    Parameters
    ----------
    x : ndarray
        Array containing the signal.
    fs : int, float
        The original sampling frequency.
    fs_target : int, float
        The target frequency.

    Returns
    -------
    resampled_x : ndarray
        Array of the resampled signal values.
    resampled_t : ndarray
        Array of the resampled signal locations.

    """
    t = np.arange(x.shape[0]).astype("float64")

    if fs == fs_target:
        return x, t

    new_length = int(x.shape[0] * fs_target / fs)
    # Resample the array if NaN values are present
    if np.isnan(x).any():
        x = pd.Series(x.reshape((-1,))).interpolate().values
    resampled_x, resampled_t = signal.resample(x, num=new_length, t=t)
    assert (
        resampled_x.shape == resampled_t.shape
        and resampled_x.shape[0] == new_length
    )
    assert np.all(np.diff(resampled_t) > 0)

    return resampled_x, resampled_t


def resample_singlechan(x, ann, fs, fs_target):
    """
    Resample a single-channel signal with its annotations.

    Parameters
    ----------
    x: ndarray
        The signal array.
    ann : WFDB Annotation
        The WFDB annotation object.
    fs : int, float
        The original frequency.
    fs_target : int, float
        The target frequency.

    Returns
    -------
    resampled_x : ndarray
        Array of the resampled signal values.
    resampled_ann : WFDB Annotation
        Annotation containing resampled annotation locations.

    """
    resampled_x, _ = resample_sig(x, fs, fs_target)
    new_sample = resample_ann(ann.sample, fs, fs_target)

    resampled_ann = Annotation(
        record_name=ann.record_name,
        extension=ann.extension,
        sample=new_sample,
        symbol=ann.symbol,
        subtype=ann.subtype,
        chan=ann.chan,
        num=ann.num,
        aux_note=ann.aux_note,
        fs=fs_target,
    )

    return resampled_x, resampled_ann


def resample_multichan(xs, ann, fs, fs_target, resamp_ann_chan=0):
    """
    Resample multiple channels with their annotations.

    Parameters
    ----------
    xs: ndarray
        The signal array.
    ann : WFDB Annotation
        The WFDB annotation object.
    fs : int, float
        The original frequency.
    fs_target : int, float
        The target frequency.
    resample_ann_channel : int, optional
        The signal channel used to compute new annotation indices.

    Returns
    -------
    ndarray
        Array of the resampled signal values.
    resampled_ann : WFDB Annotation
        Annotation containing resampled annotation locations.

    """
    assert resamp_ann_chan < xs.shape[1]

    lx = []
    for chan in range(xs.shape[1]):
        resampled_x, _ = resample_sig(xs[:, chan], fs, fs_target)
        lx.append(resampled_x)

    new_sample = resample_ann(ann.sample, fs, fs_target)

    resampled_ann = Annotation(
        record_name=ann.record_name,
        extension=ann.extension,
        sample=new_sample,
        symbol=ann.symbol,
        subtype=ann.subtype,
        chan=ann.chan,
        num=ann.num,
        aux_note=ann.aux_note,
        fs=fs_target,
    )

    return np.column_stack(lx), resampled_ann


def normalize_bound(sig, lb=0, ub=1):
    """
    Normalize a signal between the lower and upper bound.

    Parameters
    ----------
    sig : ndarray
        Original signal to be normalized.
    lb : int, float, optional
        Lower bound.
    ub : int, float, optional
        Upper bound.

    Returns
    -------
    ndarray
        Normalized signal.

    """
    mid = ub - (ub - lb) / 2
    min_v = np.min(sig)
    max_v = np.max(sig)
    mid_v = max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return sig * coef - (mid_v * coef) + mid


def smooth(sig, window_size):
    """
    Apply a uniform moving average filter to a signal.

    Parameters
    ----------
    sig : ndarray
        The signal to smooth.
    window_size : int
        The width of the moving average filter.

    Returns
    -------
    ndarray
        The convolved input signal with the desired box waveform.

    """
    box = np.ones(window_size) / window_size
    return np.convolve(sig, box, mode="same")


def get_filter_gain(b, a, f_gain, fs):
    """
    Given filter coefficients, return the gain at a particular
    frequency.

    Parameters
    ----------
    b : list
        List of linear filter b coefficients.
    a : list
        List of linear filter a coefficients.
    f_gain : int, float, optional
        The frequency at which to calculate the gain.
    fs : int, float, optional
        The sampling frequency of the system.

    Returns
    -------
    gain : int, float
        The passband gain at the desired frequency.

    """
    # Save the passband gain
    w, h = signal.freqz(b, a)
    w_gain = f_gain * 2 * np.pi / fs

    ind = np.where(w >= w_gain)[0][0]
    gain = abs(h[ind])

    return gain


def normalize(X):
    """
    Scale input vector to unit norm (vector length).

    Parameters
    ----------
    X : ndarray
        The vector to normalize.

    Returns
    -------
    ndarray
        The normalized vector.

    """
    return X / np.linalg.norm(X)
