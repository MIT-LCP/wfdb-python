import copy
import numpy as np

from .basic import smooth

import pdb
def find_peaks(sig):
    """
    Find hard peaks and soft peaks in a signal, defined as follows:
    - Hard peak: a peak that is either /\ or \/
    - Soft peak: a peak that is either /-*\ or \-*/ (In that case we define the
      middle of it as the peak)

    Parameters
    ----------
    sig : np array
        The 1d signal array

    Returns
    -------
    hard_peaks : np array
        Array containing the indices of the hard peaks:
    soft_peaks : np array
        Array containing the indices of the soft peaks

    """
    if len(sig) == 0:
        return np.empty([0]), np.empty([0])

    tmp = sig[1:]
    tmp = np.append(tmp, [sig[-1]])
    tmp = sig - tmp
    tmp[np.where(tmp>0)] = 1
    tmp[np.where(tmp==0)] = 0
    tmp[np.where(tmp<0)] = -1
    tmp2 = tmp[1:]
    tmp2 = np.append(tmp2, [0])
    tmp = tmp-tmp2

    hard_peaks = np.where(np.logical_or(tmp==-2, tmp==+2))[0] + 1
    soft_peaks = []

    for iv in np.where(np.logical_or(tmp==-1,tmp==+1))[0]:
        t = tmp[iv]
        i = iv+1
        while True:
            if i==len(tmp) or tmp[i] == -t or tmp[i] == -2 or tmp[i] == 2:
                break
            if tmp[i] == t:
                soft_peaks.append(int(iv + (i - iv)/2))
                break
            i += 1
    soft_peaks = np.array(soft_peaks, dtype='int') + 1

    return hard_peaks, soft_peaks


def find_local_peaks(sig, radius):
    """
    Find all local peaks in a signal. A sample is a local peak if it is
    the largest value within the <radius> samples on its left and right.

    In cases where it shares the max value with nearby samples, the middle
    sample is classified as the local peak.

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

    return (np.array(peak_inds))


def correct_peaks(sig, old_peak_inds, search_radius, min_gap,
                  smooth_window_size=None):
    """
    Adjust a set of detected peaks to coincide with local signal maxima,
    and

    Parameters
    ----------
    sig : numpy array
        The 1d signal array
    peak_inds : np array
        Array of the original peak indices
    max_gap : int
        The radius within which the original peaks may be shifted.
    smooth_window_size : int
        The window size of the moving average filter to apply on the
        signal. The smoothed signal

    Returns
    -------
    corrected_peak_inds : numpy array
        Array of the corrected peak indices

    """
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)

    # Peak ranges. What for?
    peak_ranges = [[peak_inds[i], peak_inds[i+1]] for i in range(n_peaks - 1)]
    sig_smoothed = smooth(sig=sig, window_size=smooth_window_size)




    return corrected_peak_inds

