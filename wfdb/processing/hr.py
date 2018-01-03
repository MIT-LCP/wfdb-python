import numpy as np


def compute_hr(sig_len, peak_indices, fs):
    """
    Compute instantaneous heart rate from peak indices.

    Parameters
    ----------
    sig_len : int
        The length of the corresponding signal
    peak_indices : numpy array
        The peak index locations
    fs : int, or float
        The corresponding signal's sampling frequency.

    Returns
    -------
    heart_rate : numpy array
        An array of the instantaneous heart rate, with the length of the
        corresponding signal. Contains numpy.nan where heart rate could not be
        computed.

    """
    heart_rate = np.full(sig_len, np.nan, dtype='float32')

    if len(peak_indices) < 2:
        return heart_rate

    current_hr = np.nan

    for i in range(0, len(peak_indices)-2):
        a = peak_indices[i]
        b = peak_indices[i+1]
        c = peak_indices[i+2]
        RR = (b-a) * (1.0 / fs) * 1000
        hr = 60000.0 / RR
        heart_rate[b+1:c+1] = hr

    heart_rate[peak_indices[-1]:] = heart_rate[peak_indices[-1]]

    return heart_rate
