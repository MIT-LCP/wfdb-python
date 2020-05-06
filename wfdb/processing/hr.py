import numpy as np


def compute_hr(sig_len, qrs_inds, fs):
    """
    Compute instantaneous heart rate from peak indices.

    Parameters
    ----------
    sig_len : int
        The length of the corresponding signal.
    qrs_inds : ndarray
        The QRS index locations.
    fs : int, float
        The corresponding signal's sampling frequency.

    Returns
    -------
    heart_rate : ndarray
        An array of the instantaneous heart rate, with the length of the
        corresponding signal. Contains numpy.nan where heart rate could
        not be computed.

    """
    heart_rate = np.full(sig_len, np.nan, dtype='float32')

    if len(qrs_inds) < 2:
        return heart_rate

    for i in range(0, len(qrs_inds)-2):
        a = qrs_inds[i]
        b = qrs_inds[i+1]
        c = qrs_inds[i+2]
        rr = (b-a) * (1.0 / fs) * 1000
        hr = 60000.0 / rr
        heart_rate[b+1:c+1] = hr

    heart_rate[qrs_inds[-1]:] = heart_rate[qrs_inds[-1]]

    return heart_rate


def calc_rr(qrs_locs, fs=None, min_rr=None, max_rr=None, qrs_units='samples',
            rr_units='samples'):
    """
    Compute R-R intervals from QRS indices by extracting the time
    differences.

    Parameters
    ----------
    qrs_locs : ndarray
        1d array of QRS locations.
    fs : float, optional
        Sampling frequency of the original signal. Needed if
        `qrs_units` does not match `rr_units`.
    min_rr : float, optional
        The minimum allowed R-R interval. Values below this are excluded
        from the returned R-R intervals. Units are in `rr_units`.
    max_rr : float, optional
        The maximum allowed R-R interval. Values above this are excluded
        from the returned R-R intervals. Units are in `rr_units`.
    qrs_units : str, optional
        The time unit of `qrs_locs`. Must be one of: 'samples',
        'seconds'.
    rr_units : str, optional
        The desired time unit of the returned R-R intervals in. Must be
        one of: 'samples', 'seconds'.

    Returns
    -------
    rr : ndarray
        Array of R-R intervals.

    """
    rr = np.diff(qrs_locs)

    # Empty input qrs_locs
    if not len(rr):
        return rr

    # Convert to desired output rr units if needed
    if qrs_units == 'samples' and rr_units == 'seconds':
        rr = rr / fs
    elif qrs_units == 'seconds' and rr_units == 'samples':
        rr = rr * fs

    # Apply R-R interval filters
    if min_rr is not None:
        rr = rr[rr > min_rr]

    if max_rr is not None:
        rr = rr[rr < max_rr]

    return rr


def calc_mean_hr(rr, fs=None, min_rr=None, max_rr=None, rr_units='samples'):
    """
    Compute mean heart rate in beats per minute, from a set of R-R
    intervals. Returns 0 if rr is empty.

    Parameters
    ----------
    rr : ndarray
        Array of R-R intervals.
    fs : int, float
        The corresponding signal's sampling frequency. Required if
        'input_time_units' == 'samples'.
    min_rr : float, optional
        The minimum allowed R-R interval. Values below this are excluded
        when calculating the heart rate. Units are in `rr_units`.
    max_rr : float, optional
        The maximum allowed R-R interval. Values above this are excluded
        when calculating the heart rate. Units are in `rr_units`.
    rr_units : str, optional
        The time units of the input R-R intervals. Must be one of:
        'samples', 'seconds'.

    Returns
    -------
    mean_hr : float
        The mean heart rate in beats per minute.

    """
    if not len(rr):
        return 0

    if min_rr is not None:
        rr = rr[rr > min_rr]

    if max_rr is not None:
        rr = rr[rr < max_rr]

    mean_rr = np.mean(rr)

    mean_hr = 60 / mean_rr

    # Convert to bpm
    if rr_units == 'samples':
        mean_hr = mean_hr * fs

    return mean_hr
