import numpy as np

from wfdb.processing.basic import smooth


def find_peaks(sig):
    r"""Find hard peaks and soft peaks in a signal, defined as follows:

    - Hard peak: a peak that is either /\ or \/.
    - Soft peak: a peak that is either /-*\ or \-*/.
      In this case we define the middle as the peak.

    Parameters
    ----------
    sig : np array
        The 1d signal array.

    Returns
    -------
    hard_peaks : ndarray
        Array containing the indices of the hard peaks.
    soft_peaks : ndarray
        Array containing the indices of the soft peaks.

    """
    if len(sig) == 0:
        return np.empty([0]), np.empty([0])

    tmp = sig[1:]
    tmp = np.append(tmp, [sig[-1]])
    tmp = sig - tmp
    tmp[np.where(tmp > 0)] = 1
    tmp[np.where(tmp == 0)] = 0
    tmp[np.where(tmp < 0)] = -1
    tmp2 = tmp[1:]
    tmp2 = np.append(tmp2, [0])
    tmp = tmp - tmp2

    hard_peaks = np.where(np.logical_or(tmp == -2, tmp == +2))[0] + 1
    soft_peaks = []

    for iv in np.where(np.logical_or(tmp == -1, tmp == +1))[0]:
        t = tmp[iv]
        i = iv + 1
        while True:
            if i == len(tmp) or tmp[i] == -t or tmp[i] == -2 or tmp[i] == 2:
                break
            if tmp[i] == t:
                soft_peaks.append(int(iv + (i - iv) / 2))
                break
            i += 1
    soft_peaks = np.array(soft_peaks, dtype="int") + 1

    return hard_peaks, soft_peaks


def find_local_peaks(sig, radius):
    """Find all local peaks in a signal.

    A sample is a local peak if it is the largest value within the <radius> samples on
    its left and right. In cases where it shares the max value with nearby samples, the
    first of these samples is classified as the local peak. The search then continues
    <radius> samples later, so a flat top that is wider than <radius> samples gives a
    peak every <radius> samples along it.

    Parameters
    ----------
    sig : ndarray
        1d numpy array of the signal.
    radius : int
        The radius in which to search for defining local maxima.

    Returns
    -------
    ndarray
        The locations of all of the local peaks of the input signal.

    """
    from scipy.ndimage import maximum_filter1d

    # TODO: Fix flat mountain scenarios.
    sig = np.asarray(sig)
    if np.min(sig) == np.max(sig):
        return np.empty(0)

    # The search continues at least one sample later
    step = max(radius, 1)
    # maximum_filter1d supports booleans, integers, float32 and float64.
    # float16 values are all exactly float32 values. Other types (such as
    # complex numbers) are checked one sample at a time.
    if sig.dtype.type is np.float16:
        sig = sig.astype(np.float32)
    elif sig.dtype.kind not in "biu" and sig.dtype.type not in (np.float32, np.float64):
        return _scan_local_peaks(sig, radius)

    n = len(sig)
    size = 2 * radius + 1
    peak_inds = []
    next_allowed = 0
    # Work on blocks of the signal, so that the arrays below stay small
    for start in range(0, n, _PEAK_BLOCK_SIZE):
        stop = min(start + _PEAK_BLOCK_SIZE, n)
        # The block with the `radius` samples on each side
        lo = max(0, start - radius)
        hi = min(n, stop + radius)
        seg = sig[lo:hi]
        seg_nan = np.isnan(seg) if sig.dtype.kind == "f" else None
        has_nan = seg_nan is not None and seg_nan.any()

        # The largest value within `radius` samples of each sample. A NaN
        # can spoil the result of maximum_filter1d well beyond its own
        # window, so filter a copy with -inf in its place.
        window_max = maximum_filter1d(
            np.where(seg_nan, -np.inf, seg) if has_nan else seg,
            size=size,
            mode="nearest",
        )[start - lo : stop - lo]
        is_candidate = sig[start:stop] == window_max
        if has_nan:
            # A window that contains NaN has no peak
            nan_nearby = maximum_filter1d(seg_nan.view(np.uint8), size, mode="nearest")
            is_candidate &= nan_nearby[start - lo : stop - lo] == 0

        # Take the candidates in order, skipping any within `radius` samples
        # after the previous peak. Within a run of consecutive candidates
        # (such as a flat top), that is every `radius` samples.
        edges = np.flatnonzero(np.diff(is_candidate.view(np.int8), prepend=0, append=0))
        for run_start, run_stop in zip(edges[0::2].tolist(), edges[1::2].tolist()):
            first = max(start + run_start, next_allowed)
            if first < start + run_stop:
                run_peaks = range(first, start + run_stop, step)
                peak_inds.extend(run_peaks)
                next_allowed = run_peaks[-1] + step

    return np.array(peak_inds)


# The number of samples that find_local_peaks works on at a time
_PEAK_BLOCK_SIZE = 65536


def _scan_local_peaks(sig, radius):
    """Find local peaks by checking one sample at a time.

    This is used by `find_local_peaks` for signal types that
    scipy.ndimage.maximum_filter1d does not support.

    """
    step = max(radius, 1)
    peak_inds = []
    i = 0
    while i < len(sig):
        if sig[i] == sig[max(0, i - radius) : i + radius + 1].max():
            peak_inds.append(i)
            i += step
        else:
            i += 1
    return np.array(peak_inds)


def correct_peaks(
    sig, peak_inds, search_radius, smooth_window_size, peak_dir="compare"
):
    """Adjust a set of detected peaks to coincide with local signal maxima.

    Parameters
    ----------
    sig : ndarray
        The 1d signal array.
    peak_inds : np array
        Array of the original peak indices.
    search_radius : int
        The radius within which the original peaks may be shifted.
    smooth_window_size : int
        The window size of the moving average filter applied on the
        signal. Peak distance is calculated on the difference between
        the original and smoothed signal.
    peak_dir : str, optional
        The expected peak direction: 'up' or 'down', 'both', or
        'compare'.

        - If 'up', the peaks will be shifted to local maxima.
        - If 'down', the peaks will be shifted to local minima.
        - If 'both', the peaks will be shifted to local maxima of the
          rectified signal.
        - If 'compare', the function will try both 'up' and 'down'
          options, and choose the direction that gives the largest mean
          distance from the smoothed signal.

    Returns
    -------
    shifted_peak_inds : ndarray
        Array of the corrected peak indices.

    """
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)

    # Subtract the smoothed signal from the original
    sig = sig - smooth(sig=sig, window_size=smooth_window_size)

    # Shift peaks to local maxima
    if peak_dir == "up":
        shifted_peak_inds = shift_peaks(
            sig=sig,
            peak_inds=peak_inds,
            search_radius=search_radius,
            peak_up=True,
        )
    elif peak_dir == "down":
        shifted_peak_inds = shift_peaks(
            sig=sig,
            peak_inds=peak_inds,
            search_radius=search_radius,
            peak_up=False,
        )
    elif peak_dir == "both":
        shifted_peak_inds = shift_peaks(
            sig=np.abs(sig),
            peak_inds=peak_inds,
            search_radius=search_radius,
            peak_up=True,
        )
    else:
        shifted_peak_inds_up = shift_peaks(
            sig=sig,
            peak_inds=peak_inds,
            search_radius=search_radius,
            peak_up=True,
        )
        shifted_peak_inds_down = shift_peaks(
            sig=sig,
            peak_inds=peak_inds,
            search_radius=search_radius,
            peak_up=False,
        )

        # Choose the direction with the biggest deviation
        up_dist = np.mean(np.abs(sig[shifted_peak_inds_up]))
        down_dist = np.mean(np.abs(sig[shifted_peak_inds_down]))

        if up_dist >= down_dist:
            shifted_peak_inds = shifted_peak_inds_up
        else:
            shifted_peak_inds = shifted_peak_inds_down

    return shifted_peak_inds


def shift_peaks(sig, peak_inds, search_radius, peak_up):
    """Helper function for correct_peaks.

    Return the shifted peaks to local maxima or minima within a radius.

    Parameters
    ----------
    sig : ndarray
        The 1d signal array.
    peak_inds : np array
        Array of the original peak indices.
    search_radius : int
        The radius within which the original peaks may be shifted.
    peak_up : bool
        Whether the expected peak direction is up.

    Returns
    -------
    shifted_peak_inds : ndarray
        Array of the corrected peak indices.

    """
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)
    # The indices to shift each peak ind by
    shift_inds = np.zeros(n_peaks, dtype="int")

    # Iterate through peaks
    for i in range(n_peaks):
        ind = peak_inds[i]
        local_sig = sig[
            max(0, ind - search_radius) : min(ind + search_radius + 1, sig_len)
        ]

        if peak_up:
            shift_inds[i] = np.argmax(local_sig)
        else:
            shift_inds[i] = np.argmin(local_sig)

    # May have to adjust early values
    for i in range(n_peaks):
        ind = peak_inds[i]
        if ind >= search_radius:
            break
        # The window starts at 0 rather than ind - search_radius
        shift_inds[i] += search_radius - ind

    shifted_peak_inds = peak_inds + shift_inds - search_radius

    return shifted_peak_inds
