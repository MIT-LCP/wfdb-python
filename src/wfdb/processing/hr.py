import posixpath

import numpy as np

from wfdb.io import download
from wfdb.io.annotation import is_qrs, rdann, wrann


def compute_hr(sig_len, qrs_inds, fs):
    """Compute instantaneous heart rate from peak indices.

    The rate from each RR interval is assigned to the samples after its second
    beat, up to and including the next beat. After the last beat, the rate
    from the last RR interval is used up to the end of the signal.

    Parameters
    ----------
    sig_len : int
        The length of the corresponding signal.
    qrs_inds : ndarray
        The QRS index locations, in strictly increasing order.
    fs : int, float
        The corresponding signal's sampling frequency.

    Returns
    -------
    heart_rate : ndarray
        An array of the instantaneous heart rate, with the length of the
        corresponding signal. Contains numpy.nan where heart rate could
        not be computed.

    Raises
    ------
    ValueError
        If `qrs_inds` is not strictly increasing.

    """
    # Compare neighbours directly, since np.diff wraps for unsigned integers.
    q = np.asarray(qrs_inds)
    not_increasing = np.flatnonzero(q[1:] <= q[:-1])
    if not_increasing.size:
        i = not_increasing[0]
        raise ValueError(
            "qrs_inds must be strictly increasing, but "
            f"qrs_inds[{i}] = {q[i]} is followed by qrs_inds[{i + 1}] = {q[i + 1]}"
        )

    heart_rate = np.full(sig_len, np.nan, dtype="float32")

    if len(qrs_inds) < 2:
        return heart_rate

    # The rate from each RR interval holds from just after its second beat up
    # to the next beat, or to the end of the signal after the last one.
    for i in range(0, len(qrs_inds) - 1):
        a = qrs_inds[i]
        b = qrs_inds[i + 1]
        end = qrs_inds[i + 2] + 1 if i + 2 < len(qrs_inds) else sig_len
        rr = (b - a) * (1.0 / fs) * 1000
        hr = 60000.0 / rr
        heart_rate[b + 1 : end] = hr

    return heart_rate


def calc_rr(
    qrs_locs,
    fs=None,
    min_rr=None,
    max_rr=None,
    qrs_units="samples",
    rr_units="samples",
):
    """Compute R-R intervals from QRS indices by extracting the time differences.

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
    if qrs_units == "samples" and rr_units == "seconds":
        rr = rr / fs
    elif qrs_units == "seconds" and rr_units == "samples":
        rr = rr * fs

    # Apply R-R interval filters
    if min_rr is not None:
        rr = rr[rr > min_rr]

    if max_rr is not None:
        rr = rr[rr < max_rr]

    return rr


def calc_mean_hr(rr, fs=None, min_rr=None, max_rr=None, rr_units="samples"):
    """Compute mean heart rate in beats per minute, from a set of R-R intervals.

    Returns 0 if rr is empty.

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
    if rr_units == "samples":
        mean_hr = mean_hr * fs

    return mean_hr


def ann2rr(
    record_name,
    extension,
    pn_dir=None,
    start_time=None,
    stop_time=None,
    format=None,
    as_array=True,
    all_annotations=False,
):
    """Obtain RR interval series from ECG annotation files.

    By default, only QRS (beat) annotations are used. The first interval
    is measured from the beginning of the record to the first selected
    annotation.

    Parameters
    ----------
    record_name : str
        The record name of the WFDB annotation file. ie. for file '100.atr',
        record_name='100'.
    extension : str
        The annotatator extension of the annotation file. ie. for  file
        '100.atr', extension='atr'.
    pn_dir : str, optional
        Option used to stream data from Physionet. The PhysioNet database
        directory from which to find the required annotation file. eg. For
        record '100' in 'http://physionet.org/content/mitdb': pn_dir='mitdb'.
    start_time : float, optional
        The time to start the intervals in seconds.
    stop_time : float, optional
        The time to stop the intervals in seconds.
    format : str, optional
        Print intervals in the specified format. By default, intervals are
        printed in units of sample intervals. Other formats include
        's' (seconds), 'm' (minutes), 'h' (hours). Set to 'None' for samples.
    as_array : bool, optional
        If True, return an 'ndarray', else print the output.
    all_annotations : bool, optional
        If True, include all annotation types, rather than only QRS
        annotations. This retains the behavior of earlier versions and
        corresponds to the WFDB ann2rr application's -A option.

    Returns
    -------
    out_interval : ndarray or None
        The intervals in the requested units, or None when printing.
        An empty array is returned if no annotations are selected.

    Examples
    --------
    >>> from wfdb import processing
    >>> processing.ann2rr('sample-data/100', 'atr', as_array=False)
    77
    293
    ...
    250
    257

    """
    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    ann = rdann(
        record_name, extension, pn_dir=pn_dir, return_label_elements=["label_store"]
    )

    samples = ann.sample
    if not all_annotations:
        samples = samples[np.isin(ann.label_store, np.flatnonzero(is_qrs))]

    rr_interval = calc_rr(samples, fs=ann.fs)
    if samples.size:
        rr_interval = np.insert(rr_interval, 0, samples[0])

    time_interval = rr_interval / ann.fs
    if start_time is not None:
        time_interval = time_interval[(time_interval > start_time).astype(bool)]
    if stop_time is not None:
        time_interval = time_interval[(time_interval < stop_time).astype(bool)]

    # Already given in seconds (format == 's')
    if format == "s":
        out_interval = time_interval
    elif format == "m":
        out_interval = time_interval / 60
    elif format == "h":
        out_interval = time_interval / (60 * 60)
    else:
        out_interval = np.around(time_interval * ann.fs).astype(np.int64)

    if as_array:
        return out_interval
    else:
        print(*out_interval, sep="\n")


def rr2ann(rr_array, record_name, extension, fs=250, as_time=False):
    """Create an annotation file from an array of intervals.

    The intervals should usually be a Numpy array in the format produced by `ann2rr`.
    (For exceptions, see the `as_time` parameter below.). An optional second column may
    be provided which gives the respective annotation mnemonic.

    Parameters
    ----------
    rr_array : ndarray
        A Numpy array consisting of the input RR intervals. If `as_time` is
        set to True, then the input should consist of times of occurences. If,
        the shape of the input array is '(n_annot,2)', then treat the second
        column as the annotation mnemonic ('N', 'V', etc.). If a second column
        is not specified, then the default annotation will the '"' which
        specifies a comment. Use ``ann2rr(..., all_annotations=True)`` to
        include these comment annotations when extracting intervals.
    record_name : str
        The record name of the WFDB annotation file. ie. for file '100.atr',
        record_name='100'.
    extension : str
        The annotatator extension of the annotation file. ie. for  file
        '100.atr', extension='atr'.
    fs : float, int, optional
        Assume the specified sampling frequency. This option has no effect
        unless the `as_time` parameter is set to convert to samples; in this
        case, a sampling frequency of 250 Hz is assumed if this option is
        omitted.
    as_time : bool
        Interpret the input as times of occurrence (if True), rather than as
        samples (if False). There is not currently a way to input RR intervals
        in time format between beats. For example, 0.2 seconds between beats
        1->2, 0.3 seconds between beats 2->3, etc.

    Returns
    -------
    N/A

    Examples
    --------
    Using time of occurence as input:
    >>> import numpy as np
    >>> rr_array = np.array([[0.2, 0.6, 1.3], ['V', 'N', 'V']]).T
    >>> wfdb.rr2ann(rr_array, 'test_ann', 'atr', fs=100, as_time=True)

    Using samples as input:
    >>> import numpy as np
    >>> rr_array = np.array([4, 17, 18, 16])
    >>> wfdb.rr2ann(rr_array, 'test_ann', 'atr')

    """
    try:
        ann_sample = rr_array[:, 0]
    except IndexError:
        ann_sample = rr_array

    # With a symbol column, the array holds strings, so convert the numbers.
    if as_time:
        ann_sample = np.round(fs * ann_sample.astype(np.float64)).astype(np.int64)
    else:
        # Sum as floats and round once, so fractional intervals don't drift.
        ann_sample = np.round(np.cumsum(ann_sample.astype(np.float64))).astype(np.int64)

    try:
        ann_symbol = rr_array[:, 1].tolist()
    except IndexError:
        ann_symbol = rr_array.shape[0] * ['"']

    wrann(record_name, extension, ann_sample, symbol=ann_symbol)
