import posixpath

import numpy as np
import pandas as pd

from wfdb.io import annotation
from wfdb.io import download
from wfdb.io.record import rdrecord


def sigavg(
    record_name,
    extension,
    pn_dir=None,
    return_df=False,
    start_range=-0.05,
    stop_range=0.05,
    ann_type="all",
    start_time=0,
    stop_time=-1,
    verbose=False,
):
    """
    A common problem in signal processing is to determine the shape of a
    recurring waveform in the presence of noise. If the waveform recurs
    periodically (for example, once per second) the signal can be divided into
    segments of an appropriate length (one second in this example), and the
    segments can be averaged to reduce the amplitude of any noise that is
    uncorrelated with the signal. Typically, noise is reduced by a factor of
    the square root of the number of segments included in the average. For
    physiologic signals, the waveforms of interest are usually not strictly
    periodic, however. This function averages such waveforms by defining
    segments (averaging windows) relative to the locations of waveform
    annotations. By default, all QRS (beat) annotations for the specified
    annotator are included.

    Parameters
    ----------
    record_name : str
        The name of the WFDB record to be read, without any file
        extensions. If the argument contains any path delimiter
        characters, the argument will be interpreted as PATH/BASE_RECORD.
        Both relative and absolute paths are accepted. If the `pn_dir`
        parameter is set, this parameter should contain just the base
        record name, and the files fill be searched for remotely.
        Otherwise, the data files will be searched for in the local path.
    pn_dir : str, optional
        Option used to stream data from Physionet. The Physionet
        database directory from which to find the required record files.
        eg. For record '100' in 'http://physionet.org/content/mitdb'
        pn_dir='mitdb'.
    return_df : bool, optional
        Whether to return a Pandas dataframe (True) or just print the output
        (False).
    start_range : float, int, optional
        Set the measurement window relative to QRS annotations. Negative
        values correspond to offsets that precede the annotations. The default
        is -0.05 seconds.
    stop_range : float, int, optional
        Set the measurement window relative to QRS annotations. Negative
        values correspond to offsets that precede the annotations. The default
        is 0.05 seconds.
    ann_type : list[str], str, optional
        Include annotations of the specified types only (i.e. 'N'). Multiple
        types are also accepted (i.e. ['V','N']). The default is 'all' which
        means to include all QRS annotations.
    start_time : float, int, optional
        Begin at the specified time in record. The default is 0 which denotes
        the start of the record.
    stop_time : float, int, optional
        Process until the specified time in record. The default is -1 which
        denotes the end of the record.
    verbose : bool, optional
        Whether to print the headers (True) or not (False).

    Returns
    -------
    N/A : Pandas dataframe
        If `return_df` is set to True, return a Pandas dataframe representing
        the output from the original WFDB package. This is the same content as
        if `return_df` were set to False, just in dataframe form.

    """
    if start_range >= stop_range:
        raise Exception("`start_range` must be less than `stop_range`")
    if start_time == stop_time:
        raise Exception("`start_time` must be different than `stop_time`")
    if (stop_time != -1) and (start_time >= stop_time):
        raise Exception("`start_time` must be less than `stop_time`")
    if start_time < 0:
        raise Exception("`start_time` must be at least 0")
    if (stop_time != -1) and (stop_time <= 0):
        raise Exception("`stop_time` must be at least greater than 0")

    if (pn_dir is not None) and ("." not in pn_dir):
        dir_list = pn_dir.split("/")
        pn_dir = posixpath.join(
            dir_list[0], download.get_version(dir_list[0]), *dir_list[1:]
        )

    rec = rdrecord(record_name, pn_dir=pn_dir, physical=False)
    ann = annotation.rdann(record_name, extension)

    if stop_time == -1:
        stop_time = max(ann.sample) / ann.fs
    samp_start = int(start_time * ann.fs)
    samp_stop = int(stop_time * ann.fs)
    filtered_samples = ann.sample[
        (ann.sample >= samp_start) & (ann.sample <= samp_stop)
    ]

    times = np.arange(
        int(start_range * rec.fs) / rec.fs,
        int(-(-stop_range // (1 / rec.fs))) / rec.fs,
        1 / rec.fs,
    )
    indices = np.rint(times * rec.fs).astype(np.int64)

    n_beats = 0
    initial_sig_avgs = np.zeros((times.shape[0], rec.n_sig))
    all_symbols = [a.symbol for a in annotation.ann_labels]

    for samp in filtered_samples:
        samp_i = np.where(ann.sample == samp)[0][0]
        current_ann = ann.symbol[samp_i]
        if (ann_type != "all") and (
            ((type(ann_type) is str) and (current_ann != ann_type))
            or ((type(ann_type) is list) and (current_ann not in ann_type))
        ):
            continue
        try:
            if not annotation.is_qrs[all_symbols.index(current_ann)]:
                continue
        except ValueError:
            continue

        for c, i in enumerate(indices):
            for j in range(rec.n_sig):
                try:
                    initial_sig_avgs[c][j] += rec.d_signal[samp + i][j]
                except IndexError:
                    initial_sig_avgs[c][j] += 0
        n_beats += 1

    if n_beats < 1:
        raise Exception("No beats found")

    if verbose and not return_df:
        print(f"# Average of {n_beats} beats:")
        s = "{:>14}" * rec.n_sig
        print(f"#        Time{s.format(*rec.sig_name)}")
        print(f"#         sec{s.format(*rec.units)}")

    final_sig_avgs = []
    for i, time in enumerate(times):
        sig_avgs = []
        for j in range(rec.n_sig):
            temp_sig_avg = initial_sig_avgs[i][j] / n_beats
            temp_sig_avg -= rec.baseline[j]
            temp_sig_avg /= rec.adc_gain[j]
            sig_avgs.append(round(temp_sig_avg, 5))
        final_sig_avgs.append(sig_avgs)

    df = pd.DataFrame(final_sig_avgs, columns=rec.sig_name)
    df.insert(0, "Time", np.around(times, decimals=5))
    if return_df:
        return df
    else:
        print(df.to_string(index=False, header=False, col_space=13))
