import copy
import numpy
from .gqrs import time_to_sample_number, Conf, Peak, Annotation
from .basic import smooth

def find_peaks(x):
    # Definitions:
    # * Hard peak: a peak that is either /\ or \/
    # * Soft peak: a peak that is either /-*\ or \-*/ (In that cas we define the middle of it as the peak)

    # Returns two numpy arrays:
    # * hard_peaks contains the indexes of the Hard peaks
    # * soft_peaks contains the indexes of the Soft peaks

    if len(x) == 0:
        return numpy.empty([0]), numpy.empty([0])

    tmp = x[1:]
    tmp = numpy.append(tmp, [x[-1]])
    tmp = x-tmp
    tmp[numpy.where(tmp>0)] = +1
    tmp[numpy.where(tmp==0)] = 0
    tmp[numpy.where(tmp<0)] = -1
    tmp2 = tmp[1:]
    tmp2 = numpy.append(tmp2, [0])
    tmp = tmp-tmp2
    hard_peaks = numpy.where(numpy.logical_or(tmp==-2,tmp==+2))[0]+1
    soft_peaks = []
    for iv in numpy.where(numpy.logical_or(tmp==-1,tmp==+1))[0]:
        t = tmp[iv]
        i = iv+1
        while True:
            if i==len(tmp) or tmp[i] == -t or tmp[i] == -2 or tmp[i] == 2:
                break
            if tmp[i] == t:
                soft_peaks.append(int(iv+(i-iv)/2))
                break
            i += 1
    soft_peaks = numpy.asarray(soft_peaks)+1
    return hard_peaks, soft_peaks

def find_peaks(x):
    # Definitions:
    # * Hard peak: a peak that is either /\ or \/
    # * Soft peak: a peak that is either /-*\ or \-*/ (In that cas we define the middle of it as the peak)

    # Returns two numpy arrays:
    # * hard_peaks contains the indexes of the Hard peaks
    # * soft_peaks contains the indexes of the Soft peaks

    if len(x) == 0:
        return numpy.empty([0]), numpy.empty([0])

    tmp = x[1:]
    tmp = numpy.append(tmp, [x[-1]])
    tmp = x-tmp
    tmp[numpy.where(tmp>0)] = +1
    tmp[numpy.where(tmp==0)] = 0
    tmp[numpy.where(tmp<0)] = -1
    tmp2 = tmp[1:]
    tmp2 = numpy.append(tmp2, [0])
    tmp = tmp-tmp2
    hard_peaks = numpy.where(numpy.logical_or(tmp==-2,tmp==+2))[0]+1
    soft_peaks = []
    for iv in numpy.where(numpy.logical_or(tmp==-1,tmp==+1))[0]:
        t = tmp[iv]
        i = iv+1
        while True:
            if i==len(tmp) or tmp[i] == -t or tmp[i] == -2 or tmp[i] == 2:
                break
            if tmp[i] == t:
                soft_peaks.append(int(iv+(i-iv)/2))
                break
            i += 1
    soft_peaks = numpy.asarray(soft_peaks)+1
    return hard_peaks, soft_peaks


def correct_peaks(x, peaks_indexes, min_gap, max_gap, smooth_window):
    N = x.shape[0]

    rpeaks = numpy.zeros(N)
    rpeaks[peaks_indexes] = 1.0

    rpeaks = rpeaks.astype('int32')

    # 1- Extract ranges where we have one or many ones side by side
    rpeaks_ranges = []
    tmp_idx = 0
    for i in range(1, len(rpeaks)):
        if rpeaks[i-1] == 1:
            if rpeaks[i] == 0:
                rpeaks_ranges.append((tmp_idx, i-1))
        else:
            if rpeaks[i] == 1:
                tmp_idx = i

    smoothed = smooth(x, smooth_window)

    # Compute signal's peaks
    hard_peaks, soft_peaks = find_peaks(x=x)
    all_peak_idxs = numpy.concatenate((hard_peaks, soft_peaks)).astype('int64')

    # Replace each range of ones by the index of the best value in it
    tmp = set()
    for rp_range in rpeaks_ranges:
        r = numpy.arange(rp_range[0], rp_range[1]+1, dtype='int64')
        vals = x[r]
        smoothed_vals = smoothed[r]
        p = r[numpy.argmax(numpy.absolute(numpy.asarray(vals)-smoothed_vals))]
        tmp.add(p)

    # Replace all peaks by the peak within x-max_gap < x < x+max_gap which have the bigget distance from smooth curve
    dist = numpy.absolute(x-smoothed) # Peak distance from the smoothed mean
    rpeaks_indexes = set()
    for p in tmp:
        a = max(0, p-max_gap)
        b = min(N, p+max_gap)
        r = numpy.arange(a, b, dtype='int64')
        idx_best = r[numpy.argmax(dist[r])]
        rpeaks_indexes.add(idx_best)

    rpeaks_indexes = list(rpeaks_indexes)

    # Prevent multiple peaks to appear in the max bpm range (max_gap)
    # If we found more than one peak in this interval, then we choose the peak with the maximum amplitude compared to the mean of the signal
    tmp = numpy.asarray(rpeaks_indexes)
    to_remove = {}
    for idx in rpeaks_indexes:
        if idx in to_remove:
            continue
        r = tmp[numpy.where(numpy.absolute(tmp-idx)<=max_gap)[0]]
        if len(r) == 1:
            continue
        rr = r.astype('int64')
        vals = x[rr]
        smoo = smoothed[rr]
        the_one = r[numpy.argmax(numpy.absolute(vals-smoo))]
        for i in r:
            if i != the_one:
                to_remove[i] = True
    for v, _ in to_remove.items():
        rpeaks_indexes.remove(v)

    return sorted(rpeaks_indexes)
