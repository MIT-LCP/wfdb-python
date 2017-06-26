import numpy


def compute_hr(length, peaks_indexes, fs):
    result = numpy.full(length, numpy.nan, dtype='float32')

    if len(peaks_indexes) < 2:
        return result

    current_hr = numpy.nan

    for i in range(0, len(peaks_indexes)-2):
        a = peaks_indexes[i]
        b = peaks_indexes[i+1]
        c = peaks_indexes[i+2]
        RR = (b-a) * (1.0 / fs) * 1000
        hr = 60000.0 / RR
        result[b+1:c+1] = hr
    result[peaks_indexes[-1]:] = result[peaks_indexes[-1]]

    return result
