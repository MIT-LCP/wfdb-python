import numpy


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
