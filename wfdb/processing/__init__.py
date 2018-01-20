"""
The processing subpackage contains signal-processing tools.
"""
from .basic import (resample_ann, resample_sig, resample_singlechan,
    resample_multichan, normalize)
from .evaluate import 
from .gqrs import gqrs_detect
from .hr import compute_hr
from .peaks import find_peaks, correct_peaks
from .qrs import Conf, XQRS, xqrs_detect

