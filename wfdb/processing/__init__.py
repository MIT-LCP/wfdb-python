from .basic import (resample_ann, resample_sig, resample_singlechan,
                    resample_multichan, normalize_bound, get_filter_gain)
from .evaluate import Comparitor, compare_annotations
from .hr import compute_hr
from .peaks import find_peaks, find_local_peaks, correct_peaks
from .qrs import XQRS, xqrs_detect, gqrs_detect
