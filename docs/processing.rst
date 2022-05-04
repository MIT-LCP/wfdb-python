processing
==========

The processing subpackage contains signal-processing tools.


Basic Utility
-------------

Basic signal processing functions

.. automodule:: wfdb.processing
    :members: resample_ann, resample_sig, resample_singlechan,
              resample_multichan, normalize_bound, get_filter_gain

Heart Rate
----------

.. automodule:: wfdb.processing
    :members: compute_hr, calc_rr, calc_mean_hr, ann2rr, rr2ann


Peaks
-----

.. automodule:: wfdb.processing
    :members: find_peaks, find_local_peaks, correct_peaks

Filters
-------

.. automodule:: wfdb.processing
    :members: sigavg

QRS Detectors
-------------

.. automodule:: wfdb.processing
    :members: XQRS, xqrs_detect, gqrs_detect


Annotation Evaluators
---------------------

.. automodule:: wfdb.processing
    :members: Comparitor, compare_annotations, benchmark_mitdb
