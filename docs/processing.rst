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
    :no-index:
    :members: compute_hr, calc_rr, calc_mean_hr, ann2rr, rr2ann


Peaks
-----

.. automodule:: wfdb.processing
    :no-index:
    :members: find_peaks, find_local_peaks, correct_peaks

Filters
-------

.. automodule:: wfdb.processing
    :no-index:
    :members: sigavg

QRS Detectors
-------------

.. automodule:: wfdb.processing
    :no-index:
    :members: XQRS, xqrs_detect, gqrs_detect


Annotation Evaluators
---------------------

.. automodule:: wfdb.processing
    :no-index:
    :members: Comparitor, compare_annotations, benchmark_mitdb


Beat-by-beat Comparison
-----------------------

.. warning::

    These functions are a development feature. They compare annotation
    files as the ``bxb`` and ``sumstats`` programs of the WFDB Software
    Package do, and were written from the ANSI/AAMI EC57 standard and the
    documentation of those programs, without reference to their source code.
    Their interface may change, and results should be checked against
    ``bxb`` before they are used for reporting.

.. automodule:: wfdb.processing
    :no-index:
    :members: bxb, bxb_record, sumstats

.. autoclass:: wfdb.processing.BxbResult
    :members: count, matrix, statistics, rms_rr_error, shutdown_seconds,
              report, line_columns, lines
