io
===

The input/output subpackage contains classes used to represent WFDB
objects, and functions to read, write, and download WFDB files.


WFDB Records
---------------

.. automodule:: wfdb.io
    :members: rdrecord, rdheader, rdsamp, wrsamp

.. autoclass:: wfdb.io.Record
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              wrsamp, adc, dac

.. autoclass:: wfdb.io.MultiRecord
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              multi_to_single


WFDB Annotations
---------------

.. automodule:: wfdb.io
    :members: rdann, wrann, show_ann_labels, show_ann_classes

.. autoclass:: wfdb.io.Annotation
    :members: wrann


Downloading
-----------

.. automodule:: wfdb.io
    :members: dl_files, dl_database, get_dbs, get_record_list, set_db_index_url
