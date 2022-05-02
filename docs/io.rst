io
===

The input/output subpackage contains classes used to represent WFDB
objects, and functions to read, write, and download WFDB files.


WFDB Records
---------------

.. automodule:: wfdb.io
    :members: rdrecord, rdheader, rdsamp, wrsamp, read_edf, wfdb_to_edf, read_wav, wfdb_to_wav, csv_to_wfdb, wfdb_to_mat

.. autoclass:: wfdb.io.Record
    :members: wrsamp, adc, dac

.. autoclass:: wfdb.io.MultiRecord
    :members: multi_to_single


WFDB Anotations
---------------

.. automodule:: wfdb.io
    :members: rdann, wrann, show_ann_labels, show_ann_classes

.. autoclass:: wfdb.io.Annotation
    :members: wrann


Downloading
-----------

.. automodule:: wfdb.io
    :members: get_dbs, get_record_list, dl_database, dl_files, set_db_index_url
