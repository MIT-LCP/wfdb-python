wfdb
====

These core components are accessible by importing the `wfdb` package
directly, as well as from their respective subpackages.


WFDB Records
---------------

.. automodule:: wfdb
    :members: rdrecord, rdheader, rdsamp, wrsamp

.. autoclass:: wfdb.Record
    :members: wrsamp, adc, dac

.. autoclass:: wfdb.MultiRecord
    :members: multi_to_single


WFDB Anotations
---------------

.. automodule:: wfdb
    :members: rdann, wrann, show_ann_labels, show_ann_classes

.. autoclass:: wfdb.Annotation
    :members: wrann


Downloading
-----------

.. automodule:: wfdb
    :members: dl_files, dl_database, get_dbs, get_record_list, set_db_index_url


Plotting
--------

.. automodule:: wfdb
    :members: plot_items, plot_wfdb, plot_all_records
