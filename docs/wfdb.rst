wfdb
----

These core components are accessible by importing the `wfdb` package directly, as well as from their respective subpackages.

.. automodule:: wfdb
    :members: rdrecord, rdsamp, wrsamp, get_dbs, dl_database, dl_files, plot_record, plot_annotation, plot_all_records

.. autoclass:: wfdb.Record
    :members: wrsamp, adc, dac

.. autoclass:: wfdb.MultiRecord
    :members: multi_to_single

.. autoclass:: wfdb.Annotation
    :members: wrann