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
              wrsamp, adc, dac, to_dataframe

.. autoclass:: wfdb.io.MultiRecord
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              multi_to_single, contained_ranges, contained_combined_ranges


WFDB Annotations
---------------

.. automodule:: wfdb.io
    :members: rdann, wrann, show_ann_labels, show_ann_classes

.. autoclass:: wfdb.io.Annotation
    :members: wrann


Cloud and Remote Access
-----------------------

WFDB-Python supports reading records and annotations directly from cloud
storage and remote URLs via the ``fsspec`` library. Instead of downloading
entire databases, you can access individual files on demand.

**Supported protocols** include ``s3://`` (Amazon S3), ``gs://`` (Google
Cloud Storage), ``az://`` (Azure Blob Storage), ``https://``, and any
other protocol supported by ``fsspec``.

**Installation:**

.. code-block:: bash

    pip install wfdb[cloud]
    # or: pip install fsspec s3fs  (for S3 specifically)

**Usage examples:**

.. code-block:: python

    import wfdb

    # Read a record from an HTTPS URL
    record = wfdb.rdrecord("100", pn_dir="https://physionet.org/files/mitdb/1.0.0/")

    # Read from Amazon S3
    record = wfdb.rdrecord("s3://my-bucket/wfdb-data/100")

    # Read annotations from a remote path
    ann = wfdb.rdann("100", "atr", pn_dir="https://physionet.org/files/mitdb/1.0.0/")

**Authentication:** For cloud providers requiring credentials (S3, GCS,
Azure), configure authentication through the standard provider-specific
mechanism (e.g., ``~/.aws/credentials`` for S3, ``GOOGLE_APPLICATION_CREDENTIALS``
for GCS). The ``fsspec`` library handles credential discovery automatically.

For PhysioNet databases that require credentialed access, you can pass
credentials via ``fsspec`` storage options or configure them in your
environment before calling ``wfdb`` functions.


Downloading
-----------

.. automodule:: wfdb.io
    :members: dl_files, dl_database, get_dbs, get_record_list, set_db_index_url
