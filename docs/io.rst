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


Cloud Storage Access
--------------------

WFDB-Python supports reading records and annotations directly from cloud
storage via the ``fsspec`` library. Pass a cloud URI as the
``record_name`` argument instead of a local path.

**Supported protocols:** ``s3://`` (Amazon S3), ``gs://`` (Google Cloud
Storage), ``az://`` (Azure Blob Storage), and ``azureml://`` (Azure ML).

**Prerequisites:** Install the ``fsspec`` backend for your cloud provider:

.. code-block:: bash

    pip install s3fs      # Amazon S3
    pip install gcsfs     # Google Cloud Storage
    pip install adlfs     # Azure Blob Storage

``fsspec`` itself is already included as a core dependency of ``wfdb``.

**Usage examples:**

.. code-block:: python

    import wfdb

    # Read a record from Amazon S3
    record = wfdb.rdrecord("s3://my-bucket/wfdb-data/100")

    # Read from Google Cloud Storage
    record = wfdb.rdrecord("gs://my-bucket/wfdb-data/100")

    # Read annotations from S3
    ann = wfdb.rdann("s3://my-bucket/wfdb-data/100", "atr")

    # For PhysioNet databases, use pn_dir with the database name:
    record = wfdb.rdrecord("100", pn_dir="mitdb")
    ann = wfdb.rdann("100", "atr", pn_dir="mitdb")

**Authentication:** Configure credentials through the standard
provider-specific mechanism (e.g., ``~/.aws/credentials`` for S3,
``GOOGLE_APPLICATION_CREDENTIALS`` for GCS). The ``fsspec`` library
handles credential discovery automatically.

.. note::

    Cloud URIs must be passed as ``record_name``, not ``pn_dir``.
    The ``pn_dir`` parameter is reserved for PhysioNet database names
    (e.g., ``"mitdb"`` or ``"mimic4wdb/0.1.0"``), which are resolved
    against the configured PhysioNet index URL.


Downloading
-----------

.. automodule:: wfdb.io
    :members: dl_files, dl_database, get_dbs, get_record_list, set_db_index_url
