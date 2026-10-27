io
===

The input/output subpackage contains classes used to represent WFDB
objects, and functions to read, write, and download WFDB files.


WFDB Records
---------------

.. automodule:: wfdb.io
    :members: rdrecord, rdheader, rdsamp, wrsamp

.. autoclass:: wfdb.io.Record
    :no-index:
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              wrsamp, adc, dac, to_dataframe

.. autoclass:: wfdb.io.MultiRecord
    :no-index:
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              multi_to_single, contained_ranges, contained_combined_ranges


WFDB Annotations
----------------

.. automodule:: wfdb.io
    :no-index:
    :members: rdann, wrann, show_ann_labels, show_ann_classes

.. autoclass:: wfdb.io.Annotation
    :no-index:
    :members: wrann


WFDB Archives
-------------

A ``.wfdb`` archive is a ZIP file containing the files of a single
record: its header file, the signal files listed in the header, and
optionally annotation files.
Header files are compressed, while signal files are stored without
compression so that part of a signal can be read without decompressing
the whole file.

To read a record from an archive, pass the path of the archive,
including the ``.wfdb`` extension, to ``rdrecord``, ``rdsamp`` or
``rdheader``. Annotations are read the same way with ``rdann``. To
write a record into an archive, pass the path of the archive as the
``wfdb_archive`` argument of ``wrsamp``.

.. code-block:: python

    import wfdb

    # Write a record into an archive
    wfdb.wrsamp("100", fs=360, units=["mV"], sig_name=["MLII"],
                p_signal=signal, fmt=["212"], wfdb_archive="100.wfdb")

    # Read the record back, or part of it
    record = wfdb.rdrecord("100.wfdb")
    record = wfdb.rdrecord("100.wfdb", sampfrom=1000, sampto=2000)

    # Put the files of an existing record, including its annotations,
    # into an archive, and read the annotations
    from wfdb.io.archive import create_archive
    create_archive(["100.hea", "100.dat", "100.atr"], "100.wfdb")
    annotation = wfdb.rdann("100.wfdb", "atr")

An archive holds a single record, so only one header file can be
written to it. Annotation files can be added alongside the record.
Multi-segment records, and archives in cloud storage or on PhysioNet,
are not supported yet.

.. autofunction:: wfdb.io.archive.create_archive

.. autoclass:: wfdb.io.archive.WFDBArchive
    :members: namelist, exists, file_size, open, header_file_name,
              read_header, write, add_file, close, abort


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
    :no-index:
    :members: dl_files, dl_database, get_dbs, get_record_list, set_db_index_url
