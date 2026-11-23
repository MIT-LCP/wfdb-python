io
===

The input/output subpackage contains classes used to represent WFDB
objects, and functions to read, write, and download WFDB files.


WFDB Records
---------------

.. automodule:: wfdb.io
    :members: rdrecord, rdheader, rdsamp, wrsamp, sampfreq, signame, wfdbdesc,
              wfdbtime

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
    :members: rdann, wrann, mrgann, show_ann_labels, show_ann_classes

.. autoclass:: wfdb.io.Annotation
    :no-index:
    :members: wrann


WFDB Archives
-------------

A ``.wfdb`` archive is a ZIP file containing the files of a single
record: its header file, the signal files listed in the header, and
optionally annotation files. The files are at the top level of the
archive, and the header file is named after the archive (for example,
``100.wfdb`` contains ``100.hea``).
Files are stored without compression, so that part of a signal can be
read without decompressing the whole file. Archives made by other tools
may compress their files, and can still be read.

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

An archive holds a single record. For a multi-segment record, the
archive contains the main header together with the header and signal
files of each segment, and ``MultiRecord.wrsamp`` writes them all when
given ``wfdb_archive``. Annotation files can be added alongside the
record. Archives in cloud storage or on PhysioNet are not supported
yet.

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


Reading From PhysioNet
----------------------

Records and annotations on PhysioNet can also be read with a
``physionet://`` URI, using the ``physionet`` package (a dependency of
``wfdb``). The URI is the database, its version and the record name, as in
``physionet://mitdb/1.0.0/100``. Use ``latest`` as the version to read the
newest one.

.. code-block:: python

    import wfdb

    record = wfdb.rdrecord("physionet://mitdb/1.0.0/100", sampto=3600)
    ann = wfdb.rdann("physionet://mitdb/1.0.0/100", "atr")

Files are read with HTTP range requests: open databases from the AWS open
data mirror, and restricted ones from PhysioNet. For signal files, only the
parts that hold the samples that are needed are read, so reading a few
seconds of a long recording does not download the whole file. Header and
annotation files are read in full (``rdann`` reads the whole annotation
file, even with ``sampto``). Files that the ``physionet`` package has
already downloaded into its cache are read from there.

Restricted databases can be read once you have access to them on PhysioNet
and have signed in, by running ``physionet login`` once in a terminal (or
setting ``PHYSIONET_TOKEN``). See the `physionet package
<https://github.com/MIT-LCP/physionet>`_ for details.

The ``pn_dir`` argument reads files the same way, so
``wfdb.rdrecord("100", pn_dir="mitdb")`` and
``wfdb.rdrecord("physionet://mitdb/1.0.0/100")`` read the same files. For a
record in a folder of the database, the folders can be given in either
argument, so ``wfdb.rdrecord("files/p1000/p10000032/s40689238/40689238",
pn_dir="mimic-iv-ecg/1.0")`` and ``wfdb.rdrecord("40689238",
pn_dir="mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238")`` read the same
record. When ``pn_dir`` has no version, the latest version is looked up once
per database, and used until the program ends. If the database index URL has been changed
with ``wfdb.io.download.set_db_index_url``, files are read from that server
instead, as before.


Reading From Memory
-------------------

Records and annotations can also be read from the in-memory filesystem of
``fsspec``, for example when the files come from an upload form instead of
from disk. Store all files of the record in the same directory, and pass a
``memory://`` URI as the ``record_name`` argument:

.. code-block:: python

    import fsspec
    import wfdb

    fs = fsspec.filesystem("memory")
    fs.pipe("/upload/100.hea", hea_bytes)
    fs.pipe("/upload/100.dat", dat_bytes)
    fs.pipe("/upload/100.atr", atr_bytes)

    record = wfdb.rdrecord("memory://upload/100")
    ann = wfdb.rdann("memory://upload/100", "atr")

The files stay in memory until they are removed, for example with
``fs.rm("/upload", recursive=True)``.

.. warning::

    The in-memory filesystem is shared by the whole process, and everything
    that opens a file shares a single file position. Use a unique directory
    for each upload, and do not read the same record from several threads at
    once, as this can return wrong data without raising an error.


Downloading
-----------

.. automodule:: wfdb.io
    :no-index:
    :members: dl_files, dl_database, get_dbs, get_record_list, set_db_index_url
