io
===

The input/output subpackage contains classes used to represent WFDB
objects, and functions to read, write, and download WFDB files.


WFDB Records
---------------

.. automodule:: wfdb.io
    :members: rdrecord, rdheader, read_header, rdsamp, iter_dataframes,
              wrsamp, sampfreq, signame, wfdbdesc, wfdbtime

``read_header`` returns a structured, immutable description of one WFDB header.
Its ``signals`` or ``segments`` entries describe the source file, while
``rdheader`` continues to return a mutable ``Record`` or ``MultiRecord``.
``n_frames`` counts base sample frames, and remains ``None`` when the header
omits the length. ``n_signals`` is the declared number of channels.
For a multi-segment record, ``read_header`` reads the main header only; each
``SegmentReference`` contains a name and length. To inspect a segment's
signal names, units, or calibration, pass its path to ``read_header`` in a
separate call. For a local ``.wfdb`` archive, pass the archive path with
``segment=reference.name`` instead. Both approaches work for a
variable-layout record's zero-length layout segment. A ``~`` entry represents
a gap and has no header to read.
For recursive loading of segment headers, use ``rdheader(...,
rd_segments=True)``. ``read_header`` describes one physical header per call;
it does not replace that recursive API.

.. code-block:: python

    import wfdb

    multi = wfdb.read_header("sample-data/multi-segment/041s/041s")
    assert isinstance(multi, wfdb.MultiSegmentHeader)
    first = wfdb.read_header(
        "sample-data/multi-segment/041s/" + multi.segments[0].name
    )
    assert isinstance(first, wfdb.SingleSegmentHeader)
    print(first.signals[0].name)

For a multi-segment archive, read the main header and a referenced segment
from the same archive:

.. code-block:: python

    multi = wfdb.read_header("041s.wfdb")
    assert isinstance(multi, wfdb.MultiSegmentHeader)
    first = wfdb.read_header("041s.wfdb", segment=multi.segments[0].name)
    assert isinstance(first, wfdb.SingleSegmentHeader)
    print(first.signals[0].name)

In a signal header, an omitted ``baseline`` is resolved to ``adc_zero`` if
present, or zero otherwise. Omitted optional fields such as ``adc_zero``,
``adc_res``, and ``init_value`` remain ``None`` so callers can distinguish
them from values explicitly stated in the file.

.. autoclass:: wfdb.io.SingleSegmentHeader
    :no-index:

.. autoclass:: wfdb.io.MultiSegmentHeader
    :no-index:

.. autoclass:: wfdb.io.SignalHeader
    :no-index:

.. autoclass:: wfdb.io.SegmentReference
    :no-index:

Use ``iter_dataframes`` to process long recordings without loading the whole
signal into memory. Each chunk is a pandas DataFrame with the same columns
and absolute time index as a full-record DataFrame. Record and segment
headers are reused within one iterator, including for remote records:

.. code-block:: python

    import wfdb

    for chunk in wfdb.iter_dataframes("100", chunk_size=3600, pn_dir="mitdb"):
        print(chunk.shape)

.. autoclass:: wfdb.io.Record
    :no-index:
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              wrsamp, adc, dac, to_dataframe

.. autoclass:: wfdb.io.MultiRecord
    :no-index:
    :members: get_frame_number, get_elapsed_time, get_absolute_time,
              multi_to_single, contained_ranges, contained_combined_ranges

.. autoclass:: wfdb.io.SourceSegment
    :no-index:


Checking WFDB records
---------------------

Run ``wfdb check PATH`` before sharing a record or a dataset directory. The
command checks each local header and ``.wfdb`` archive, follows multi-segment
references, compares segment lengths and fixed-layout signals, verifies signal
file references and the minimum size of uncompressed signal files, and reads
the first frame of each segment. It exits with status 1 if it finds a problem.
It does not scan all samples or annotation files.

The same checks are available from Python:

.. code-block:: python

    import wfdb

    for issue in wfdb.check_dataset("my-dataset"):
        print(issue.path, issue.code, issue.message)

.. automodule:: wfdb.io.check
    :members: check_record, check_dataset, CheckIssue

Baseline and ADC zero
---------------------

The `WFDB header specification
<https://physionet.org/physiotools/wag/header-5.htm>`_ defines ``baseline``
as the digital sample value that corresponds to zero physical units.
``adc_zero`` describes the midpoint of the digitizer's input range. These
values can differ. When reading physical samples, WFDB uses
``(digital_sample - baseline) / adc_gain``, even if ``adc_zero`` has another
value.

The `MIT-BIH Noise Stress Test Database
<https://physionet.org/content/nstdb/1.0.0/>`_ illustrates why this matters.
Record ``118e24`` has gain 200 adu/mV, baseline 1024, and ADC zero 0. Its
noise-free opening contains digital samples 1024 lower than those of the
source MIT-BIH Arrhythmia Database record ``118``, whose baseline and ADC zero
are both 1024. Applying each record's header calibration makes the physical
samples in ``118e24`` 5.12 mV lower than those in ``118`` over that interval.
WFDB-Python follows the published headers here; replacing ``baseline`` with
``adc_zero`` would change the meaning of the stored calibration. See
`issue #752 <https://github.com/MIT-LCP/wfdb-python/issues/752>`_ for a
reproducible comparison with the WFDB C tools.

Physical units and format ranges
--------------------------------

When writing a record with ``wrsamp``, digital values must fit the chosen
WFDB format. ``Record.adc`` converts physical values with
``round(physical * adc_gain + baseline)`` and does not clip. ``wrsamp`` then
raises ``IndexError`` if those digital values fall outside the format's
storage range (for format 212, -2048 to 2047). It does not silently clip or
wrap, and it does not warn before raising.

Format 212 stores 12-bit two's-complement values, but digital -2048 is its
missing-sample marker: ``wrsamp`` accepts it and ``rdrecord`` returns NaN.
The usable signal range is therefore -2047 to 2047. At gain 200 adu/mV and
baseline 1024 that is -15.355 mV to +5.115 mV.

Some PhysioNet databases, including the MIT-BIH Arrhythmia Database, were
digitized with an 11-bit ADC at gain 200 and baseline 1024, so their stored
samples lie in 0 to 2047 (-5.12 mV to +5.115 mV). That ADC range is
narrower than what format 212 can store at the same gain and baseline, so
values below -5.12 mV still write and read back correctly.

As an example of data that does not fit, in the St Petersburg INCART
12-lead Arrhythmia Database (version 1.0.0, 75 records, lead II only, no
missing samples), converting to gain 200 and baseline 1024 puts 2,153,612
of 34,695,000 samples (6.21%) outside format 212's storage range, so
``wrsamp`` would raise. A further 27 samples land exactly on -2048 and would
read back as NaN. Choose a format and gain that cover the physical range,
or requantize explicitly, and count how many samples would not fit before
writing.

The INCART documentation says gains vary from 250 to 1100 adu/mV. The
``.hea`` files actually use 240 to 1063 adu/mV (seven records have gain 240
or 248). Always read the gain from the header.

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

WFDB-Python supports reading and writing records and annotations in cloud
storage via the ``fsspec`` library. Pass a cloud URI as ``record_name`` when
reading, or as ``write_dir`` when writing.

**Supported protocols:** ``s3://`` (Amazon S3), ``gs://`` (Google Cloud
Storage), ``az://`` (Azure Blob Storage), ``azureml://`` (Azure ML), and
``memory://`` (in-memory filesystems, useful for testing).

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

    # Write a record and its annotations to S3
    record.wrsamp(write_dir="s3://my-bucket/wfdb-data")
    ann.wrann(write_dir="s3://my-bucket/wfdb-data")

    # For PhysioNet databases, use pn_dir with the database name:
    record = wfdb.rdrecord("100", pn_dir="mitdb")
    ann = wfdb.rdann("100", "atr", pn_dir="mitdb")

**Authentication:** Configure credentials through the standard
provider-specific mechanism (e.g., ``~/.aws/credentials`` for S3,
``GOOGLE_APPLICATION_CREDENTIALS`` for GCS). The ``fsspec`` library
handles credential discovery automatically.

Cloud writing also works with ``wfdb.wrsamp``, ``MultiRecord.wrsamp``, and
``wfdb.wrann``. WFDB-Python first writes the files to a temporary local
directory, then uploads them. A write needs enough local space for the
complete record. The main header is uploaded last, but replacing a record
with multiple files is not atomic: readers may see a mix of old and new
files while an upload is in progress. ``physionet://`` destinations and
``.wfdb`` archives in cloud storage are not supported.

.. note::

    When reading, pass cloud URIs as ``record_name``, not ``pn_dir``.
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
