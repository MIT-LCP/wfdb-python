Recent Changes
==============

This page lists recent changes in the `wfdb` package (since version 4.0.0) that may be relevant if you are upgrading from an older version of the package.  For the complete history of changes in the package, please refer to the `development repository`_ on GitHub.

.. _development repository: https://github.com/MIT-LCP/wfdb-python

.. towncrier release notes start

Version 5.6.0 (Nov 2026)
------------------------

This release prepares the record reader for a clearer API while preserving
existing public calls. It also includes two fixes and a calibration note.

**Clarify baseline and ADC zero in record headers**
  Document how physical samples are calibrated when ``baseline`` and
  ``adc_zero`` differ, using the MIT-BIH Noise Stress Test Database as an
  example.

**Return false for records with different signal values**
  Comparing ``Record`` objects with unequal uniform or expanded signal arrays
  now returns ``False`` instead of raising an exception.

**Prepare record I/O for a clearer API**
  Record reads now keep source headers, selected ranges, and loaded signal
  modes in separate internal structures. The existing public readers and
  mutable record attributes retain their behavior.

**Round csv2ann time onsets to the nearest sample**
  ``csv2ann`` with ``time_onset=True`` now rounds onset times to the nearest
  sample instead of truncating them, so 0.29 seconds at 100 Hz gives sample 29
  instead of 28.

Version 5.5.1 (Nov 2026)
------------------------

**Explain malformed header references in wfdb check**
  When a header still contains directory prefixes after a dataset is
  reorganized, ``wfdb check`` now identifies the affected lines and shows the
  record and signal filenames expected beside the header.

Version 5.5.0 (Nov 2026)
------------------------

This release adds a way to omit excluded folders and records from the local
WFDB checker, useful when validating a mixed dataset before publication.

**Filter records when checking a dataset**
  ``check_dataset`` has an ``ignore`` argument, a function that returns True
  for paths to skip. It chooses which headers and archives to check, and the
  contents of skipped folders are not read. Signal files and segments that a
  checked record refers to are still checked.

Version 5.4.0 (Nov 2026)
------------------------

This release adds a command and Python API for checking local WFDB records
before sharing them. It also fixes precision errors in annotation comparisons
and peak detection, and corrects signal averaging at record boundaries. The
documentation now explains format 212's usable range and missing-sample value.

**Check local WFDB records before sharing**
  Use ``wfdb check PATH`` or the ``check_record`` and ``check_dataset`` Python
  functions to find malformed headers, mismatched record names, missing signal
  or segment files, inconsistent segments, truncated uncompressed signal files,
  and signal data that cannot be read. The same checks apply to ``.wfdb``
  archives.

**Avoid integer overflow when comparing annotation samples**
  ``processing.compare_annotations`` and ``Comparitor`` widen NumPy integer
  sample positions before subtracting them, so unsigned positions can match
  nearby later detections and large signed differences do not wrap.

**Preserve small differences in large integer local-peak inputs**
  ``processing.find_local_peaks`` uses its exact scanning path for integer
  values outside the range represented exactly by a double, avoiding false
  or missing peaks caused by rounding in the fast maximum filter.

**Pad waveform averages with zero in physical units**
  ``processing.sigavg`` no longer includes unrelated end-of-record samples
  through negative NumPy indices. Out-of-range positions at either end use
  the channel's digital baseline so padding is zero after calibration,
  including records with a nonzero baseline.

Version 5.3.0 (Nov 2026)
------------------------

This release adds cloud writing for records and annotations, chunked DataFrame
reads, and support for writing flattened multi-segment records. It also
corrects EDF calibration, mixed-rate reads, and EDF+ annotation timing and
decoding. EDF+ annotation channels are no longer treated as physiologic
signals; writing annotations from an annotation-only EDF file now requires an
explicit sampling frequency. See the entries below for details.

**Write WFDB records and annotations to cloud storage**
  ``wrsamp`` and ``wrann`` now accept cloud URIs in ``write_dir``, including
  S3, Google Cloud Storage, Azure Blob Storage, and in-memory filesystems.

**Iterate over long records as DataFrame chunks**
  ``wfdb.iter_dataframes`` reads one sample window at a time and yields
  pandas DataFrames with continuous time indexes, avoiding a full-record
  signal allocation. Ranged reads also work when the header omits the signal
  length.

**Reuse headers while iterating over DataFrame chunks**
  ``wfdb.iter_dataframes`` now reads each record and segment header only once
  per iterator, reducing requests for remote records.

**Correct EDF ADC resolution metadata**
  ``read_edf`` now reports the bit width of the inclusive digital range, so
  full 12-bit and 16-bit EDF channels report ``adc_res`` 12 and 16.

**Preserve EDF+ annotation timing with mixed signal rates**
  ``rdedfann`` now uses the fastest physiologic signal rate for annotation
  sample numbers by default. The selected rate is returned as ``fs`` in the
  info dictionary and stored in written WFDB annotation files.

**Correct EDF+ annotation sample numbers**
  ``rdedfann`` now converts onset times to sample numbers using the record's
  sampling frequency and rounds to the nearest sample in both return modes.

**Exclude EDF+ annotation channels from signal rates**
  ``read_edf`` now calculates sampling frequency and signal frames from
  physiologic channels only, and no longer includes ``EDF Annotations`` in
  ``sig_name`` or ``p_signal``. Annotation-only files have no signal rate:
  ``rdedfann`` returns ``None`` sample numbers unless ``fs`` is supplied, and
  writing a WFDB annotation from one now requires ``fs``.

**Decode EDF+ annotations from their original bytes**
  ``rdedfann`` now reads annotation text from the EDF digital samples, so
  physical calibration and other channels' sample rates do not alter it.
  EDF+ annotation text is decoded as UTF-8 by default.

**Make records read from EDF writable**
  ``read_edf`` now stores the computed baseline as a list, so its returned
  record can be written with ``Record.wrsamp``.

**Speed up EDF conversion and correct mixed-rate scaling**
  EDF writing now assembles data by block without per-sample Python lists.
  EDF reading averages higher-rate channels with NumPy and applies their ADC
  gain correctly when sample rates differ.

**Correct checksums in records read from EDF files**
  ``read_edf`` now computes channel checksums from the converted digital
  samples, including channels with inverted scales or averaged samples.

**Avoid an extra EDF data record at exact block boundaries**
  ``wfdb_to_edf`` no longer appends a full block of padding when the input
  signal length is an exact multiple of the EDF data record length.

**Write EDF records with inverted physical ranges**
  ``read_edf`` now reflects the digital scale of inverted channels so the
  returned record has a positive WFDB gain and can be written with ``wrsamp``.
  The shared signal file uses format 32 only when a converted sample cannot
  be stored as valid format 16 data.

**Parse EDF+ annotation delimiters correctly**
  ``rdedfann`` now preserves plus signs in annotation text, handles negative
  onsets, optional durations, and multiple annotations in a TAL.

**Evaluate filter gain at the requested frequency**
  ``processing.get_filter_gain`` no longer rounds up to a fixed frequency grid.
  This also fixes an ``IndexError`` at the Nyquist frequency and frequencies
  just below it.

**Preserve source segment ranges on flattened records**
  ``rdrecord`` now retains source segment names and frame ranges, including
  empty segments, in the ``source_segments`` attribute when flattening a
  multi-segment record. Each range is a ``wfdb.io.SourceSegment`` named tuple
  of ``(name, start, stop)``.

**Display annotation notes on plots**
  ``plot_wfdb`` can now display annotation auxiliary notes beside their markers
  when called with ``plot_aux=True``. Annotations on channels beyond the
  record's signals are now drawn on the correct axes, even when some channels
  have no annotations.

**Correct EDF calibration and mixed-rate reads**
  ``read_edf`` no longer shifts physical values by one digital unit, and it
  reads files with differing per-signal sample counts consistently. EDF+
  annotation decoding uses the corrected values. Physical samples can differ
  from the WFDB C ``edf2mit`` output by up to one digital unit when the EDF
  offset is fractional, because its baseline is rounded to an integer.

**Improve the README and PyPI project page**
  Add a task-focused quick start, clearer feature and documentation links, and
  more useful PyPI project metadata, including a link to the WFDB project site.

**Validate multi-segment header fields as lists**
  ``seg_name`` and ``seg_len`` must be lists with correctly typed elements.
  Invalid values now raise a clear ``TypeError`` before writing a header.

**Avoid FLAC seek failures in older SoundFile builds**
  Require SoundFile 0.13.1 or newer so supported wheels use updated bundled
  FLAC libraries. Document the separate system-library requirement.

**Read digital variable-layout records with absent channels**
  ``rdrecord`` and ``iter_dataframes`` now handle windows where a requested
  channel has no samples in any selected segment. Those channels are filled
  with a missing-value marker instead of raising ``KeyError``.

**Write flattened multi-segment records**
  Records returned by ``rdrecord`` with ``m2s=True`` can now be written after
  assigning a new record name and signal file names. Missing write metadata is
  filled in, and incompatible or unknown formats are converted to format 32.
  The caller's record is not changed. Records with conflicting source units
  require an explicit ``units`` value.

Version 5.2.0 (Nov 2026)
------------------------

This release changes ``rdann`` and ``Annotation.apply_range`` to stop before
``sampto``, as ``rdrecord`` and ``rdsamp`` do, so an annotation exactly at
``sampto`` is no longer included (pass ``sampto + 1`` for the old result).
``get_dbs`` works again after a change on PhysioNet, ``pn_dir`` reads record
names that include folders, and ``read_edf``, ``rdedfann`` and ``read_wav``
read ``physionet://`` and other URIs. Headers with a base time or date in
another form can now be read, multi-segment records are no longer left partly
written after an error, and ``wrsamp`` warns before replacing another record's
signal files. ``processing.bxb`` is a new development feature that compares
annotation files as the WFDB ``bxb`` program does. See the entries below for
details.

**Beat-by-beat comparison of annotations, as WFDB bxb does (development feature)**
  ``processing.bxb`` and ``processing.bxb_record`` compare a test annotation
  file with a reference annotation file beat by beat, as the ``bxb`` program
  of the WFDB Software Package does for the ANSI/AAMI EC57 and EC38
  standards. The result holds the comparison matrix and the QRS, VEB and SVEB
  sensitivity and positive predictivity, RMS RR interval error and shutdown
  statistics, and gives the standard and line-format reports of ``bxb``.
  ``processing.sumstats`` combines the results of several records, as WFDB
  ``sumstats`` does. These functions were written from the standards and the
  documentation of ``bxb``, and by comparing results with it, without
  reference to its source code. They are a development feature: their
  interface may change, and their results should be checked against ``bxb``
  before they are used for reporting.

**Read EDF and WAV files from physionet:// and other URIs**
  ``read_edf``, ``rdedfann`` and ``read_wav`` now take a cloud URI as
  ``record_name``, such as ``physionet://chbmit/1.0.0/chb01/chb01_01.edf``,
  ``s3://`` or ``memory://``, as ``rdrecord`` and ``rdann`` do, and read the
  file without saving a copy. All of the functions that take ``pn_dir`` now
  also take a ``physionet://`` URI. Output files written by ``wfdb_to_edf``
  and ``wfdb_to_wav`` for a URI are named after the record, on every platform.

**Correct unsorted peak guesses near the signal start**
  ``correct_peaks`` now applies its boundary adjustment to every near-start
  guess, including those listed after a later guess, while preserving input
  order.

**Make docstring examples runnable outside the source tree**
  Replace repository-only sample paths with published PhysioNet records or
  self-contained data, correct API calls and doctest prompts, and show printed
  output accurately.

**Read format 61 signals with smooth_frames=False**
  ``rdrecord`` with ``physical=False`` and ``smooth_frames=False`` failed with ``ValueError: invalid literal for int()`` for records with format 61 signals, which are stored in big-endian byte order. These signals are now returned with the requested native dtype.

**Fix peak detection for integer signals**
  ``processing.find_peaks`` now compares integer samples without subtracting
  them first, avoiding overflow for signed samples and errors for unsigned
  samples. This also preserves differences between large 64-bit integers.

**Fix get_dbs, which PhysioNet no longer supported**
  ``get_dbs`` failed with a 404 error, because PhysioNet has removed the
  ``/rest/database-list/`` page that it read. It now lists databases through
  PhysioNet's API, using the ``physionet`` package, and returns the slug and
  title of the latest version of each database, as before.

**Read headers whose base time or date is in another form**
  ``rdheader``, ``rdrecord`` and ``rdsamp`` raised a ``ValueError`` for a
  header whose base date is not a DD/MM/YYYY date (such as ``25.12.2020``)
  or whose base time is a year, so the record could not be read at all, and
  read a time and date in another order (such as ``05-Feb-2020 11:39:16``,
  in the 2020 and 2021 Challenges) as a wrong time. As in the WFDB library, a
  base time or date that is not valid is now ignored with a warning, and the
  rest of the record is read as usual.

**Leave no partly written multi-segment record after an error**
  ``MultiRecord.wrsamp`` wrote the main header and the earlier segments
  before finding that a later segment could not be written, which left a
  partly written record. It now writes the files into a temporary folder in
  ``write_dir``, and moves them into place only once every segment has been
  written. Files that are replaced keep their permissions, and if a move
  fails, the files already moved are removed and the replaced files are
  restored. The files are moved one at a time, so a program reading them
  meanwhile, or a crash while they are moved, can still see a mix of files.
  A segment without the signal to write, or without
  ``samps_per_frame`` when writing expanded signals, now raises a
  ``ValueError`` that names the segment, instead of a ``TypeError``. A
  relative ``wfdb_archive`` path is now relative to ``write_dir``, as it is
  for ``Record.wrsamp``.

**Normalize constant signals to the middle of the bounds**
  ``normalize_bound`` raised a ``ZeroDivisionError`` for a constant integer
  signal, and returned NaN (with warnings) for a constant float signal. Both
  are now normalized to the middle of the bounds.

**Avoid integer overflow when normalizing signal bounds**
  ``normalize_bound`` now computes the range of integer signals without
  wrapping, so large signed ranges map to the requested bounds.

**Avoid overflow with NumPy integer normalization bounds**
  ``normalize_bound`` now widens NumPy integer bounds before subtracting them,
  preventing overflow when their difference exceeds the bounds' dtype.

**Preserve small differences in large integer signals when normalizing**
  ``normalize_bound`` now subtracts an integer signal's minimum exactly before
  scaling, so large offsets do not lose small differences, and the results are
  always within the requested bounds.

**Set axis labels in plot_wfdb, and fix xlabel in plot_items**
  ``plot_wfdb`` takes ``xlabel`` and ``ylabel`` arguments, lists with a label
  for each channel, to replace the default labels (the time unit, and the
  signal names and units). ``plot_items`` failed with an ``AttributeError``
  whenever ``xlabel`` was given, and now sets the labels.

**Read PhysioNet records whose names include folders**
  ``rdrecord``, ``rdsamp``, ``rdheader`` and ``rdann`` with ``pn_dir`` now
  read a record name that starts with folders, such as
  ``rdrecord("files/p1000/p10000032/s40689238/40689238",
  pn_dir="mimic-iv-ecg/1.0")``, from those folders of the database. They used
  to drop the folders and look for the record at the top of the database,
  which failed with a "file not found" error.

**rdann and Annotation.apply_range stop before sampto, as rdrecord does**
  ``rdann`` and ``Annotation.apply_range`` used to include an annotation at
  sample ``sampto``, while ``rdrecord`` and ``rdsamp`` return the samples
  before it. They now return the annotations before ``sampto``, so that the
  annotations read with a signal's range are within that signal, and can index
  it when read with ``shift_samps=True``. To include an annotation at a given
  sample, pass one more than that sample as ``sampto``.

**Warn when a renamed record would replace another record's signal files**
  ``Record.wrsamp`` names the header after ``record_name`` and the signal
  files by ``file_name``, so writing a record that was read under a new
  ``record_name``, next to the original, replaced the original's signal
  files without any sign. It now warns when writing would replace an existing
  signal file that the record's own header in ``write_dir`` does not list. The
  documentation of ``wrsamp`` and ``file_name`` now explains this, with an
  example of saving a copy under a new name.

**Write custom annotation symbols of more than one character**
  ``wrann`` and ``Annotation.wrann`` failed with a ``TypeError`` when a custom
  label's symbol had more than one character, and symbols were limited to 3
  characters. Symbols may now have any length (and still no whitespace), as
  in the WFDB library, and ``rdann`` reads them. A custom label is stored as
  a note of at most 255 Latin-1 characters ("<number> <symbol>
  <description>"), and ``wrann`` now raises a ``ValueError`` for a longer one,
  which it used to write as a file that could not be read.

**Do not apply skew twice to rewritten records**
  ``Record.wrsamp`` no longer writes the skew of a record whose signals were read with the skew applied. The samples were written already aligned but the header kept the skew, so reading the new record shifted those signals a second time. ``wfdb_to_wav`` with ``write_header=True`` had the same problem. The ``skew`` attribute is unchanged, and records read with ``ignore_skew=True`` are still written with their skew.

Version 5.1.0 (Nov 2026)
------------------------

This release reads records and annotations from PhysioNet through the
``physionet`` package, which is now a dependency: ``physionet://`` URIs, such
as ``physionet://mitdb/1.0.0/100``, and ``pn_dir`` read only the parts of signal
files that are needed, ``dl_database`` and ``dl_files`` check each file against
the database's checksums, and restricted databases can be read and downloaded
after ``physionet login``. ``xqrs_detect`` and ``gqrs_detect`` are faster.
``rdann`` now raises a ``ValueError`` when ``sampto`` is not greater than
``sampfrom``, including ``sampto=0``, which used to return every annotation.
See the entries below for details.

**Keep every annotation field in Annotation.apply_range**
  ``Annotation.apply_range`` now keeps the same annotations in every field that is set, including ``symbol`` and ``description``, which it used to leave unchanged, so that they no longer matched ``sample``. It also works on an annotation read by ``rdann`` with the default ``return_label_elements``, where ``label_store`` is ``None``, instead of failing with a ``TypeError``.

**Download databases with the physionet package**
  ``dl_database`` and ``dl_files`` now download files with the ``physionet`` package (0.7.2 or later). Each file is checked against the database's checksum list, a file that is already saved with the right content is skipped, a partly downloaded one is completed, ``overwrite=True`` replaces each saved file only once its new copy is complete and checked, and restricted databases can be downloaded after ``physionet login``. ``dl_database`` finds which annotation files there are from the checksum list instead of requesting each one, which makes downloading three MIT-BIH records with their annotations take 20 requests instead of 38. ``dl_files`` reports a file that the database does not have before downloading anything. With a database index URL set by ``set_db_index_url``, files are downloaded from that server as before.

**Speed up find_local_peaks and XQRS**
  ``find_local_peaks`` now finds the largest value around each sample with ``scipy.ndimage.maximum_filter1d`` instead of checking each sample in a Python loop, and is about 20 times faster. ``XQRS`` uses it to find candidate peaks, so QRS detection with ``xqrs_detect`` is about 10 times faster (35 ms instead of 380 ms for the 30-minute MIT-BIH record 100). The peaks found are unchanged.

**Speed up gqrs_detect**
  ``gqrs_detect`` now computes its smoothing and QRS matched filters for the whole signal with NumPy, instead of one sample at a time in Python, and is about 4 times faster (0.36 seconds instead of 1.4 for the 30-minute MIT-BIH record 100). The detected QRS locations are unchanged.

**Read records from PhysioNet with physionet:// URIs**
  ``rdrecord``, ``rdsamp``, ``rdheader`` and ``rdann`` now accept ``physionet://`` URIs such as ``physionet://mitdb/1.0.0/100``, which read from PhysioNet through the ``physionet`` package, now a dependency. Only the parts of signal files that are needed are read (header and annotation files are read in full), and restricted databases can be read after ``physionet login``. ``fsspec`` 2025.7.0 or later is now required, as for the ``physionet`` package.

**Read pn_dir records through the physionet package**
  Records, headers and annotations read with ``pn_dir`` (and EDF and WAV files read with ``pn_dir`` by ``read_edf`` and ``read_wav``) are now read through the ``physionet`` package, as ``physionet://`` URIs are. Open databases are read directly from the AWS open data mirror, which makes reading part of a record from MIT-BIH about twice as fast, and restricted databases can be read after ``physionet login``. The latest version of a database, used when ``pn_dir`` has no version, is now looked up with the PhysioNet API once per database, and retried if PhysioNet is unavailable, instead of reading the database's web page on every call. With a database index URL set by ``set_db_index_url``, files are read from that server as before.

**Apply sampfrom in rdann without sampto**
  ``rdann`` now returns only the annotations at or after ``sampfrom`` when ``sampto`` is not given. It used to ignore ``sampfrom`` and return every annotation in the file, which gave negative sample numbers with ``shift_samps=True``. ``Annotation.apply_range`` with ``sampto=None`` now keeps annotations up to the end without assuming they are sorted, and no longer fails when there are no annotations.

**Reject sampto=0 in rdann, and summarize no annotations**
  ``rdann`` now raises a ``ValueError`` when ``sampto`` is not greater than ``sampfrom``, as ``rdrecord`` does. ``sampto=0`` used to return every annotation, but with ``sampfrom`` it returned none. With ``summarize_labels=True``, ``rdann`` and ``Annotation.get_contained_labels`` no longer fail with ``KeyError: 'n_occurrences'`` when there are no annotations, for example when ``sampfrom`` is after the last one.

Version 5.0.0 (Nov 2026)
------------------------

This release adds ``.wfdb`` archive support and requires Python 3.11 or newer.
``ann2rr`` now uses only beat annotations by default and selects intervals by
their ending times in the record. Pass ``all_annotations=True`` to include
non-beat annotations. See the entries below for details.

**Fix writing channels that are entirely NaN**
  ``wrsamp`` and ``calc_adc_params`` no longer fail with ``cannot convert float NaN to integer`` when a channel contains only NaN values, and no longer emit an "All-NaN slice encountered" warning for such channels.

**Reject negative infinity in expanded signals**
  ``calc_adc_params`` now raises "Signal contains inf" when ``e_p_signal`` contains ``-inf``, as it already did for ``p_signal``, instead of failing with ``cannot convert float NaN to integer``.

**Return 0 from calc_mean_hr when all intervals are filtered out**
  ``calc_mean_hr`` now returns 0 when ``min_rr`` or ``max_rr`` removes every R-R interval, as it does for empty input. It used to return ``nan`` with a "Mean of empty slice" warning.

**Accept lists in calc_mean_hr with min_rr or max_rr**
  ``calc_mean_hr`` now accepts R-R intervals as a list when ``min_rr`` or ``max_rr`` is given. It used to raise a ``TypeError``.

**Allow empty annotations in compare_annotations**
  ``compare_annotations`` (``Comparitor.compare``) no longer fails when the reference or test annotations are empty, for example when a detector finds no beats. The sensitivity or positive predictivity that cannot be computed is NaN.

**Fix test annotations matched twice in compare_annotations**
  ``compare_annotations`` (``Comparitor.compare``) could match one test annotation to two reference annotations, which gave a negative number of false positives and a positive predictivity above 1. Each test annotation is now matched at most once. Results are unchanged when no annotation was matched twice.

**Check that QRS indices are increasing in compute_hr**
  ``compute_hr`` now raises a ``ValueError`` if ``qrs_inds`` is not strictly increasing. Repeated beats used to give a ``ZeroDivisionError`` (or ``inf`` for NumPy arrays), and beats out of order gave a negative heart rate.

**Use the last RR interval in compute_hr**
  ``compute_hr`` now uses the interval between the last two beats for the heart rate after the last beat, instead of repeating the previous rate. With exactly two beats it now returns their rate after the second beat instead of only NaN.

**Fix correct_peaks for peaks near the start of the signal**
  ``correct_peaks`` no longer moves peaks within ``search_radius`` samples of the start of the signal to the wrong place, often before the start (for example ``-6`` instead of ``0``). The offset for these peaks was subtracted instead of added.

**Fix search window in correct_peaks**
  ``correct_peaks`` now searches ``search_radius`` samples on both sides of each peak, including the last sample of the signal. It used to stop one sample short on the right, so a peak exactly ``search_radius`` samples after the original index, or on the last sample, could not be found.

**Fix parameter handling in csv_to_wfdb**
  ``csv_to_wfdb`` now builds the same ``Record`` whether it writes files or returns the record (``record_only=True``), so all header parameters (such as ``dat_file_name``, ``counter_freq``, ``base_counter``, ``skew``, ``byte_offset``, ``adc_res`` and ``adc_zero``) are written instead of being ignored. Single values of any numeric type (including NumPy scalars and floats) are used for all channels, and an explicit ``0`` is no longer replaced by the default. The initial value and checksum are computed from the digital samples, and the default ADC resolution now follows ``fmt`` in both modes. The ``samps_per_frame``, ``init_value`` and ``checksum`` arguments are deprecated and ignored. The parameter descriptions in the docstring have been corrected.

**Reduce memory use when writing format 16, 24 and 32 signals**
  ``wrsamp`` and ``wr_dat_file`` now convert signals in formats 16, 24 and 32 to bytes directly, instead of splitting every sample into separate byte arrays. For a 4-hour, 8-channel record at 500 Hz, the peak memory used to pack a format 16 signal drops from about 2 GB to 110 MB, and packing is about 13 times faster. The written files are unchanged.

**Drop support for Python 3.9 and 3.10**
  The minimum supported Python version is now 3.11. The package is now also tested with Python 3.14 and 3.15.

**Include the right edge of the window in find_local_peaks**
  ``find_local_peaks`` now compares each sample with ``radius`` samples on both sides, as documented, and is about 3 times faster. The sample ``radius`` positions to the right, and the last sample of the signal, used to be left out. ``XQRS`` uses ``find_local_peaks`` to find candidate peaks, so its detections may change slightly.

**Read units containing any non-whitespace characters**
  Header files with units containing characters such as ``.`` or ``*`` (for example ``a.u.`` or ``uV*s``) are now read correctly. The units used to be cut off at the first such character, and the rest of the signal line (ADC resolution, ADC zero and so on) ended up in the signal name.

**Read records from memory**
  ``rdrecord``, ``rdsamp`` and ``rdheader`` now accept ``memory://`` URIs, which read from the in-memory filesystem of ``fsspec``, as ``rdann`` already did. This also makes cloud URIs work for signal files and multi-segment records on Windows, where the file paths used to be built with backslashes.

**Raise minimum versions of dependencies**
  The minimum versions of ``soundfile`` (0.12.1), ``matplotlib`` (3.8.0) and ``requests`` (2.31.0) have been raised. The minimum versions of the other dependencies are unchanged.

**Write variable layout multi-segment records**
  ``MultiRecord.wrsamp`` can now write variable layout multi-segment records, such as those read with ``rdrecord(..., m2s=False)``. Previously, writing the layout header failed because its signals have no samples (format ``0``, file name ``~``).

**Validate multi-segment record writes**
  ``MultiRecord.wrsamp`` now checks the segment lengths and sampling frequencies of all segments before writing any files, instead of writing a segment whose length does not match ``seg_len``.

**Plot annotations outside the signal**
  ``plot_items`` and ``plot_wfdb`` no longer fail with an ``IndexError`` when an annotation is after the end of the signal, and no longer draw annotations before the start of the signal at values from its end. These annotations are now drawn at zero, as for channels without a signal, with a warning suggesting ``shift_samps=True`` in ``rdann``.

**Raise errors when rdann reads fs from the header**
  When an annotation file doesn't store the sampling frequency, ``rdann`` reads it from the record's header. It used to ignore any error from that, for example when the header couldn't be downloaded from PhysioNet, and return the annotation with ``fs`` set to ``None``. Now only a missing header is ignored, and other errors are raised.

**Accept pathlib.Path record names in rdann**
  ``rdann`` now accepts a ``pathlib.Path`` (or other path-like object) as ``record_name``, as ``rdrecord``, ``rdsamp`` and ``rdheader`` already do. It used to fail with ``TypeError: unsupported operand type(s) for +``.

**Speed up reading annotation files**
  ``rdann`` now decodes annotation files using plain Python integers instead of indexing a NumPy array one byte at a time. Reading annotations is about 2 to 3 times faster (a 100,000-beat file now takes about 75 ms instead of 175). The annotations read are unchanged.

**Fix rdedfann for EDF+ annotations**
  ``rdedfann`` no longer passes obsolete options to ``read_edf`` and can read
  EDF+ annotations again.

**Read remote FLAC files with fewer requests**
  Remote files read in small pieces, such as FLAC signal files on PhysioNet, now use larger requests as reading goes on (up to 8 MB) instead of one request for every 32 kB. Reading a 7.4 MB MIMIC-IV signal file now takes 8 requests instead of 229, which saves several minutes per file.

**Add timeouts to requests for remote files**
  Requests for files on PhysioNet (and other web servers) now time out if the server does not respond (10 seconds to connect, and 60 seconds to wait for data), instead of waiting indefinitely. A ``requests.exceptions.Timeout`` (``ConnectTimeout`` or ``ReadTimeout``) is raised in this case.

**Fix resample_sig for signals starting with NaN**
  ``resample_sig`` (and the functions that use it, such as ``resample_singlechan`` and ``resample_multichan``) returned only NaN values if the signal started with NaN. Leading NaN values now take the first valid value before resampling, as trailing ones already took the last.

**Keep missing values missing in resample_sig**
  ``resample_sig`` (and ``resample_singlechan`` and ``resample_multichan``) now return NaN for output samples with a NaN input value within half an output sample spacing (or within one input sample, when upsampling). NaN values are still filled before resampling, so the rest of the signal is unaffected, but the filled values are no longer returned as if they were real data.

**Use Ruff for formatting and linting**
  The ``dev`` extra now includes ``ruff`` instead of ``black`` and ``pylint``, and the code is formatted with a line length of 88 (the default).

**Speed up the digital range check when writing records**
  ``wrsamp`` now uses NumPy to check that digital samples are within the range of each format, instead of Python's built-in ``min`` and ``max``. Writing a 4-hour, 8-channel record at 500 Hz in format 16 now takes about 0.5 seconds instead of 3. The written files are unchanged.

**Clarify smooth_frames behavior**
  The ``rdrecord`` documentation now explains that ``smooth_frames`` controls
  the output structure even when every channel has one sample per frame.

**Fix smoothing of frames with invalid samples**
  When a signal with more than one sample per frame is read with ``smooth_frames=True`` (the default), a frame that contains an invalid sample now gives an invalid value (NaN in physical units), as in the WFDB library. The invalid sample value used to be averaged with the other samples, which gave a large but valid-looking value.

**Fix drifting timestamps in to_dataframe**
  ``Record.to_dataframe`` now computes the timestamp of each sample directly. Previously the sample interval was truncated to whole nanoseconds and added up, so for sampling frequencies such as 360 Hz or 62.4725 Hz the timestamps drifted by up to about a millisecond over a long record. The index no longer has a ``freq`` attribute.

**Build with uv_build**
  The package is now built with ``uv_build`` instead of ``hatchling``. The version number is specified in ``pyproject.toml`` only, and ``wfdb.__version__`` is read from the installed package metadata (it is ``"unknown"`` if the package is run from a source tree without being installed).

**Read and write .wfdb archive files**
  A record can be stored in a single ``.wfdb`` archive file (a ZIP file containing the header and signal files). ``rdrecord``, ``rdsamp``, ``rdheader`` and ``rdann`` read a record or its annotations from an archive when given the path of the archive, and ``wrsamp`` writes a record into an archive with the new ``wfdb_archive`` argument. Multi-segment records are supported, with all their segments stored in the same archive.

**Fix wfdb_to_mat encoding, slicing, and header metadata**
  ``wfdb_to_mat`` now passes ``channels`` to ``rdrecord`` and computes output
  length from the sliced record. It writes 8-bit (offset binary) and 32-bit
  MAT files with proper data types and 8-byte MAT v5 padding, preserves missing
  samples, clears already-applied channel skew, handles unset ``adc_res`` and
  ``adc_zero`` fields, updates header checksums, and handles forward-slash
  record paths across platforms.

**Preserve shifted samples in wfdb_to_mat**
  ``wfdb_to_mat`` now chooses a wider output type when samples shifted by
  ``adc_zero`` do not fit the type suggested by the header. It raises an error
  instead of silently wrapping samples that cannot fit in a 32-bit MAT file.

**Fix channels and invalid samples in wfdb_to_wav**
  ``wfdb_to_wav`` now writes only the signals selected with ``channels``, which it used to ignore. Invalid samples are written as -32768, the format 16 invalid value, so they are read back as missing. For a resolution below 12 bits, they used to overflow and become 0. The samples are now read as digital values, instead of being read as physical values and converted back.

**Fix wfdb_to_wav failing for every record**
  ``wfdb_to_wav`` now reads the record with ``smooth_frames=True``, since a .wav file holds one sample per frame for each signal. It used to read the expanded samples, and then failed with an ``AttributeError`` (a ``TypeError`` in older versions) because ``p_signal`` was not set. With ``write_header=True``, the header now gives one sample per frame for each signal, as in the .wav file, so a converted record with more than one sample per frame can be read back.

**Reject aux notes that can't be written in wrann**
  ``wrann`` now raises a ``ValueError`` if an ``aux_note`` is longer than 255 characters or contains characters outside Latin-1, instead of writing a corrupt annotation file. A longer note used to be read back as many extra annotations, and other characters were silently replaced.

**Allow None for annotations without notes in wrann**
  ``wrann`` now accepts ``None`` in ``aux_note`` for annotations without a note, as the docstring says, and writes it the same way as an empty string. It used to raise a ``TypeError``.

**Accept sample numbers stored as floats in wrann**
  ``wrann`` now accepts a ``sample`` array of floats if all the values are whole numbers (as from a detector that returns floats), and writes them as integers. Other floats, such as ``20.5`` or NaN, raise a ``ValueError``. Float arrays used to fail with ``TypeError: ufunc 'bitwise_and' not supported for the input types``.

**Fix writing physical-only records**
  ``Record.wrsamp`` now temporarily converts physical signals to digital samples
  when the corresponding digital signals are missing, then restores the record.

**Speed up XQRS backsearch**
  ``XQRS`` no longer rechecks peaks that an earlier backsearch already rejected. Long stretches without QRS complexes (for example a disconnected lead) made detection take quadratic time, up to hours on long records. The detected QRS locations are unchanged.

**Prevent ADC overflow with float32 signals**
  ADC parameter calculation and conversion now use float64 arithmetic for
  float32 physical signals. Samples near digital format limits no longer fail
  validation because of a one-value rounding overflow.

**Use only beat annotations in ann2rr**
  ``ann2rr`` now computes intervals between beat (QRS) annotations only, as the WFDB ``ann2rr`` program does, so rhythm changes, comments and other non-beat annotations no longer split an RR interval in two. The first interval is still measured from the start of the record. To include all annotations as before, pass ``all_annotations=True`` (the equivalent of ``ann2rr -A``).

**Select intervals by time in the record in ann2rr**
  The ``start_time`` and ``stop_time`` arguments of ``ann2rr`` now select intervals by when they end in the record, as the WFDB ``ann2rr`` program's ``-f`` and ``-t`` options do, and the first interval is measured from ``start_time``. They used to filter intervals by their length instead. A ``start_time`` after ``stop_time`` now raises a ``ValueError``.

**Fix get_contained_labels with pandas 3**
  ``Annotation.get_contained_labels`` no longer raises a ``TypeError`` with pandas 3 for annotations without ``label_store`` values, such as those read with ``rdann`` by default.

**Fix record name and add write directory in csv2ann**
  ``csv2ann`` now takes the record name from the CSV file name without its directory and extension, as ``csv_to_wfdb`` does. Paths with a directory (such as ``data/x.csv``) used to fail when writing, and ``./x.csv`` gave an empty name. The annotation file is written to the new ``write_dir`` argument, which defaults to the current directory. ``csv2ann`` is now included in the API documentation.

**Fix rr2ann with times and with a symbol column**
  ``rr2ann`` with ``as_time=True`` now rounds times to the nearest sample instead of truncating them, so (for example) 0.29 seconds at 100 Hz gives sample 29 instead of 28. Intervals in samples with a second column of symbols no longer raise an error, and fractional intervals are summed before rounding.

Version 4.3.1 (Feb 2026)
-----------------------------

**Add fix for Pandas 3.0**
  In Pandas 3.0, DataFrame.set_index() no longer accepts a raw numpy array as the keys argument. Adds fix by wrapping the index array in pd.Index().

**Fix ADC formula**
  Fixes [formula for ADC gain](https://github.com/MIT-LCP/wfdb-python/pull/551).

**Allow multifrequency signals to be written with expanded argument**
  Allow [expanded parameter in MultiRecord.wrsamp](https://github.com/MIT-LCP/wfdb-python/pull/555).

**Date validation when saving multi-segment records**
  Adds validation check for date and time fields in headers when saving  multi-segment records.

Version 4.3.0 (Apr 2025)
-----------------------------

**Bump Sphinx to 7.0.0**
  Bump Sphinx to 7.0.0

**Integrate `fsspec` for reading WFDB files from the cloud**
  Enables reading WFDB files from cloud URLs


Version 4.2.0 (Jan 2025)
-----------------------------

**Add support for Numpy 2.0**
  Fixes were added to address [changes to type promotion](https://numpy.org/devdocs/numpy_2_0_migration_guide.html#changes-to-numpy-data-type-promotion) that led to overflow errors (e.g. https://github.com/MIT-LCP/wfdb-python/issues/493). 

**Fix UnboundLocalError in GQRS algorithm**
  Fixes the GQRS algorithm to address an `UnboundLocalError`.

**Support write directory in `csv_to_wfdb`**
  `write_dir` can now be specified when calling `csv_to_wfdb`.

**Use uv for for package management**
  Moves package management from poetry to uv.

**Fix misordered arguments in `util.lines_to_file`**
  Fixes misordered arguments in `util.lines_to_file`.

**Allow signals to be written with unique samples per frame**
  Adds capability to write signal with unique samps_per_frame to `wfdb.io.wrsamp`.

**Allow expanded physical signal in `calc_adc_params`**
  Updates `calc_adc_params` to allow an expanded physical signal to be passed. Previously only a non-expanded signal was allowed.

**Allow selection of channels when converting to EDF**
  Fixes the `wfdb-to_edf()` function to support an optional channels argument.

**Migrates Ricker wavelet from SciPy to WFDB after deprecation**
  The Ricker wavelet (`scipy.signal.ricker`) was removed in SciPy v1.15, so the original implementation was migrated to the WFDB package.

**Miscellaneous style and typing fixes**
  Various fixes were made to code style and handling of data types.


Version 4.1.2 (June 2023)
-----------------------------

**Handle more than 8 compressed signals in wrsamp**
  Previously, the package did not support writing of compressed records with more than 8 channels.

**Use int64 instead of int for ann2rr**
  Fixes 'np has no attribute np.int' error raised when running ann2rr.

Version 4.1.1 (April 2023)
-----------------------------

**Remove upper bound on dependencies**
  Previously, the package provided restrictive caps on version number of dependencies. These caps have been removed.

**Miscellaneous style and typing fixes**
  Various fixes were made to code style and handling of data types.


Version 4.1.0 (December 2022)
-----------------------------

**Converting a record into a DataFrame**
  The new method :meth:`wfdb.Record.to_dataframe` can be used to convert signal data from a Record object into a Pandas DataFrame, which can then be manipulated using Pandas methods.

**Locating signals in a multi-segment record**
  The new method :meth:`wfdb.MultiRecord.contained_ranges` can be used to search for time intervals within a record that contain a specific channel.  The method :meth:`wfdb.MultiRecord.contained_combined_ranges` searches for time intervals that contain several specific channels at once.

**Writing custom annotation symbols**
  The :func:`wfdb.wrann` function can now be used to write annotations with custom annotation types (``symbol`` strings.)  Custom annotation types must be defined using the ``custom_labels`` argument.

**Correct rounding when converting floating-point signal data**
  When calling :func:`wfdb.wrsamp` with a ``p_signal`` argument, input values will be *rounded* to the nearest sample value rather than being *truncated* towards zero.  The same applies to the :meth:`wfdb.Record.adc` method.

**Writing signals in compressed format**
  The :func:`wfdb.wrsamp` function, and the :meth:`wfdb.Record.wrsamp` method, now support writing compressed signal files.  To write a compressed file, set the ``fmt`` value to ``"508"`` (for an 8-bit channel), ``"516"`` (for a 16-bit channel), or ``"524"`` (for a 24-bit channel).

**Decoding non-ASCII text in EDF files**
  The :func:`wfdb.io.convert.edf.read_edf` and :func:`wfdb.io.convert.edf.rdedfann` functions now take an optional argument ``encoding``, which specifies the character encoding for text fields.  ISO-8859-1 encoding is used by default, in contrast to older versions of the package which used UTF-8.

**Bug fixes when writing signal metadata**
  When calling :meth:`wfdb.Record.wrsamp`, the checksum and samples-per-frame fields in the header file will correctly match the signal data, rather than relying on attributes of the Record object.
