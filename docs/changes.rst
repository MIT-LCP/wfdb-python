Recent Changes
==============

This page lists recent changes in the `wfdb` package (since version 4.0.0) that may be relevant if you are upgrading from an older version of the package.  For the complete history of changes in the package, please refer to the `development repository`_ on GitHub.

.. _development repository: https://github.com/MIT-LCP/wfdb-python

Unreleased
----------

**Drop support for Python 3.9 and 3.10**
  The minimum supported Python version is now 3.11. The package is now also tested with Python 3.14 and 3.15.

**Raise minimum versions of dependencies**
  The minimum versions of ``soundfile`` (0.12.1), ``matplotlib`` (3.8.0) and ``requests`` (2.31.0) have been raised. The minimum versions of the other dependencies are unchanged.

**Read units containing any non-whitespace characters**
  Header files with units containing characters such as ``.`` or ``*`` (for example ``a.u.`` or ``uV*s``) are now read correctly. The units used to be cut off at the first such character, and the rest of the signal line (ADC resolution, ADC zero and so on) ended up in the signal name.

**Add timeouts to requests for remote files**
  Requests for files on PhysioNet (and other web servers) now time out if the server does not respond (10 seconds to connect, and 60 seconds to wait for data), instead of waiting indefinitely. A ``requests.exceptions.Timeout`` (``ConnectTimeout`` or ``ReadTimeout``) is raised in this case.

**Fix smoothing of frames with invalid samples**
  When a signal with more than one sample per frame is read with ``smooth_frames=True`` (the default), a frame that contains an invalid sample now gives an invalid value (NaN in physical units), as in the WFDB library. The invalid sample value used to be averaged with the other samples, which gave a large but valid-looking value.

**Build with uv_build**
  The package is now built with ``uv_build`` instead of ``hatchling``. The version number is specified in ``pyproject.toml`` only, and ``wfdb.__version__`` is read from the installed package metadata (it is ``"unknown"`` if the package is run from a source tree without being installed).

**Speed up XQRS backsearch**
  ``XQRS`` no longer rechecks peaks that an earlier backsearch already rejected. Long stretches without QRS complexes (for example a disconnected lead) made detection take quadratic time, up to hours on long records. The detected QRS locations are unchanged.

**Fix test annotations matched twice in compare_annotations**
  ``compare_annotations`` (``Comparitor.compare``) could match one test annotation to two reference annotations, which gave a negative number of false positives and a positive predictivity above 1. Each test annotation is now matched at most once. Results are unchanged when no annotation was matched twice.

**Fix record name and add write directory in csv2ann**
  ``csv2ann`` now takes the record name from the CSV file name without its directory and extension, as ``csv_to_wfdb`` does. Paths with a directory (such as ``data/x.csv``) used to fail when writing, and ``./x.csv`` gave an empty name. The annotation file is written to the new ``write_dir`` argument, which defaults to the current directory. ``csv2ann`` is now included in the API documentation.

**Speed up the digital range check when writing records**
  ``wrsamp`` now uses NumPy to check that digital samples are within the range of each format, instead of Python's built-in ``min`` and ``max``. Writing a 4-hour, 8-channel record at 500 Hz in format 16 now takes about 0.5 seconds instead of 3. The written files are unchanged.

**Fix parameter handling in csv_to_wfdb**
  ``csv_to_wfdb`` now builds the same ``Record`` whether it writes files or returns the record (``record_only=True``), so all header parameters (such as ``dat_file_name``, ``counter_freq``, ``base_counter``, ``skew``, ``byte_offset``, ``adc_res`` and ``adc_zero``) are written instead of being ignored. Single values of any numeric type (including NumPy scalars and floats) are used for all channels, and an explicit ``0`` is no longer replaced by the default. The initial value and checksum are computed from the digital samples, and the default ADC resolution now follows ``fmt`` in both modes. The ``samps_per_frame``, ``init_value`` and ``checksum`` arguments are deprecated and ignored. The parameter descriptions in the docstring have been corrected.

**Reduce memory use when writing format 16, 24 and 32 signals**
  ``wrsamp`` and ``wr_dat_file`` now convert signals in formats 16, 24 and 32 to bytes directly, instead of splitting every sample into separate byte arrays. For a 4-hour, 8-channel record at 500 Hz, the peak memory used to pack a format 16 signal drops from about 2 GB to 110 MB, and packing is about 13 times faster. The written files are unchanged.

**Use the last RR interval in compute_hr**
  ``compute_hr`` now uses the interval between the last two beats for the heart rate after the last beat, instead of repeating the previous rate. With exactly two beats it now returns their rate after the second beat instead of only NaN.

**Check that QRS indices are increasing in compute_hr**
  ``compute_hr`` now raises a ``ValueError`` if ``qrs_inds`` is not strictly increasing. Repeated beats used to give a ``ZeroDivisionError`` (or ``inf`` for NumPy arrays), and beats out of order gave a negative heart rate.

**Fix rr2ann with times and with a symbol column**
  ``rr2ann`` with ``as_time=True`` now rounds times to the nearest sample instead of truncating them, so (for example) 0.29 seconds at 100 Hz gives sample 29 instead of 28. Intervals in samples with a second column of symbols no longer raise an error, and fractional intervals are summed before rounding.

**Use Ruff for formatting and linting**
  The ``dev`` extra now includes ``ruff`` instead of ``black`` and ``pylint``, and the code is formatted with a line length of 88 (the default).

**Fix search window in correct_peaks**
  ``correct_peaks`` now searches ``search_radius`` samples on both sides of each peak, including the last sample of the signal. It used to stop one sample short on the right, so a peak exactly ``search_radius`` samples after the original index, or on the last sample, could not be found.

**Fix correct_peaks for peaks near the start of the signal**
  ``correct_peaks`` no longer moves peaks within ``search_radius`` samples of the start of the signal to the wrong place, often before the start (for example ``-6`` instead of ``0``). The offset for these peaks was subtracted instead of added.

**Fix resample_sig for signals starting with NaN**
  ``resample_sig`` (and the functions that use it, such as ``resample_singlechan`` and ``resample_multichan``) returned only NaN values if the signal started with NaN. Leading NaN values now take the first valid value before resampling, as trailing ones already took the last.

**Keep missing values missing in resample_sig**
  ``resample_sig`` (and ``resample_singlechan`` and ``resample_multichan``) now return NaN for output samples with a NaN input value within half an output sample spacing (or within one input sample, when upsampling). NaN values are still filled before resampling, so the rest of the signal is unaffected, but the filled values are no longer returned as if they were real data.

**Reject aux notes that can't be written in wrann**
  ``wrann`` now raises a ``ValueError`` if an ``aux_note`` is longer than 255 characters or contains characters outside Latin-1, instead of writing a corrupt annotation file. A longer note used to be read back as many extra annotations, and other characters were silently replaced.

**Allow None for annotations without notes in wrann**
  ``wrann`` now accepts ``None`` in ``aux_note`` for annotations without a note, as the docstring says, and writes it the same way as an empty string. It used to raise a ``TypeError``.

**Read and write .wfdb archive files**
  A record can be stored in a single ``.wfdb`` archive file (a ZIP file containing the header and signal files). ``rdrecord``, ``rdsamp``, ``rdheader`` and ``rdann`` read a record or its annotations from an archive when given the path of the archive, and ``wrsamp`` writes a record into an archive with the new ``wfdb_archive`` argument.

**Use only beat annotations in ann2rr**
  ``ann2rr`` now computes intervals between beat (QRS) annotations only, as the WFDB ``ann2rr`` program does, so rhythm changes, comments and other non-beat annotations no longer split an RR interval in two. The first interval is still measured from the start of the record. To include all annotations as before, pass ``all_annotations=True`` (the equivalent of ``ann2rr -A``).

**Include the right edge of the window in find_local_peaks**
  ``find_local_peaks`` now compares each sample with ``radius`` samples on both sides, as documented, and is about 3 times faster. The sample ``radius`` positions to the right, and the last sample of the signal, used to be left out. ``XQRS`` uses ``find_local_peaks`` to find candidate peaks, so its detections may change slightly.

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
