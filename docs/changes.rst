Recent Changes
==============

This page lists recent changes in the `wfdb` package (since version 4.0.0) that may be relevant if you are upgrading from an older version of the package.  For the complete history of changes in the package, please refer to the `development repository`_ on GitHub.

.. _development repository: https://github.com/MIT-LCP/wfdb-python

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
