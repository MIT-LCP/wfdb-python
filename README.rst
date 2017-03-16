wfdb-python
===========

|Build Status|

.. figure:: https://raw.githubusercontent.com/MIT-LCP/wfdb-python/master/demoimg1.png
   :alt: wfdb signals

Introduction
------------

Native python scripts for reading and writing WFDB signals and
annotations.

Usage
-----

Distribution hosted on pypi. No need to manualy download or clone this
repository. To install the package, run from your terminal:
``pip install wfdb``

See the **wfdbdemo.ipynb** file for example scripts on how to call the
functions.

Reading Signals
~~~~~~~~~~~~~~~

**rdsamp** - Read a WFDB record and return the signal and record descriptors as attributes in a wfdb.Record object.

::
    record = rdsamp(recordname, sampfrom=0, sampto=None, channels=None, physical=True, m2s=True)

Example Usage:

::

    import wfdb
    record = wfdb.rdsamp('100', sampto=2000)

Input Arguments:

-  ``recordname`` (mandatory) - The name of the WFDB record to be read
   (without any file extensions).
-  ``sampfrom`` (default=0) - The starting sample number to read for
   each channel.
-  ``sampto`` (default=length of entire signal)- The final sample number
   to read for each channel.
-  ``channels`` (default=all channels) - Indices specifying the channels
   to be returned.
-  ``physical`` (default=True) - Flag that specifies whether to return
   signals in physical (True) or digital (False) units.
-  ``m2s`` (default=True) - Flag used only for multi-segment
   records. Specifies whether to convert the returned wfdb.MultiRecord object
   into a wfdb.Record object (True) or not (False).

Output Arguments:

-  ``record`` - 

-  ``fields`` - A dictionary of metadata about the record extracted or
   deduced from the header/signal file. If the input record is a
   multi-segment record, the output argument will be a list of
   dictionaries:

   -  The first list element will be a dictionary of metadata about the
      master header.
   -  If the record is in variable layout format, the next list element
      will be a dictionary of metadata about the layout specification
      header.
   -  The last list element will be a list of dictionaries of metadata
      for each segment. For empty segments, the dictionary will be
      replaced by a single string: ‘Empty Segment’


Writing Signals
~~~~~~~~~~~~~~~

**wrsamp** - Read a WFDB file and return the signal as a numpy array and
the metadata as a dictionary.

::

    wrsamp(recordname, sampfrom=0, sampto=[], channels=[], physical=1, 
        stacksegments=1, pbdl=0, dldir=os.cwd())

Example Usage:

::

    import wfdb
    sig, fields = wfdb.wrsamp('mitdb/100', sampto=2000, pbdl=1)

Input Arguments:

-  ``recordname`` (mandatory) - The name of the WFDB record to be read
   (without any file extensions).
-  ``sampfrom`` (default=0) - The starting sample number to read for
   each channel.
-  ``sampto`` (default=length of entire signal)- The final sample number
   to read for each channel.
-  ``channels`` (default=all channels) - Indices specifying the channel
   to be returned.
-  ``physical`` (default=1) - Flag that specifies whether to return
   signals in physical (1) or digital (0) units.
-  ``stacksegments`` (default=1) - Flag used only for multi-segment
   files. Specifies whether to return the signal as a single
   stacked/concatenated numpy array (1) or as a list of one numpy array
   for each segment (0).
-  ``pbdl`` (default=0): If this argument is set, the function will
   assume that the user is trying to download a physiobank file.
   Therefore the ‘recordname’ argument will be interpreted as a
   physiobank record name including the database subdirectory, rather
   than a local directory.
-  ``dldir`` (default=os.getcwd()): The directory to download physiobank
   files to.

Output Arguments:

-  ``sig`` - An nxm numpy array where n is the signal length and m is
   the number of channels. If the input record is a multi-segment
   record, depending on the input stacksegments flag, sig will either be
   a single stacked/concatenated numpy array (1) or a list of one numpy
   array for each segment (0). For empty segments, stacked format will
   contain Nan values, and non-stacked format will contain a single
   integer specifying the length of the empty segment.
-  ``fields`` - A dictionary of metadata about the record extracted or
   deduced from the header/signal file. If the input record is a
   multi-segment record, the output argument will be a list of
   dictionaries:

   -  The first list element will be a dictionary of metadata about the
      master header.
   -  If the record is in variable layout format, the next list element
      will be a dictionary of metadata about the layout specification
      header.
   -  The last list element will be a list of dictionaries of metadata
      for each segment. For empty segments, the dictionary will be
      replaced by a single string: ‘Empty Segment’

Reading Annotations
~~~~~~~~~~~~~~~~~~~

**rdann** - Read a WFDB annotation file ``recordname.annot`` and return
the fields as lists or arrays.

::

    annsamp, anntype, subtype, chan, num, aux, annfs = wfdb.rdann(recordname, 
        annot, sampfrom=0, sampto=[], anndisp=1)

Example Usage:

::

    import wfdb
    annsamp, anntype, subtype, chan, num, aux, annfs = wfdb.rdann('100', 'atr')

Input Arguments:

-  ``recordname`` (required) - The record name of the WFDB annotation
   file. ie. for file ``100.atr`` recordname=‘100’.
-  ``annot`` (required) - The annotator extension of the annotation
   file. ie. for file ``100.atr`` annot=‘atr’.
-  ``sampfrom`` (default=0)- The minimum sample number for annotations
   to be returned.
-  ``sampto`` (default=the final annotation sample) - The maximum sample
   number for annotations to be returned.
-  ``anndisp`` (default=1) - The annotation display flag that controls
   the data type of the ``anntype`` output parameter. ``anntype`` will
   either be an integer key(0), a shorthand display symbol(1), or a
   longer annotation code(2).

Output arguments:

-  ``annsamp`` - The annotation location in samples relative to the
   beginning of the record.
-  ``anntype`` - The annotation type according the the standard WFDB
   keys.
-  ``subtype`` - The marked class/category of the annotation.
-  ``chan`` - The signal channel associated with the annotations.
-  ``num`` - The marked annotation number. This is not equal to the
   index of the current annotation.
-  ``aux`` - The auxiliary information string for the annotation.
-  ``annfs`` - The sampling frequency written in the beginning of the
   annotation file if present.

\*\ **NOTE**: Every annotation contains the ‘annsamp’ and ‘anntype’
field. All other fields default to 0 or empty if not present.


Plotting Data
~~~~~~~~~~~~~

**plotwfdb** - Subplot and label each channel of an nxm signal on a
graph. Also subplot annotation locations on selected channels if
present.

::

    plotwfdb(sig, fields, annsamp=[], annch=[0], title=[], plottime=1)

Example Usage:

::

    import wfdb
    sig, fields = wfdb.rdsamp('100')
    annsamp=wfdb.rdann('100', 'atr')[0]
    wfdb.plotwfdb(sig, fields, annsamp, 'mitdb record 100'): 
     

Input Arguments:

-  ``sig`` (required)- An nxm numpy array containing the signal to be
   plotted - the first output argument of ``wfdb.rdsamp``.
-  ``fields`` (required) - A dictionary of metadata about the record -
   the second output argument of ``wfdb.rdsamp``.
-  ``annsamp`` (optional) - A 1d numpy array of annotation locations to
   be plotted on top of selected channels - first output argument of
   ``rdann``.
-  ``annch`` (default=[0]) - A list of channels on which to plot the
   annotations.
-  ``title`` (optional)- A string containing the title of the graph.
-  ``plottime`` (default=1) - Flag that specifies whether to plot the x
   axis as time (1) or samples (0). Defaults to samples if the input
   ``fields`` dictionary does not contain a value for ``fs``.

Based on the original WFDB software package specifications
----------------------------------------------------------

| `WFDB Software Package`_
| `WFDB Applications Guide`_
| `WFDB Header File Specifications`_

.. _WFDB Software Package: http://physionet.org/physiotools/wfdb.shtml
.. _WFDB Applications Guide: http://physionet.org/physiotools/wag/
.. _WFDB Header File Specifications: https://physionet.org/physiotools/wag/header-5.htm


.. |Build Status| image:: https://travis-ci.org/MIT-LCP/wfdb-python.svg?branch=master
   :target: https://travis-ci.org/MIT-LCP/wfdb-python
