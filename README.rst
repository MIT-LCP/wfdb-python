wfdb-python
===========

|Build Status|

.. figure:: https://raw.githubusercontent.com/MIT-LCP/wfdb-python/master/demoimg.png
   :alt: wfdb signals

Introduction
------------

Native python scripts for reading and writing WFDB signals and annotations. Package to be expanded with other useful functionalities.


Installation
------------

The distribution is hosted on pypi and directly installable via pip without needing clone or download this repository. Note that the pypi package does not contain the demo scripts or the example data. To install the package from pypi, run from your terminal:
``pip install wfdb``

Download or clone this repository https://github.com/MIT-LCP/wfdb-python for the latest development version, the demo scripts, and the example data. To install the downloaded package, change directory into the base directory of the repository and run:
``pip install .``


Usage
-----

See the **demo.ipynb** file for example cases.

Objects
~~~~~~~

As of version 1.0.0, wfdb records are stored in **Record** or **MultiRecord** objects, and annotations are stored in **Annotation** objects. To see all attributes of an object, call `object.__dict__`


**Record** - The class representing WFDB headers, and single segment WFDB records.

Record objects can be created using the constructor, by reading a WFDB header
with 'rdheader', or a WFDB record (header and associated dat files) with rdsamp'
or 'srdsamp'.

The attributes of the Record object give information about the record as specified
by https://www.physionet.org/physiotools/wag/header-5.htm

In addition, the d_signals and p_signals attributes store the digital and physical
signals of WFDB records with at least one channel.

Contructor function:
::

    def __init__(self, p_signals=None, d_signals=None,
                 recordname=None, nsig=None,
                 fs=None, counterfreq=None, basecounter=None,
                 siglen=None, basetime=None, basedate=None,
                 filename=None, fmt=None, sampsperframe=None,
                 skew=None, byteoffset=None, adcgain=None,
                 baseline=None, units=None, adcres=None,
                 adczero=None, initvalue=None, checksum=None,
                 blocksize=None, signame=None, comments=None)

Example Usage:
::

    import wfdb
    record1 = wfdb.Record(recordname='r1', fs=250, nsig=2, siglen=1000, filename=['r1.dat','r1.dat'])


**MultiRecord** - The class representing multi-segment WFDB records.

MultiRecord objects can be created using the constructor, or by reading a multi-segment
WFDB record using 'rdsamp' with the 'm2s' (multi to single) input parameter set to False.

The attributes of the MultiRecord object give information about the entire record as specified
by https://www.physionet.org/physiotools/wag/header-5.htm

In addition, the 'segments' parameter is a list of Record objects representing each
individual segment, or 'None' representing empty segments, of the entire multi-segment record.

Noteably, this class has no attribute representing the signals as a whole. The 'multi_to_single'
instance method can be called on MultiRecord objects to return a single segment representation
of the record as a Record object. The resulting Record object will have its 'p_signals' field set.

Contructor function:
::

    def __init__(self, segments = None, layout = None,
                 recordname=None, nsig=None, fs=None,
                 counterfreq=None, basecounter=None,
                 siglen=None, basetime=None, basedate=None,
                 segname = None, seglen = None, comments=None)

Example Usage:
::

    import wfdb
    recordM = wfdb.MultiRecord(recordname='rm', fs=50, nsig=8, siglen=9999, segname=['rm_1', '~', rm_2'], seglen=[800, 200, 900])

    recordL = wfdb.rdsamp('s00001-2896-10-10-00-31', m2s = False)
    recordL = recordL.multi_to_single()


**Annotation** - The class representing WFDB annotations.

Annotation objects can be created using the constructor, or by reading a WFDB annotation
file with 'rdann'.

The attributes of the Annotation object give information about the annotation as specified
by https://www.physionet.org/physiotools/wag/annot-5.htm:
- ``annsamp``: The annotation location in samples relative to the beginning of the record.
- ``anntype``: The annotation type according the the standard WFDB codes.
- ``subtype``: The marked class/category of the annotation.
- ``chan``: The signal channel associated with the annotations.
- ``num``: The labelled annotation number.
- ``aux``: The auxiliary information string for the annotation.
- ``fs``: The sampling frequency of the record if contained in the annotation file.
- ``custom_anntypes``: The custom annotation types defined in the annotation file. A dictionary with {key:value} corresponding to {anntype:description}. eg. {'#': 'lost connection', 'C': 'reconnected'}

Constructor function:
::

    def __init__(self, recordname, annotator, annsamp, anntype, subtype = None,
                 chan = None, num = None, aux = None, fs = None, custom_anntypes = None)

Call `showanncodes()` to see the list of standard annotation codes. Any text used to label annotations that are not one of these codes should go in the 'aux' field rather than the 'anntype' field.

Example usage:
::

    import wfdb
    ann1 = wfdb.Annotation(recordname='ann1', annotator='atr', annsamp=[10,20,400],
                           anntype = ['N','N','['], aux=[None, None, 'Serious Vfib'])

Reading Signals
~~~~~~~~~~~~~~~


**rdsamp** - Read a WFDB record and return the signal and record descriptors as attributes in a Record or MultiRecord object.

::

    record = rdsamp(recordname, sampfrom=0, sampto=None, channels=None, physical=True, pbdir = None, m2s=True)

Example Usage:

::

    import wfdb
    ecgrecord = wfdb.rdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])

Input Arguments:

-  ``recordname`` (required): The name of the WFDB record to be read (without any file extensions).
-  ``sampfrom`` (default=0): The starting sample number to read for each channel.
-  ``sampto`` (default=length of entire signal)- The final sample number to read for each channel.
-  ``channels`` (default=all channels): Indices specifying the channels to be returned.
-  ``physical`` (default=True): Flag that specifies whether to return  signals in physical (True) or digital (False) units.
-  ``pbdir`` (default=None): Option used to stream data from Physiobank. The Physiobank database directory from which to find the required record files. eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb', pbdir = 'mitdb'.
-  ``m2s`` (default=True): Flag used only for multi-segment records. Specifies whether to convert the returned wfdb.MultiRecord object into a wfdb.Record object (True) or not (False).
-  ``smoothframes`` (default=True): Flag used when reading records with signals having multiple samples per frame. Specifies whether to smooth the samples in signals with more than one sample per frame and return an mxn uniform numpy array as the d_signals or p_signals field (True), or to return a list of 1d numpy arrays containing every expanded sample as the e_d_signals or e_p_signals field (False).
-  ``ignoreskew`` (default=False): Flag used when reading records with at least one skewed signal. Specifies whether to apply the skew to align the signals in the output variable (False), or to ignore the skew field and load in all values contained in the dat files unaligned (True).

Output Arguments:

-  ``record`` - The wfdb Record or MultiRecord object representing the contents of the record read.

**srdsamp** - A simplified wrapper function around rdsamp. Read a WFDB record and return the physical signal and a few important descriptor fields.

::

    signals, fields = srdsamp(recordname, sampfrom=0, sampto=None, channels=None, pbdir=None)

Example Usage:

::

    import wfdb
    sig, fields = wfdb.srdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])

Input arguments:

- ``recordname`` (required): The name of the WFDB record to be read (without any file extensions). If the argument contains any path delimiter characters, the argument will be interpreted as PATH/baserecord and the data files will be searched for in the local path.
- ``sampfrom`` (default=0): The starting sample number to read for each channel.
- ``sampto`` (default=None): The sample number at which to stop reading for each channel.
- ``channels`` (default=all): Indices specifying the channel to be returned.

Output arguments:

- ``signals``: A 2d numpy array storing the physical signals from the record.
- ``fields``: A dictionary specifying several key attributes of the read record:
    - ``fs``: The sampling frequency of the record
    - ``units``: The units for each channel
    - ``signame``: The signal name for each channel
    - ``comments``: Any comments written in the header


Writing Signals
~~~~~~~~~~~~~~~

The Record class has a **wrsamp** instance method for writing wfdb record files. Create a valid Record object and call ``record.wrsamp()``. If you choose this more advanced method, see also the `setdefaults`, `set_d_features`, and `set_p_features` instance methods to help populate attributes. In addition, there is also the following simpler module level **wrsamp** function.


**wrsamp** - Write a single segment WFDB record, creating a WFDB header file and any associated dat files.

::

    wrsamp(recordname, fs, units, signames, p_signals = None, d_signals=None, fmt = None, gain = None, baseline = None, comments = None)

Example Usage:

::

    import wfdb
    sig, fields = wfdb.srdsamp('a103l', sampfrom = 50000, channels = [0,1], pbdir = 'challenge/2015/training')
    wfdb.wrsamp('ecgrecord', fs = 250, units = ['mV', 'mV'], signames = ['I', 'II'], p_signals = sig, fmt = ['16', '16'])

Input Arguments:

- ``recordname`` (required): The string name of the WFDB record to be written (without any file extensions).
- ``fs`` (required): The numerical sampling frequency of the record.
- ``units`` (required): A list of strings giving the units of each signal channel.
- ``signames`` (required): A list of strings giving the signal name of each signal channel.
- ``p_signals`` (default=None): An MxN 2d numpy array, where M is the signal length. Gives the physical signal
  values intended to be written. Either p_signals or d_signals must be set, but not both. If p_signals
  is set, this method will use it to perform analogue-digital conversion, writing the resultant digital
  values to the dat file(s). If fmt is set, gain and baseline must be set or unset together. If fmt is
  unset, gain and baseline must both be unset.
- ``d_signals`` (default=None): An MxN 2d numpy array, where M is the signal length. Gives the digital signal
  values intended to be directly written to the dat file(s). The dtype must be an integer type. Either
  p_signals or d_signals must be set, but not both. In addition, if d_signals is set, fmt, gain and baseline
  must also all be set.
- ``fmt`` (default=None): A list of strings giving the WFDB format of each file used to store each channel.
  Accepted formats are: "80","212","16","24", and "32". There are other WFDB formats but this library
  will not write (though it will read) those file types.
- ``gain`` (default=None): A list of integers specifying the DAC/ADC gain.
- ``baseline`` (default=None): A list of integers specifying the digital baseline.
- ``comments`` (default-None): A list of string comments to be written to the header file.


Reading Annotations
~~~~~~~~~~~~~~~~~~~

**rdann** - Read a WFDB annotation file ``recordname.annot`` and return an Annotation object.

::

    annotation = rdann(recordname, annotator, sampfrom=0, sampto=None, shiftsamps=False, pbdir=None)

Example Usage:
::

    import wfdb
    ann = wfdb.rdann('sampledata/100', 'atr', sampto = 300000)

Input arguments:

- ``recordname`` (required): The record name of the WFDB annotation file. ie. for file `100.atr`, recordname='100'
- ``annotator`` (required): The annotator extension of the annotation file. ie. for
  file '100.atr', annotator='atr'
- ``sampfrom`` (default=0): The minimum sample number for annotations to be returned.
- ``sampto`` (default=None): The maximum sample number for annotations to be returned.
- ``shiftsamps`` (default=False): Boolean flag that specifies whether to return the sample indices relative to 'sampfrom' (True), or sample 0 (False). Annotation files store exact sample locations.
- ``pbdir`` (default=None): Option used to stream data from Physiobank. The Physiobank database directory from which to find the required annotation file. eg. For record '100' in 'http://physionet.org/physiobank/database/mitdb', pbdir = 'mitdb'.

Output arguments:

- ``annotation``: The Annotation object. Contains the following attributes:
    - ``annsamp``: The annotation location in samples relative to the beginning of the record.
    - ``anntype``: The annotation type according the the standard WFDB codes.
    - ``subtype``: The marked class/category of the annotation.
    - ``chan``: The signal channel associated with the annotations.
    - ``num``: The labelled annotation number.
    - ``aux``: The auxiliary information string for the annotation.
    - ``fs``: The sampling frequency of the record if contained in the annotation file.

\*\ **NOTE**: In annotation files, every annotation contains the ‘annsamp’ and ‘anntype’ field. All other fields default to 0 or empty if not present.

**showanncodes** -  Display the annotation symbols and the codes they represent according to the standard WFDB library 10.5.24

::

    showanncodes()

Writing Annotations
~~~~~~~~~~~~~~~~~~~

The Annotation class has a **wrann** instance method.

The Annotation class has a **wrann** instance method for writing wfdb annotation files. Create a valid Annotation object and call ``annotation.wrsamp()``. In addition, there is also the following simpler module level **wrann** function.

**wrann** - Write a WFDB annotation file.

::

    wrann(recordname, annotator, annsamp, anntype, num = None, subtype = None, chan = None, aux = None, fs = None)

Example Usage:

::

    import wfdb
    annotation = wfdb.rdann('b001', 'atr', pbdir='cebsdb')
    wfdb.wrann('b001', 'cpy', annotation.annsamp, annotation.anntype)

Input Arguments:

- ``recordname`` (required): The string name of the WFDB record to be written (without any file extensions).
- ``annotator`` (required): The string annotation file extension.
- ``annsamp`` (required): The annotation location in samples relative to the beginning of the record. List or numpy array.
- ``anntype`` (required): The annotation type according the the standard WFDB codes. List or numpy array.
- ``subtype`` (default=None): The marked class/category of the annotation. List or numpy array.
- ``chan`` (default=None): The signal channel associated with the annotations. List or numpy array.
- ``num`` (default=None): The labelled annotation number. List or numpy array.
- ``aux`` (default=None): The auxiliary information string for the annotation. List or numpy array.
- ``fs`` (default=None): The numerical sampling frequency of the record to be written to the file.

\*\ **NOTE**: Each annotation stored in a WFDB annotation file contains an annsamp and an anntype field. All other fields may or may not be present. Therefore in order to save space, when writing additional features such as 'aux' that are not present for every annotation, it is recommended to make the field a list, with empty indices set to None so that they are not written to the file.


Plotting Data
~~~~~~~~~~~~~

**plotrec** - Subplot and label each channel of a WFDB Record. Optionally, subplot annotation locations over selected channels.

::

    plotrec(record=None, title = None, annotation = None, timeunits='samples',
            sigstyle='', figsize=None, returnfig = False, ecggrids=[]):

Example Usage:

::

    import wfdb
    record = wfdb.rdsamp('sampledata/100', sampto = 3000)
    annotation = wfdb.rdann('sampledata/100', 'atr', sampto = 3000)

    wfdb.plotrec(record, annotation = annotation, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds', figsize = (10,4), ecggrids = 'all')

Input Arguments:

- ``record`` (required): A wfdb Record object. The p_signals attribute will be plotted.
- ``title`` (default=None): A string containing the title of the graph.
- ``annotation`` (default=None): A list of Annotation objects or numpy arrays. The locations of the Annotation objects' 'annsamp' attribute, or the locations of the numpy arrays' values, will be overlaid on the signals. The list index of the annotation item corresponds to the signal channel that each annotation set will be plotted on. For channels without annotations to plot, put None in the list. This argument may also be just an Annotation object or numpy array, which will be plotted over channel 0.
- ``timeunits`` (default='samples'): String specifying the x axis unit. Allowed options are: 'samples', 'seconds', 'minutes', and 'hours'.
- ``sigstyle`` (default=''): String, or list of strings, specifying the styling of the matplotlib plot for the signals. If 'sigstyle' is a string, each channel will have the same style. If it is a list, each channel's style will correspond to the list element. ie. sigtype=['r','b','k'].
- ``annstyle`` (default='r*'): String, or list of strings, specifying the styling of the matplotlib plot for the annotations. If 'annstyle' is a string, each channel will have the same style. If it is a list, each channel's style will correspond to the list element.
- ``figsize`` (default=None): Tuple pair specifying the width, and height of the figure. Same as the 'figsize' argument passed into matplotlib.pyplot's figure() function.
- ``returnfig`` (default=False): Specifies whether the figure is to be returned as an output argument
- ``ecggrids`` (default=[]): List of integers specifying channels in which to plot ecg grids. May be set to [] for no channels, or 'all' for all channels. Major grids at 0.5mV, and minor grids at 0.125mV. All channels to be plotted with grids must have units equal to 'uV', 'mV', or 'V'.

Output argument:
- ``figure``: The matplotlib figure generated. Only returned if the 'returnfig' option is set to True.


**plotann** - Plot sample locations of an Annotation object.

::

    plotann(annotation, title = None, timeunits = 'samples', returnfig = False)

Example Usage:

::

    import wfdb
    record = wfdb.rdsamp('sampledata/100', sampto = 15000)
    annotation = wfdb.rdann('sampledata/100', 'atr', sampto = 15000)

    wfdb.plotrec(record, annotation = annotation, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')


Input Arguments:

- ``annotation`` (required): An Annotation object. The annsamp attribute locations will be overlaid on the signal.
- ``title`` (default=None): A string containing the title of the graph.
- ``annotation`` (default=None): An Annotation object. The annsamp attribute locations will be overlaid on the signal.
- ``timeunits`` (default='samples'): String specifying the x axis unit. Allowed options are: 'samples', 'seconds', 'minutes', and 'hours'.
- ``returnfig`` (default=False): Specifies whether the figure is to be returned as an output argument

Output argument:
- ``figure``: The matplotlib figure generated. Only returned if the 'returnfig' option is set to True.

Downloading Physiobank Content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download files from various Physiobank databases. The Physiobank index page located at http://physionet.org/physiobank/database lists all available databases.


**getdblist** - Return a list of all the physiobank databases available.

::

    dblist = wfdb.getdblist()

Example Usage:

::

    import wfdb
    dblist = wfdb.getdblist()

**dldatabase** - Download WFDB record (and optionally annotation) files from a Physiobank database. The database must contain a 'RECORDS' file in its base directory which lists its WFDB records.

::

    dldatabase(pbdb, dlbasedir, records = 'all', annotators = 'all' , keepsubdirs = True, overwrite = False)

Example Usage:

::

    import wfdb
    wfdb.dldatabase('ahadb', os.getcwd())

Input arguments:

- ``pbdb`` (required): The Physiobank database directory to download. eg. For database 'http://physionet.org/physiobank/database/mitdb', pbdb = 'mitdb'.
- ``dlbasedir`` (required): The full local directory path in which to download the files.
- ``records`` (default='all'): Specifier of the WFDB records to download. Is either a list of strings which each specify a record, or 'all' to download all records listed in the database's RECORDS file. eg. records = ['test01_00s', test02_45s] for database https://physionet.org/physiobank/database/macecgdb/
- ``annotators`` (default='all'): Specifier of the WFDB annotation file types to download along with the record files. Is either None to skip downloading any annotations, 'all' to download all annotation types as specified by the ANNOTATORS file, or a list of strings which each specify an annotation extension. eg. annotators = ['anI'] for database https://physionet.org/physiobank/database/prcp/
- ``keepsubdirs`` (default=True): Whether to keep the relative subdirectories of downloaded files as they are organized in Physiobank (True), or to download all files into the same base directory (False).
- ``overwrite`` (default=False): If set to True, all files will be redownloaded regardless. If set to False, existing files with the same name and relative subdirectory will be checked. If the local file is the same size as the online file, the download is skipped. If the local file is larger, it will be deleted and the file will be redownloaded. If the local file is smaller, the file will be assumed to be partially downloaded and the remaining bytes will be downloaded and appended.


**dldatabasefiles** - Download specified files from a Physiobank database.

::

    dldatabasefiles(pbdb, dlbasedir, files, keepsubdirs = True, overwrite = False)

Example Usage:

::

    import wfdb
    wfdb.dldatabasefiles('ahadb', os.getcwd(), ['STAFF-Studies-bibliography-2016.pdf', 'data/001a.hea', 'data/001a.dat'])

Input arguments:

- ``pbdb`` (required): The Physiobank database directory to download. eg. For database 'http://physionet.org/physiobank/database/mitdb', pbdb = 'mitdb'.
- ``dlbasedir`` (required): The full local directory path in which to download the files.
- ``files`` (required): A list of strings specifying the file names to download relative to the database base directory
- ``keepsubdirs`` (default=True): Whether to keep the relative subdirectories of downloaded files as they are organized in Physiobank (True), or to download all files into the same base directory (False).
- ``overwrite`` (default=False): If set to True, all files will be redownloaded regardless. If set to False, existing files with the same name and relative subdirectory will be checked. If the local file is the same size as the online file, the download is skipped. If the local file is larger, it will be deleted and the file will be redownloaded. If the local file is smaller, the file will be assumed to be partially downloaded and the remaining bytes will be downloaded and appended.


Signal processing
-----------------

Basic functionalities
~~~~~~~~~~~~~~~~~~~~~

**resample_sig** - Resample a single-channel signal

::

    resample_sig(x, fs, fs_target)

Example Usage:

::

    import wfdb
    sig, fields = wfdb.srdsamp('sampledata/100', sampto=10000)
    x, _ = wfdb.processing.resample_sig(x=sig[:,0], fs=fields['fs'], fs_target=128)

Input arguments:

- ``x`` (required): The signal.
- ``fs`` (required): The signal frequency.
- ``fs_target`` (required): The target signal frequency.


**resample_singlechan** - Resample a single-channel signal and its annotation.

::

    resample_singlechan(x, ann, fs, fs_target)

Example Usage:

::

    import wfdb
    sig, fields = wfdb.srdsamp('sampledata/100')
    ann = wfdb.rdann('sampledata/100', 'atr')
    new_sig, new_ann = wfdb.processing.resample_singlechan(x=sig[:, 0], ann=ann, fs=fields['fs'], fs_target=50)

Input arguments:

- ``x`` (required): The signal.
- ``ann`` (required): The signal Annotation.
- ``fs`` (required): The signal frequency.
- ``fs_target`` (required): The target signal frequency.



**resample_multichan** - Resample a multi-channel signal and its annotation.

::

    resample_multichan(sig, ann, fs, fs_target)

Example Usage:

::

    import wfdb
    sig, fields = wfdb.srdsamp('sampledata/100')
    ann = wfdb.rdann('sampledata/100', 'atr')
    new_sig, new_ann = wfdb.processing.resample_multichan(sig=sig, ann=ann, fs=fields['fs'], fs_target=50)

Input arguments:

- ``x`` (required): The signal.
- ``ann`` (required): The signal Annotation.
- ``fs`` (required): The signal frequency.
- ``fs_target`` (required): The target signal frequency.



**normalize** - Resizes a signal between a lower and upper bound

::

    normalize(x, lb=0, ub=1)

Example Usage:

::

    import wfdb
    sig, _ = wfdb.srdsamp('sampledata/100')
    x = wfdb.processing.normalize(x=sig[:, 0], lb=-2, ub=15)

Input arguments:

- ``x`` (required): The signal.
- ``lb`` (required): The lower bound.
- ``ub`` (required): The upper bound.



**smooth** - Signal smoothing

::

    smooth(x, window_size)

Example Usage:

::

    import wfdb
    sig, _ = wfdb.srdsamp('sampledata/100')
    x = smooth(x=sig[:,0], window_size=150)

Input arguments:

- ``x`` (required): The signal.
- ``window_size`` (required): The smoothing window width.


Peak detection
~~~~~~~~~~~~~~

**gqrs_detect** - The GQRS detector function

::

  gqrs_detect(x, frequency, adcgain, adczero, threshold=1.0, hr=75, RRdelta=0.2, RRmin=0.28, RRmax=2.4, QS=0.07, QT=0.35, RTmin=0.25, RTmax=0.33, QRSa=750, QRSamin=130)

Example Usage:

::

    import wfdb
    t0 = 10000
    tf = 20000
    sig, fields = wfdb.srdsamp('sampledata/100', sampfrom=t0, sampto=tf, channels=[0])
    record = wfdb.rdsamp("sampledata/100", sampfrom=t0, sampto=tf, channels=[0], physical=False)
    peaks_indexes = wfdb.processing.gqrs_detect(x=sig[:,0], frequency=fields['fs'], adcgain=record.adcgain[0], adczero=record.adczero[0], threshold=1.0)

Input arguments:

- ``x`` (required): The signal.
- ``frequency`` (required): The signal frequency.
- ``adcgain`` (required): The gain of the signal (the number of adus (q.v.) per physical unit).
- ``adczero`` (required): The value produced by the ADC given a 0 volt input.
- ``threshold`` (default=1.0): The threshold for detection.
- ``hr`` (default=75): Typical heart rate, in beats per minute.
- ``RRdelta`` (default=0.2): Typical difference between successive RR intervals in seconds.
- ``RRmin`` (default=0.28): Minimum RR interval ("refractory period"), in seconds.
- ``RRmax`` (default=2.4): Maximum RR interval, in seconds; thresholds will be adjusted if no peaks are detected within this interval.
- ``QS`` (default=0.07): Typical QRS duration, in seconds.
- ``QT`` (default=0.35): Typical QT interval, in seconds.
- ``RTmin`` (default=0.25): Minimum interval between R and T peaks, in seconds.
- ``RTmax`` (default=0.33): Maximum interval between R and T peaks, in seconds.
- ``QRSa`` (default=750): Typical QRS peak-to-peak amplitude, in microvolts.
- ``QRSamin`` (default=130): Minimum QRS peak-to-peak amplitude, in microvolts.

Output Arguments:

- ``peaks_indexes``: A python list containing the peaks indexes.


**correct_peaks** - A post-processing algorithm to correct peaks position.

See code comments for details about the algorithm.


::

  correct_peaks(x, peaks_indexes, min_gap, max_gap, smooth_window)

Example Usage:

::

    import wfdb
    t0 = 10000
    tf = 20000
    sig, fields = wfdb.srdsamp('sampledata/100', sampfrom=t0, sampto=tf, channels=[0])
    record = wfdb.rdsamp("sampledata/100", sampfrom=t0, sampto=tf, channels=[0], physical=False)
    peak_indexes = wfdb.processing.gqrs_detect(x=sig[:,0], frequency=fields['fs'], adcgain=record.adcgain[0], adczero=record.adczero[0], threshold=1.0)
    fs = fields['fs']
    min_bpm = 10
    max_bpm = 350
    min_gap = fs*60/min_bpm
    max_gap = fs*60/max_bpm
    new_indexes = wfdb.processing.correct_peaks(x=sig[:,0], peak_indexes=peak_indexes, min_gap=min_gap, max_gap=max_gap, smooth_window=150)

Input arguments:
peaks_indexes, min_gap, max_gap, smooth_window
- ``x`` (required): The signal.
- ``peaks_indexes`` (required): The location of the peaks.
- ``min_gap`` (required): The minimum gap in samples between two peaks.
- ``max_gap`` (required): The maximum gap in samples between two peaks.
- ``smooth_window`` (required): The size of the smoothing window.

Output Arguments:

- ``new_indexes``: A python list containing the new peaks indexes.


Heart rate
~~~~~~~~~~~~~~

**compute_hr** - Compute heart rate from peak indexes and signal frequency.

::

  compute_hr(length, peaks_indexes, fs)

Example Usage:

::

    import wfdb
    t0 = 10000
    tf = 20000
    sig, fields = wfdb.srdsamp('sampledata/100', sampfrom=t0, sampto=tf, channels=[0])
    record = wfdb.rdsamp("sampledata/100", sampfrom=t0, sampto=tf, channels=[0], physical=False)
    peaks_indexes = wfdb.processing.gqrs_detect(x=sig[:,0], frequency=fields['fs'], adcgain=record.adcgain[0], adczero=record.adczero[0], threshold=1.0)
    hr = compute_hr(length=tf-t0, peaks_indexes=peaks_indexes, fs=fields['fs'])

Input arguments:

- ``length`` (required): The length of the corresponding signal.
- ``peaks_indexes`` (required): The peak indexes.
- ``fs`` (required): The signal frequency.


Output Arguments:

- ``hr``: A numpy.array containing heart rate for each sample. Contains numpy.nan where heart rate could not be computed.



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
