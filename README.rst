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



Converting between Analog and Digital Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When reading signal sample values into ``record`` objects using ``rdsamp``, the samples are stored in either the ``p_signals`` or the ``d_signal`` field depending on the specified return type (``physical`` = ``True`` or ``False`` respectively).

One can also use existing objects to obtain physical values from digital values and vice versa, without having to re-read the wfdb file with a different set of options. The two following instance methods perform the conversions.


**adc** - Performs analogue to digital conversion of the physical signal stored in p_signals if expanded is False, or e_p_signals if expanded is True. The p_signals/e_p_signals, fmt, gain, and baseline fields must all be valid. If inplace is True, the adc will be performed inplace on the variable, the d_signal/e_d_signal attribute will be set, and the p_signals/e_p_signals field will be set to None.

::

    record.adc(self, expanded=False, inplace=False)

Input arguments:

- ``expanded`` (default=False): Boolean specifying whether to transform the e_p_signals attribute (True) or the p_signals attribute (False).
- ``inplace`` (default=False): Boolean specifying whether to automatically set the object's corresponding digital signal attribute and set the physical signal attribute to None (True), or to return the converted signal as a separate variable without changing the original physical signal attribute (False).

Possible output argument:

- ``d_signal``: The digital conversion of the signal. Either a 2d numpy array or a list of 1d numpy arrays.

Example Usage:
        
::

  import wfdb
  record = wfdb.rdsamp('sampledata/100')
  d_signal = record.adc()
  record.adc(inplace=True)
  record.dac(inplace=True)


**dac** - Performs digital to analogue conversion of the digital signal stored in d_signal if expanded is False, or e_d_signal if expanded is True. The d_signal/e_d_signal, fmt, gain, and baseline fields must all be valid. If inplace is True, the dac will be performed inplace on the variable, the p_signals/e_p_signals attribute will be set, and the d_signal/e_d_signal field will be set to None.

::

    record.dac(self, expanded=False, inplace=False)

Input arguments:

- ``expanded`` (default=False): Boolean specifying whether to transform the e_d_signal attribute (True) or the d_signal attribute (False).
- ``inplace`` (default=False): Boolean specifying whether to automatically set the object's corresponding physical signal attribute and set the digital signal attribute to None (True), or to return the converted signal as a separate variable without changing the original digital signal attribute (False).

Possible output argument:

- ``p_signals``: The physical conversion of the signal. Either a 2d numpy array or a list of 1d numpy arrays.

Example Usage:
        
::

  import wfdb
  record = wfdb.rdsamp('sampledata/100', physical=False)
  p_signal = record.dac()
  record.dac(inplace=True)
  record.adc(inplace=True)
















Signal Processing
-----------------

Basic Functionalities
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


Peak Detection
~~~~~~~~~~~~~~

**gqrs_detect** - The GQRS detector function

::

  gqrs_detect(x, fs, adcgain, adczero, threshold=1.0, hr=75, RRdelta=0.2, 
              RRmin=0.28, RRmax=2.4, QS=0.07, QT=0.35, RTmin=0.25, RTmax=0.33,
              QRSa=750, QRSamin=130):

Example Usage:

::

    import wfdb
    t0 = 10000
    tf = 20000
    record = wfdb.rdsamp("sampledata/100", sampfrom=t0, sampto=tf, channels=[0])
    d_signal = record.adc()[:,0]
    peak_indices = wfdb.processing.gqrs_detect(x=d_signal, fs=record.fs, adcgain=record.adcgain[0], adczero=record.adczero[0], threshold=1.0)

Input arguments:

- ``x`` (required): The digital signal as a numpy array
- ``fs`` (required): The sampling frequency of the signal
- ``adcgain``: The gain of the signal (the number of adus (q.v.) per physical unit)
- ``adczero`` (required): The value produced by the ADC given a 0 volt input.
- ``threshold`` (default=1.0): The threshold for detection
- ``hr`` (default=75): Typical heart rate, in beats per minute
- ``RRdelta`` (default=0.2): Typical difference between successive RR intervals in seconds
- ``RRmin`` (default=0.28): Minimum RR interval ("refractory period"), in seconds
- ``RRmax`` (default=2.4): Maximum RR interval, in seconds; thresholds will be adjusted if no peaks are detected within this interval
- ``QS`` (default=0.07): Typical QRS duration, in seconds
- ``QT`` (default=0.35): Typical QT interval, in seconds
- ``RTmin`` (default=0.25): Minimum interval between R and T peaks, in seconds
- ``RTmax`` (default=0.33): Maximum interval between R and T peaks, in seconds
- ``QRSa`` (default=750): Typical QRS peak-to-peak amplitude, in microvolts
- ``QRSamin`` (default=130): Minimum QRS peak-to-peak amplitude, in microvolts

Output Arguments:

- ``peak_indices``: A python list containing the peak indices.


**correct_peaks** - A post-processing algorithm to correct peaks position.

See code comments for details about the algorithm.


::

  correct_peaks(x, peak_indices, min_gap, max_gap, smooth_window)

Input arguments:

- ``x`` (required): The signal.
- ``peak_indices`` (required): The location of the peaks.
- ``min_gap`` (required): The minimum gap in samples between two peaks.
- ``max_gap`` (required): The maximum gap in samples between two peaks.
- ``smooth_window`` (required): The size of the smoothing window.

Output Arguments:

- ``new_indices``: A python list containing the new peaks indices.


Example Usage:

::

    import wfdb
    t0 = 10000
    tf = 20000
    record = wfdb.rdsamp('sampledata/100', sampfrom=t0, sampto=tf, channels=[0])
    d_signal = record.adc()[:,0]
    peak_indices = wfdb.processing.gqrs_detect(d_signal, fs=record.fs, 
                                               adcgain=record.adcgain[0], 
                                               adczero=record.adczero[0],
                                               threshold=1.0)
    min_bpm = 10
    max_bpm = 350
    min_gap = record.fs*60/min_bpm
    max_gap = record.fs*60/max_bpm
    new_indices = wfdb.processing.correct_peaks(d_signal, peak_indices=peak_indices,
                                                min_gap=min_gap, max_gap=max_gap, 
                                                smooth_window=150)


Heart Rate
~~~~~~~~~~~~~~

**compute_hr** - Compute instantaneous heart rate from peak indices and signal frequency.

::

  compute_hr(siglen, peak_indices, fs)

Input arguments:

- ``siglen`` (required): The length of the corresponding signal.
- ``peak_indices`` (required): The peak indices.
- ``fs`` (required): The corresponding signal's sampling frequency.


Output Arguments:

- ``hr``: A numpy array of the instantaneous heart rate, with the length of the corresponding signal. Contains numpy.nan where heart rate could not be computed.


Example Usage:

::

    import wfdb
    t0 = 10000
    tf = 20000
    record = wfdb.rdsamp("sampledata/100", sampfrom=t0, sampto=tf, channels=[0])
    peak_indices = wfdb.processing.gqrs_detect(record.adc(), fs=record.fs,
                                               adcgain=record.adcgain[0],
                                               adczero=record.adczero[0],
                                               threshold=1.0)
    hr = wfdb.processing.compute_hr(siglen=tf-t0, peak_indices=peak_indices, fs=record.fs)




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
