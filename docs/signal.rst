
Converting between Analog and Digital Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When reading signal sample values into ``record`` objects using ``rdrecord``, the samples are stored in either the ``p_signals`` or the ``d_signal`` field depending on the specified return type (``physical``=``True`` or ``False`` respectively).

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
