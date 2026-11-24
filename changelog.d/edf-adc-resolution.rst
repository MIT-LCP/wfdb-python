**Correct EDF ADC resolution metadata**
  ``read_edf`` now reports the bit width of the inclusive digital range, so
  full 12-bit and 16-bit EDF channels report ``adc_res`` 12 and 16.
