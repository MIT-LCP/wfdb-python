**Correct EDF calibration and mixed-rate reads**
  ``read_edf`` no longer shifts physical values by one digital unit, and it
  reads files with differing per-signal sample counts consistently. EDF+
  annotation decoding uses the corrected values. Physical samples can differ
  from the WFDB C ``edf2mit`` output by up to one digital unit when the EDF
  offset is fractional, because its baseline is rounded to an integer.
