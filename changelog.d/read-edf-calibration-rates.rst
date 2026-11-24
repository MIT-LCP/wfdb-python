**Correct EDF calibration and mixed-rate reads**
  ``read_edf`` no longer shifts physical values by one digital unit, and it
  reads files with differing per-signal sample counts consistently. EDF+
  annotation decoding uses the corrected values.
