**Avoid wrapping waveform averages before the record starts**
  ``processing.sigavg`` pads samples before the start of a record with zero,
  matching its existing end-of-record padding, instead of including unrelated
  samples from the end of the record through negative NumPy indices.
