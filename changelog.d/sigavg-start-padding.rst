**Pad waveform averages with zero in physical units**
  ``processing.sigavg`` no longer includes unrelated end-of-record samples
  through negative NumPy indices. Out-of-range positions at either end use
  the channel's digital baseline so padding is zero after calibration,
  including records with a nonzero baseline.
