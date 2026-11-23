**Fix peak detection for integer signals**
  ``processing.find_peaks`` now compares integer samples without subtracting
  them first, avoiding overflow for signed samples and errors for unsigned
  samples. This also preserves differences between large 64-bit integers.
