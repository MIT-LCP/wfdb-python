**Fix search window in correct_peaks**
  ``correct_peaks`` now searches ``search_radius`` samples on both sides of each peak, including the last sample of the signal. It used to stop one sample short on the right, so a peak exactly ``search_radius`` samples after the original index, or on the last sample, could not be found.
