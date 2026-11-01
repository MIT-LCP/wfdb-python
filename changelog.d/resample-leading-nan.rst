**Fix resample_sig for signals starting with NaN**
  ``resample_sig`` (and the functions that use it, such as ``resample_singlechan`` and ``resample_multichan``) returned only NaN values if the signal started with NaN. Leading NaN values now take the first valid value before resampling, as trailing ones already took the last.
