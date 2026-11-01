**Speed up the digital range check when writing records**
  ``wrsamp`` now uses NumPy to check that digital samples are within the range of each format, instead of Python's built-in ``min`` and ``max``. Writing a 4-hour, 8-channel record at 500 Hz in format 16 now takes about 0.5 seconds instead of 3. The written files are unchanged.
