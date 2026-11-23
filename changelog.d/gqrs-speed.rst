**Speed up gqrs_detect**
  ``gqrs_detect`` now computes its smoothing and QRS matched filters for the whole signal with NumPy, instead of one sample at a time in Python, and is about 4 times faster (0.36 seconds instead of 1.4 for the 30-minute MIT-BIH record 100). The detected QRS locations are unchanged.
