**Avoid overflow with NumPy integer normalization bounds**
  ``normalize_bound`` now widens NumPy integer bounds before subtracting them,
  preventing overflow when their difference exceeds the bounds' dtype.
