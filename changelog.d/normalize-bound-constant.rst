**Normalize constant signals to the middle of the bounds**
  ``normalize_bound`` raised a ``ZeroDivisionError`` for a constant integer
  signal, and returned NaN (with warnings) for a constant float signal. Both
  are now normalized to the middle of the bounds.
