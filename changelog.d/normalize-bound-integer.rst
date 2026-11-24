**Avoid integer overflow when normalizing signal bounds**
  ``normalize_bound`` now computes the range of integer signals without
  wrapping, so large signed ranges map to the requested bounds.
