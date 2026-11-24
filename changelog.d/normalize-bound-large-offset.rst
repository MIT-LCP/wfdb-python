**Preserve small differences in large integer signals when normalizing**
  ``normalize_bound`` now subtracts an integer signal's minimum exactly before
  scaling, so large offsets do not put results outside the requested bounds.
