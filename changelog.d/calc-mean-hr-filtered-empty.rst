**Return 0 from calc_mean_hr when all intervals are filtered out**
  ``calc_mean_hr`` now returns 0 when ``min_rr`` or ``max_rr`` removes every R-R interval, as it does for empty input. It used to return ``nan`` with a "Mean of empty slice" warning.
