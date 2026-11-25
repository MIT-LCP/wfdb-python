**Preserve small differences in large integer local-peak inputs**
  ``processing.find_local_peaks`` uses its exact scanning path for integer
  values outside the range represented exactly by a double, avoiding false
  or missing peaks caused by rounding in the fast maximum filter.
