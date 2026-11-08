**Fix writing channels that are entirely NaN**
  ``wrsamp`` and ``calc_adc_params`` no longer fail with ``cannot convert float NaN to integer`` when a channel contains only NaN values, and no longer emit an "All-NaN slice encountered" warning for such channels.
