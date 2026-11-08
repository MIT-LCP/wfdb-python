**Reject negative infinity in expanded signals**
  ``calc_adc_params`` now raises "Signal contains inf" when ``e_p_signal`` contains ``-inf``, as it already did for ``p_signal``, instead of failing with ``cannot convert float NaN to integer``.
