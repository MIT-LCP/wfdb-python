**Fix -inf detection for expanded signals in calc_adc_params**
  ``calc_adc_params`` now raises the same "Signal contains inf" error for ``e_p_signal`` as it already did for ``p_signal``. The expanded-signal check only tested for ``+inf``, so a channel containing only ``-inf`` fell through and later failed with an unrelated "cannot convert float NaN to integer" error instead.
