**Check that QRS indices are increasing in compute_hr**
  ``compute_hr`` now raises a ``ValueError`` if ``qrs_inds`` is not strictly increasing. Repeated beats used to give a ``ZeroDivisionError`` (or ``inf`` for NumPy arrays), and beats out of order gave a negative heart rate.
