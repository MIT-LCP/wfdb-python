**Use the last RR interval in compute_hr**
  ``compute_hr`` now uses the interval between the last two beats for the heart rate after the last beat, instead of repeating the previous rate. With exactly two beats it now returns their rate after the second beat instead of only NaN.
