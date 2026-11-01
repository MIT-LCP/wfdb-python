**Fix rr2ann with times and with a symbol column**
  ``rr2ann`` with ``as_time=True`` now rounds times to the nearest sample instead of truncating them, so (for example) 0.29 seconds at 100 Hz gives sample 29 instead of 28. Intervals in samples with a second column of symbols no longer raise an error, and fractional intervals are summed before rounding.
