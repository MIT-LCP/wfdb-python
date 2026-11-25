**Evaluate filter gain at the requested frequency**
  ``processing.get_filter_gain`` no longer rounds up to a fixed frequency grid.
  This also fixes an ``IndexError`` at the Nyquist frequency and frequencies
  just below it.
