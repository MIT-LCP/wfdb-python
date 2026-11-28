**Handle unsigned and narrow integer guesses in peak correction**
  ``processing.correct_peaks`` widens integer peak indices before window
  arithmetic, avoiding wraparound near signal boundaries and keeping corrected
  ``uint64`` indices usable for NumPy indexing.
