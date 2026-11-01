**Fix correct_peaks for peaks near the start of the signal**
  ``correct_peaks`` no longer moves peaks within ``search_radius`` samples of the start of the signal to the wrong place, often before the start (for example ``-6`` instead of ``0``). The offset for these peaks was subtracted instead of added.
