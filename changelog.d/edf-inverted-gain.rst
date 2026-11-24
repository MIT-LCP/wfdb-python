**Write EDF records with inverted physical ranges**
  ``read_edf`` now reflects the digital scale of inverted channels so the
  returned record has a positive WFDB gain and can be written with ``wrsamp``.
  The shared signal file uses format 32 only when a converted sample cannot
  be stored as valid format 16 data.
