**Write EDF records with inverted physical ranges**
  ``read_edf`` now reflects the digital scale of inverted channels so the
  returned record has a positive WFDB gain and can be written with ``wrsamp``.
  Such records use format 32 to preserve the full 16-bit EDF sample range.
