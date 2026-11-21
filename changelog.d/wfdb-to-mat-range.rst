**Preserve shifted samples in wfdb_to_mat**
  ``wfdb_to_mat`` now chooses a wider output type when samples shifted by
  ``adc_zero`` do not fit the type suggested by the header. It raises an error
  instead of silently wrapping samples that cannot fit in a 32-bit MAT file.
