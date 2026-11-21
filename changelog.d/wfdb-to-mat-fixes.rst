**Fix wfdb_to_mat encoding, slicing, and header metadata**
  ``wfdb_to_mat`` now passes ``channels`` to ``rdrecord`` and computes output
  length from the sliced record. It writes 8-bit (offset binary) and 32-bit
  MAT files with proper data types and 8-byte MAT v5 padding, preserves missing
  samples, clears already-applied channel skew, handles unset ``adc_res`` and
  ``adc_zero`` fields, updates header checksums, and handles forward-slash
  record paths across platforms.
