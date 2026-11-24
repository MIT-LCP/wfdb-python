**Avoid an extra EDF data record at exact block boundaries**
  ``wfdb_to_edf`` no longer appends a full block of padding when the input
  signal length is an exact multiple of the EDF data record length.
