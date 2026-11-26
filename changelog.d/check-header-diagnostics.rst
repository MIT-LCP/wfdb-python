**Explain malformed header references in wfdb check**
  When a header still contains directory prefixes after a dataset is
  reorganized, ``wfdb check`` now identifies the affected lines and shows the
  record and signal filenames expected beside the header.
