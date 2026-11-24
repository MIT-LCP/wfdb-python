**Iterate over long records as DataFrame chunks**
  ``wfdb.iter_dataframes`` reads one sample window at a time and yields
  pandas DataFrames with continuous time indexes, avoiding a full-record
  signal allocation. Ranged reads also work when the header omits the signal
  length.
