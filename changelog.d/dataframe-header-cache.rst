**Reuse headers while iterating over DataFrame chunks**
  ``wfdb.iter_dataframes`` now reads each record and segment header only once
  per iterator, reducing requests for remote records.
