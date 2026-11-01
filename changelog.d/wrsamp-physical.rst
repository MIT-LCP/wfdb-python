**Fix writing physical-only records**
  ``Record.wrsamp`` now temporarily converts physical signals to digital samples
  when the corresponding digital signals are missing, then restores the record.
