**Make records read from EDF writable**
  ``read_edf`` now stores the computed baseline as a list, so its returned
  record can be written with ``Record.wrsamp``.
