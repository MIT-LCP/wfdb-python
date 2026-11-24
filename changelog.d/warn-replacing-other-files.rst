**Warn when a renamed record would replace another record's signal files**
  ``Record.wrsamp`` names the header after ``record_name`` and the signal
  files by ``file_name``, so writing a record that was read under a new
  ``record_name``, next to the original, replaced the original's signal
  files without any sign. It now warns when writing would replace an existing
  signal file whose name does not start with ``record_name``. The
  documentation of ``wrsamp`` and ``file_name`` now explains this, with an
  example of saving a copy under a new name.
