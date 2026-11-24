**Leave no partly written multi-segment record after an error**
  ``MultiRecord.wrsamp`` wrote the main header and the earlier segments
  before finding that a later segment could not be written, which left a
  partly written record. It now writes the files into a temporary folder in
  ``write_dir``, and moves them into place only once every segment has been
  written. Files that are replaced keep their permissions, and if a move
  fails, the files already moved are removed and the replaced files are
  restored. The files are moved one at a time, so a program reading them
  meanwhile, or a crash while they are moved, can still see a mix of files.
  A segment without the signal to write, or without
  ``samps_per_frame`` when writing expanded signals, now raises a
  ``ValueError`` that names the segment, instead of a ``TypeError``. A
  relative ``wfdb_archive`` path is now relative to ``write_dir``, as it is
  for ``Record.wrsamp``.
