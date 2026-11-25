**Write flattened multi-segment records**
  Records returned by ``rdrecord`` with ``m2s=True`` can now be written after
  assigning a new record name and signal file names. Missing write metadata is
  filled in, and incompatible or unknown formats are converted to format 32.
  The caller's record is not changed. Records with conflicting source units
  require an explicit ``units`` value.
