**Read headers whose base time or date is in another form**
  ``rdheader``, ``rdrecord`` and ``rdsamp`` raised a ``ValueError`` for a
  header whose base date is not a DD/MM/YYYY date (such as ``25.12.2020``)
  or whose base time is a year, so the record could not be read at all, and
  read a time and date in another order (such as ``05-Feb-2020 11:39:16``,
  in the 2020 and 2021 Challenges) as a wrong time. As in the WFDB library, a
  base time or date that is not valid is now ignored with a warning, and the
  rest of the record is read as usual.
