**Read headers whose base date is not a valid DD/MM/YYYY date**
  ``rdheader``, ``rdrecord`` and ``rdsamp`` raised a ``ValueError`` for a
  header whose base date is in another form, such as ``25.12.2020``,
  ``2020-12-25`` or just a year, so the record could not be read at all. As in
  the WFDB library, such a date is now ignored with a warning: ``base_date``
  and ``base_datetime`` are None, and the rest of the record is read as usual.
