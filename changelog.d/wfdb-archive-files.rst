**Read and write .wfdb archive files**
  A record can be stored in a single ``.wfdb`` archive file (a ZIP file containing the header and signal files). ``rdrecord``, ``rdsamp``, ``rdheader`` and ``rdann`` read a record or its annotations from an archive when given the path of the archive, and ``wrsamp`` writes a record into an archive with the new ``wfdb_archive`` argument.
