**Read records from memory**
  ``rdrecord``, ``rdsamp`` and ``rdheader`` now accept ``memory://`` URIs, which read from the in-memory filesystem of ``fsspec``, as ``rdann`` already did. This also makes cloud URIs work for signal files and multi-segment records on Windows, where the file paths used to be built with backslashes.
