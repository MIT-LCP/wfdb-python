**Accept pathlib.Path record names in rdann**
  ``rdann`` now accepts a ``pathlib.Path`` (or other path-like object) as ``record_name``, as ``rdrecord``, ``rdsamp`` and ``rdheader`` already do. It used to fail with ``TypeError: unsupported operand type(s) for +``.
