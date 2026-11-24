**Read EDF and WAV files from physionet:// and other URIs**
  ``read_edf``, ``rdedfann`` and ``read_wav`` now take a cloud URI as
  ``record_name``, such as ``physionet://chbmit/1.0.0/chb01/chb01_01.edf``,
  ``s3://`` or ``memory://``, as ``rdrecord`` and ``rdann`` do, and read the
  file without saving a copy. All of the functions that take ``pn_dir`` now
  also take a ``physionet://`` URI. Output files written by ``wfdb_to_edf``
  and ``wfdb_to_wav`` for a URI are named after the record, on every platform.
