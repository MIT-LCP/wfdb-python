**Fix wfdb_to_wav failing for every record**
  ``wfdb_to_wav`` now reads the record with ``smooth_frames=True``, since a .wav file holds one sample per frame for each signal. It used to read the expanded samples, and then failed with an ``AttributeError`` (a ``TypeError`` in older versions) because ``p_signal`` was not set.
