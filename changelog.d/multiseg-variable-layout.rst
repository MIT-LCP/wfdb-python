**Write variable layout multi-segment records**
  ``MultiRecord.wrsamp`` can now write variable layout multi-segment records, such as those read with ``rdrecord(..., m2s=False)``. Previously, writing the layout header failed because its signals have no samples (format ``0``, file name ``~``).
