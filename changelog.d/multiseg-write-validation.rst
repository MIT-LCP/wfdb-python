**Validate multi-segment record writes**
  ``MultiRecord.wrsamp`` now checks the segment lengths and sampling frequencies of all segments before writing any files, instead of writing a segment whose length does not match ``seg_len``.
