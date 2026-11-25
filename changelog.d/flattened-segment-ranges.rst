**Preserve source segment ranges on flattened records**
  ``rdrecord`` now retains source segment names and frame ranges, including
  empty segments, in the ``source_segments`` attribute when flattening a
  multi-segment record. Each range is a ``wfdb.io.SourceSegment`` named tuple
  of ``(name, start, stop)``.
