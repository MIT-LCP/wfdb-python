**Add an immutable header reader with published type information**
  ``read_header`` returns an immutable single- or multi-segment header with
  per-signal fields or segment references. It reads one physical header per
  call. ``rdheader`` retains its existing return types and recursive option.
