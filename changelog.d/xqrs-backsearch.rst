**Speed up XQRS backsearch**
  ``XQRS`` no longer rechecks peaks that an earlier backsearch already rejected. Long stretches without QRS complexes (for example a disconnected lead) made detection take quadratic time, up to hours on long records. The detected QRS locations are unchanged.
