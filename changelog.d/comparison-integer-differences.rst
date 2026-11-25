**Avoid integer overflow when comparing annotation samples**
  ``processing.compare_annotations`` and ``Comparitor`` widen NumPy integer
  sample positions before subtracting them, so unsigned positions can match
  nearby later detections and large signed differences do not wrap.
