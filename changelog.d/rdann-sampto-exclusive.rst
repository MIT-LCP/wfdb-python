**rdann and Annotation.apply_range stop before sampto, as rdrecord does**
  ``rdann`` and ``Annotation.apply_range`` used to include an annotation at
  sample ``sampto``, while ``rdrecord`` and ``rdsamp`` return the samples
  before it. They now return the annotations before ``sampto``, so that the
  annotations read with a signal's range are within that signal, and can index
  it when read with ``shift_samps=True``. To include an annotation at a given
  sample, pass one more than that sample as ``sampto``.
