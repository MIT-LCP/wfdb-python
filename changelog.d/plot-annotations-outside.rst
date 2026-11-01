**Plot annotations outside the signal**
  ``plot_items`` and ``plot_wfdb`` no longer fail with an ``IndexError`` when an annotation is after the end of the signal, and no longer draw annotations before the start of the signal at values from its end. These annotations are now drawn at zero, as for channels without a signal, with a warning suggesting ``shift_samps=True`` in ``rdann``.
