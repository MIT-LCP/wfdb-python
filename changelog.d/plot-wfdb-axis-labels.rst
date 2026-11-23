**Set axis labels in plot_wfdb, and fix xlabel in plot_items**
  ``plot_wfdb`` takes ``xlabel`` and ``ylabel`` arguments, lists with a label
  for each channel, to replace the default labels (the time unit, and the
  signal names and units). ``plot_items`` failed with an ``AttributeError``
  whenever ``xlabel`` was given, and now sets the labels.
