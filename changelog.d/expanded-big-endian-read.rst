**Read format 61 signals with smooth_frames=False**
  ``rdrecord`` with ``physical=False`` and ``smooth_frames=False`` failed with ``ValueError: invalid literal for int()`` for records with format 61 signals, which are stored in big-endian byte order. These signals are now returned with the requested native dtype.
