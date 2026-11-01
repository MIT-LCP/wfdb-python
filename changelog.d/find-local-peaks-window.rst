**Include the right edge of the window in find_local_peaks**
  ``find_local_peaks`` now compares each sample with ``radius`` samples on both sides, as documented, and is about 3 times faster. The sample ``radius`` positions to the right, and the last sample of the signal, used to be left out. ``XQRS`` uses ``find_local_peaks`` to find candidate peaks, so its detections may change slightly.
