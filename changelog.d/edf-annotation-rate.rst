**Exclude EDF+ annotation channels from signal rates**
  ``read_edf`` now calculates sampling frequency and signal frames from
  physiologic channels only. ``rdedfann`` returns no sample numbers for
  annotation-only files unless an ``fs`` value is supplied.
