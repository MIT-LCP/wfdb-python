**Exclude EDF+ annotation channels from signal rates**
  ``read_edf`` now calculates sampling frequency and signal frames from
  physiologic channels only, and no longer includes ``EDF Annotations`` in
  ``sig_name`` or ``p_signal``. Annotation-only files have no signal rate:
  ``rdedfann`` returns ``None`` sample numbers unless ``fs`` is supplied, and
  writing a WFDB annotation from one now requires ``fs``.
