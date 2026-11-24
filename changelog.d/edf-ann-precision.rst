**Preserve EDF+ annotation timing with mixed signal rates**
  ``rdedfann`` now uses the fastest physiologic signal rate for annotation
  sample numbers by default. The selected rate is returned as ``fs`` in the
  info dictionary and stored in written WFDB annotation files.
