**Allow None for annotations without notes in wrann**
  ``wrann`` now accepts ``None`` in ``aux_note`` for annotations without a note, as the docstring says, and writes it the same way as an empty string. It used to raise a ``TypeError``.
