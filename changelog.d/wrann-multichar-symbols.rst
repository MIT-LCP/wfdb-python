**Write custom annotation symbols of more than one character**
  ``wrann`` and ``Annotation.wrann`` failed with a ``TypeError`` when a custom
  label's symbol had more than one character, and symbols were limited to 3
  characters. Symbols may now have any length (and still no whitespace), as
  in the WFDB library, and ``rdann`` reads them. A custom label is stored as
  a note of at most 255 Latin-1 characters ("<number> <symbol>
  <description>"), and ``wrann`` now raises a ``ValueError`` for a longer one,
  which it used to write as a file that could not be read.
