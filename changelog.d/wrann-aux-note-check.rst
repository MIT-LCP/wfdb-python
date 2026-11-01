**Reject aux notes that can't be written in wrann**
  ``wrann`` now raises a ``ValueError`` if an ``aux_note`` is longer than 255 characters or contains characters outside Latin-1, instead of writing a corrupt annotation file. A longer note used to be read back as many extra annotations, and other characters were silently replaced.
