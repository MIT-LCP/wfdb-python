**Write custom annotation symbols of more than one character**
  ``wrann`` and ``Annotation.wrann`` failed with a ``TypeError`` when a custom
  label's symbol had 2 or 3 characters, which the symbol field allows and
  ``rdann`` reads. They now write such symbols.
