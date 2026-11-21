**Raise errors when rdann reads fs from the header**
  When an annotation file doesn't store the sampling frequency, ``rdann`` reads it from the record's header. It used to ignore any error from that, for example when the header couldn't be downloaded from PhysioNet, and return the annotation with ``fs`` set to ``None``. Now only a missing header is ignored, and other errors are raised.
