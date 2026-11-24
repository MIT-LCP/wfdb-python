**Read digital variable-layout records with absent channels**
  ``rdrecord`` and ``iter_dataframes`` now handle windows where a requested
  channel has no samples in any selected segment. Those channels are filled
  with a missing-value marker instead of raising ``KeyError``.
