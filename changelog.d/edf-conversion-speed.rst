**Speed up EDF conversion and correct mixed-rate scaling**
  EDF writing now assembles data by block without per-sample Python lists.
  EDF reading averages higher-rate channels with NumPy and applies their ADC
  gain correctly when sample rates differ.
