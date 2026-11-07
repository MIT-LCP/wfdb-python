**Speed up reading annotation files**
  ``rdann`` now decodes annotation files using plain Python integers instead of indexing a NumPy array one byte at a time. Reading annotations is about 2 to 3 times faster (a 100,000-beat file now takes about 75 ms instead of 175). The annotations read are unchanged.
