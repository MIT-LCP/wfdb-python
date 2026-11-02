**Accept sample numbers stored as floats in wrann**
  ``wrann`` now accepts a ``sample`` array of floats if all the values are whole numbers (as from a detector that returns floats), and writes them as integers. Other floats, such as ``20.5`` or NaN, raise a ``ValueError``. Float arrays used to fail with ``TypeError: ufunc 'bitwise_and' not supported for the input types``.
