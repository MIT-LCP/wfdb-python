**Read units containing any non-whitespace characters**
  Header files with units containing characters such as ``.`` or ``*`` (for example ``a.u.`` or ``uV*s``) are now read correctly. The units used to be cut off at the first such character, and the rest of the signal line (ADC resolution, ADC zero and so on) ended up in the signal name.
