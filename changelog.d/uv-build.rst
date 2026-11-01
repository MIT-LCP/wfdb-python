**Build with uv_build**
  The package is now built with ``uv_build`` instead of ``hatchling``. The version number is specified in ``pyproject.toml`` only, and ``wfdb.__version__`` is read from the installed package metadata (it is ``"unknown"`` if the package is run from a source tree without being installed).
