from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wfdb")
except PackageNotFoundError:
    # Running from a source tree that has not been installed
    __version__ = "unknown"
