Installation
------------

WFDB requires Python 3.11 or newer. Install the latest release from PyPI::

    pip install wfdb

To install from a checkout of the development repository, run::

    pip install .

See the `contributor guide
<https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md>`_ for development
setup and testing instructions.

FLAC files
----------

WFDB reads FLAC-compressed signal files using the ``soundfile`` package.
Platform-specific SoundFile wheels bundle ``libsndfile`` and ``libFLAC``.
Older bundled libFLAC versions can fail when seeking within some valid FLAC
files with ``Internal psf_fseek() failed``. Use SoundFile 0.13.1 or newer to
get updated bundled libraries.

If SoundFile uses a system-installed ``libsndfile`` instead, the system's
``libFLAC`` must be updated separately. This is the case on Linux systems with
glibc older than 2.28, for which SoundFile 0.13.1 has no wheel with bundled
libraries. Versions 1.4.0 through 1.4.2 have
a known seek failure; use libFLAC 1.4.3 or newer. See the reports for
`older libFLAC versions <https://github.com/MIT-LCP/wfdb-python/issues/486>`_
and `libFLAC 1.4.0--1.4.2 <https://github.com/MIT-LCP/wfdb-python/issues/488>`_.
