wfdb
====

Introduction
------------

The native Python waveform-database (WFDB) package. A library of tools
for reading, writing, and processing WFDB signals and annotations.

Core components of this package are based on the original WFDB
specifications. This package does not contain the exact same
functionality as the original WFDB package. It aims to implement as many
of its core features as possible, with user-friendly APIs. Additional
useful physiological signal-processing tools are added over time.


Development
-----------

The development repository is hosted at: https://github.com/MIT-LCP/wfdb-python

The package is to be expanded with physiological signal-processing tools, and general improvements. Development is made for Python 3.6+ only.


API Reference
--------------

The exact API of all accessible functions and classes, as given by the docstrings, grouped by subpackage:

.. toctree::
   :maxdepth: 2

   io
   plot
   processing


Core Components
---------------

A subset of the above components are accessible by directly importing the base package.

.. toctree::
   :maxdepth: 2

   wfdb


Other Content
-------------
.. toctree::
   :maxdepth: 2

   installation
   wfdb-specifications


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _docstrings: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _existing tests: https://github.com/MIT-LCP/wfdb-python/tree/master/tests
