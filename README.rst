wfdb-python
===========

|Build Status|

.. figure:: https://raw.githubusercontent.com/MIT-LCP/wfdb-python/master/demo-img.png
   :alt: wfdb signals


Introduction
------------

The native Python waveform-database (WFDB) package. A library of tools for reading, writing, and processing WFDB signals and annotations.

Core components of this package are based on the original WFDB specifications. This package does not contain the exact same functionality as the original WFDB package. It aims to implement as many of its core features as possible, with user-friendly APIs. Additional useful physiological signal-processing tools are added over time.


Documentation and Usage
-----------------------

See the `documentation site`_ for the public APIs.

See the `demo.ipynb`_ notebook file for more example use cases.


Installation
------------

The distribution is hosted on pypi at: https://pypi.python.org/pypi/wfdb/. To directly install the package from pypi without needing to explicitly download content, run from your terminal::

    $ pip install wfdb

The development version is hosted at: https://github.com/MIT-LCP/wfdb-python. This repository also contains demo scripts and example data. To install the development version, clone or download the repository, navigate to the base directory, and run::

    $ pip install .


Development
-----------

The development repository is hosted at: https://github.com/MIT-LCP/wfdb-python

The package is to be expanded with physiological signal-processing tools, and general improvements. Development is made for Python 2.7 and 3.5+ only.


Contributing
------------

We welcome community contributions in the form of pull requests. When contributing code, please ensure:

* PEP8_ style guidelines are followed.
* Documentation is provided. New functions and classes should have numpy/scipy style docstrings_.
* Unit tests are written for new features that are not covered by `existing tests`_.


Authors
-------

`Chen Xie`_

`Julien Dubiel`_


.. |Build Status| image:: https://travis-ci.org/MIT-LCP/wfdb-python.svg?branch=master
   :target: https://travis-ci.org/MIT-LCP/wfdb-python

.. _documentation site: http://wfdb.readthedocs.io

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _docstrings: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _existing tests: https://github.com/MIT-LCP/wfdb-python/tree/master/tests

.. _demo.ipynb: https://github.com/MIT-LCP/wfdb-python/blob/master/demo.ipynb

.. _Chen Xie: https://github.com/cx1111/
.. _Julien Dubiel: https://github.com/Dubrzr/
