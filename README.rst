wfdb-python
===========

|Build Status|

.. figure:: https://raw.githubusercontent.com/MIT-LCP/wfdb-python/master/demoimg.png
   :alt: wfdb signals

Introduction
------------

The native Python waveform-database (WFDB) package. A library of tools for reading and writing WFDB signals and annotations, and general physiological signal processing.

Core components of this package are based on the original WFDB specifications:

| `WFDB Software Package`_
| `WFDB Applications Guide`_
| `WFDB Header Specifications`_
| `WFDB Signal Specifications`_
| `WFDB Annotation Specifications`_

Installation
------------

The distribution is hosted on pypi at: https://pypi.python.org/pypi/wfdb/ and directly installable via pip without needing clone or download this repository. Note that the pypi package does not contain the demo scripts or the example data. To install the package from pypi, run from your terminal:
``pip install wfdb``

Download or clone this repository: https://github.com/MIT-LCP/wfdb-python for the latest development version, the demo scripts, and the example data. To install the downloaded package, navigate to the base directory of the repository and run:
``pip install .``


Documentation and Usage
-----------------------

See the documentation site for public classes and functions. 

See the **demo.ipynb** file for example use cases.






.. _WFDB Software Package: http://physionet.org/physiotools/wfdb.shtml
.. _WFDB Applications Guide: http://physionet.org/physiotools/wag/
.. _WFDB Header Specifications: https://physionet.org/physiotools/wag/header-5.htm
.. _WFDB Signal Specifications: https://physionet.org/physiotools/wag/signal-5.htm
.. _WFDB Annotation Specifications: https://physionet.org/physiotools/wag/annot-5.htm

.. |Build Status| image:: https://travis-ci.org/MIT-LCP/wfdb-python.svg?branch=master
   :target: https://travis-ci.org/MIT-LCP/wfdb-python
