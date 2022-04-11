# The WFDB Python Package

![signals](https://raw.githubusercontent.com/MIT-LCP/wfdb-python/master/demo-img.png)

[![tests workflow](https://github.com/MIT-LCP/wfdb-python/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MIT-LCP/wfdb-python/actions?query=workflow%3Arun-tests+event%3Apush+branch%3Amaster)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wfdb.svg?label=PyPI%20downloads)](https://pypi.org/project/wfdb/)
[![PhysioNet Project](https://img.shields.io/badge/DOI-10.13026%2Fegpf--2788-blue)](https://doi.org/10.13026/egpf-2788)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/wfdb.svg)](https://pypi.org/project/wfdb)

## Introduction

A Python-native package for reading, writing, processing, and plotting physiologic signal and annotation data. The core I/O functionality is based on the Waveform Database (WFDB) [specifications](https://github.com/wfdb/wfdb-spec/).

This package is heavily inspired by the original [WFDB Software Package](https://www.physionet.org/content/wfdb/), and initially aimed to replicate many of its command-line APIs. However, the projects are independent, and there is no promise of consistency between the two, beyond each package adhering to the core specifications.

## Documentation and Usage

See the [documentation site](http://wfdb.readthedocs.io) for the public APIs.

See the [demo.ipynb](https://github.com/MIT-LCP/wfdb-python/blob/master/demo.ipynb) notebook file for example use cases.

## Installation

The distribution is hosted on pypi at: <https://pypi.python.org/pypi/wfdb/>. To directly install the package from pypi, run from your terminal::

```sh
pip install wfdb
```

The development version is hosted at: <https://github.com/MIT-LCP/wfdb-python>. This repository also contains demo scripts and example data. To install the development version, clone or download the repository, navigate to the base directory, and run:

```sh
pip install .
```

## Development

The package is to be expanded with physiological signal-processing tools, and general improvements. Development is made for Python 3.6+ only.

## Contributing

We welcome community contributions in the form of pull requests. When contributing code, please ensure:

- [PEP8](https://www.python.org/dev/peps/pep-0008/) style guidelines are followed.
- Documentation is provided. New functions and classes should have numpy/scipy style [docstrings](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
- Unit tests are written for new features that are not covered by [existing tests](https://github.com/MIT-LCP/wfdb-python/tree/master/tests).

## Citing

When using this resource, please cite the software [publication](https://physionet.org/content/wfdb-python/) oh PhysioNet.
