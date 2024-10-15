# The WFDB Python Package

![signals](https://raw.githubusercontent.com/MIT-LCP/wfdb-python/main/demo-img.png)

[![tests workflow](https://github.com/MIT-LCP/wfdb-python/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MIT-LCP/wfdb-python/actions?query=workflow%3Arun-tests+event%3Apush+branch%3Amain)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wfdb.svg?label=PyPI%20downloads)](https://pypi.org/project/wfdb/)
[![PhysioNet Project](https://img.shields.io/badge/DOI-10.13026%2Fegpf--2788-blue)](https://doi.org/10.13026/egpf-2788)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/wfdb.svg)](https://pypi.org/project/wfdb)

## Introduction

A Python-native package for reading, writing, processing, and plotting physiologic signal and annotation data. The core I/O functionality is based on the Waveform Database (WFDB) [specifications](https://github.com/wfdb/wfdb-spec/).

This package is heavily inspired by the original [WFDB Software Package](https://www.physionet.org/content/wfdb/), and initially aimed to replicate many of its command-line APIs. However, the projects are independent, and there is no promise of consistency between the two, beyond each package adhering to the core specifications.

## Documentation and Usage

See the [documentation site](http://wfdb.readthedocs.io) for the public APIs.

See the [demo.ipynb](https://github.com/MIT-LCP/wfdb-python/blob/main/demo.ipynb) notebook file for example use cases.

## Installation

The distribution is hosted on PyPI at: <https://pypi.python.org/pypi/wfdb/>. The package can be directly installed from PyPI using pip:

```sh
pip install wfdb
```

On some less-common systems, you may need to install `libsndfile` separately.  See the [soundfile installation notes](https://pypi.org/project/soundfile/) for more information.

The development version is hosted at: <https://github.com/MIT-LCP/wfdb-python>. This repository also contains demo scripts and example data. To install the development version, clone or download the repository, navigate to the base directory, and run:

```sh
pip install .
```

If you intend to make changes to the repository, you can install additional packages that are useful for development by running:

```sh
pip install ".[dev]"
```

## Developing

Please see the [DEVELOPING.md](https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md) document for contribution/development instructions.

### Creating a new release

For guidance on creating a new release, see: https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md#creating-distributions

## Citing

When using this resource, please cite the software [publication](https://physionet.org/content/wfdb-python/) on PhysioNet.
