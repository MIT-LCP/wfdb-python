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

The distribution is hosted on PyPI at: <https://pypi.python.org/pypi/wfdb/>. The package can be directly installed from PyPI using either pip or poetry:

```sh
pip install wfdb
poetry add wfdb
```

On Linux systems, accessing _compressed_ WFDB signal files requires installing `libsndfile`, by running `sudo apt-get install libsndfile1` or `sudo yum install libsndfile`. Support for Apple M1 systems is a work in progess (see <https://github.com/bastibe/python-soundfile/issues/310> and <https://github.com/bastibe/python-soundfile/issues/325>).

The development version is hosted at: <https://github.com/MIT-LCP/wfdb-python>. This repository also contains demo scripts and example data. To install the development version, clone or download the repository, navigate to the base directory, and run:

```sh
# Without dev dependencies
pip install .
poetry install

# With dev dependencies
pip install ".[dev]"
poetry install -E dev

# Install the dependencies only
poetry install -E dev --no-root
```

**See the [note](https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md#package-and-dependency-management) about dev dependencies.**

## Developing

Please see the [DEVELOPING.md](https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md) document for contribution/development instructions.

## Citing

When using this resource, please cite the software [publication](https://physionet.org/content/wfdb-python/) on PhysioNet.
