# The WFDB Python Package

![signals](https://raw.githubusercontent.com/MIT-LCP/wfdb-python/main/demo-img.png)

[![PyPI Downloads](https://img.shields.io/pypi/dm/wfdb.svg?label=PyPI%20downloads)](https://pypi.org/project/wfdb/)
[![PhysioNet Project](https://img.shields.io/badge/DOI-10.13026%2Fjt5e--gq15-blue)](https://doi.org/10.13026/jt5e-gq15)

## Introduction

WFDB defines a set of open standards and software tools for the storage, sharing, and analysis of physiologic signal and annotation data. It has been widely used for decades to support biomedical research, clinical studies, and education. For more information on WFDB and its various implementations, see the [WFDB website](https://wfdb.mit.edu/).

This package is heavily inspired by the original [WFDB Software Package](https://www.physionet.org/content/wfdb/). It aims to replicate many of its command-line APIs in Python, but there is no promise of consistency between the two. Core I/O functionality is based on the Waveform Database (WFDB) [specifications](https://github.com/wfdb/wfdb-spec/).

## Resources

- WFDB Website: [https://wfdb.mit.edu/](https://wfdb.mit.edu/)
- WFDB-Python Documentation: [http://wfdb.readthedocs.io](http://wfdb.readthedocs.io)
- WFDB Specification: [https://github.com/wfdb/wfdb-spec/](https://github.com/wfdb/wfdb-spec/)
- Demo Notebook: [https://github.com/MIT-LCP/wfdb-python/blob/main/demo.ipynb](https://github.com/MIT-LCP/wfdb-python/blob/main/demo.ipynb)
- WFDB Development Repository: [https://github.com/MIT-LCP/wfdb-python](https://github.com/MIT-LCP/wfdb-python)

## Installation

The distribution is hosted on PyPI at: [https://pypi.python.org/pypi/wfdb/](https://pypi.python.org/pypi/wfdb/). The package can be directly installed from PyPI using pip:

```sh
pip install wfdb
```

Alternatively, add it to a project managed with [uv](https://docs.astral.sh/uv/):

```sh
uv add wfdb
```

On some less-common systems, you may need to install `libsndfile` separately.  See the [soundfile installation notes](https://pypi.org/project/soundfile/) for more information.

## Development

The development version is hosted at: [https://github.com/MIT-LCP/wfdb-python](https://github.com/MIT-LCP/wfdb-python). This repository also contains demo scripts and example data. To set up a development environment, install [uv](https://docs.astral.sh/uv/), clone the repository, navigate to the base directory, and run:

```sh
uv sync
```

This creates a virtual environment in `.venv` containing the package and all development tools. Commands can then be run inside this environment with `uv run`, for example:

```sh
uv run pytest
```

Please see the [DEVELOPING.md](https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md) document for contribution/development instructions.

### Creating a new release

For guidance on creating a new release, see: https://github.com/MIT-LCP/wfdb-python/blob/main/DEVELOPING.md#creating-distributions

## Citing

When using this resource, please cite the software [publication](https://physionet.org/content/wfdb-python/) on PhysioNet.