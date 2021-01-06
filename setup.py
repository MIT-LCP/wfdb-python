"""The WFDB Python Toolbox.
See: https://www.physionet.org/physiotools/wfdb.shtml
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version number from the version.py file
with open('wfdb/version.py') as f:
    __version__ = f.read().split()[-1].strip("'")

setup(
    name='wfdb',

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='The WFDB Python Toolbox',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/MIT-LCP/wfdb-python',

    # Author details
    author='The Laboratory for Computational Physiology',
    author_email='support@physionet.org',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='WFDB clinical waveform',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'certifi>=2016.8.2',
        'chardet>=3.0.0',
        'cycler>=0.10.0',
        'idna>=2.2',
        'joblib>=0.11',
        'kiwisolver>=1.1.0',
        'matplotlib>=2.0.0',
        'numpy>=1.10.1',
        'pandas>=0.17.0',
        'pyparsing>=2.0.4',
        'python-dateutil>=2.4.2',
        'pytz>=2017.2',
        'requests>=2.8.1',
        'scikit-learn>=0.18',
        'scipy>=0.17.0',
        'threadpoolctl>=1.0.0',
        'urllib3>=1.22'
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'test': ['nose>=1.3.7']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={'wfdb': ['wfdb.config'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    # data_files=[('config', ['wfdb.config'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

    # Add ways to quickly filter project
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License"
    ],
)
