#!/usr/bin/env python
from setuptools import setup, find_packages


tests_require = ['nose', 'asynctest', 'graphviz']

setup(
    name="katsdpsigproc",
    description="Karoo Array Telescope accelerated signal processing tools",
    author="MeerKAT SDP team",
    author_email="sdpdev+katsdpsigproc@ska.ac.za",
    packages=find_packages(),
    package_data={'': ['*.mako'], 'katsdpsigproc': ['py.typed']},
    url="https://github.com/ska-sa/katsdpsigproc",
    setup_requires=["katversion"],
    install_requires=[
        "appdirs",
        "decorator",
        "mako",
        "numba>=0.36.1",   # Older versions have bugs in median functions
        "numpy>=1.10",
        "pandas",
        "scipy",
        "typing_extensions"
    ],
    extras_require={
        "CUDA": ["pycuda>=2015.1.3"],
        "OpenCL": ["pyopencl>=2017.2.1"],
        "test": tests_require,
        "doc": ["sphinx>=1.3", "sphinxcontrib-tikz", "sphinx-rtd-theme"]
    },
    python_requires=">=3.6",
    tests_require=tests_require,
    zip_safe=False,
    use_katversion=True
)
