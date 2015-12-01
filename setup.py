#!/usr/bin/env python
from setuptools import setup, find_packages

tests_require = ['nose', 'mock', 'unittest2']

setup(
    name = "katsdpsigproc",
    version = "trunk",
    description = "Karoo Array Telescope accelerated signal processing tools",
    author_email = "spt@ska.ac.za",
    packages = find_packages(),
    package_data = {'': ['*.mako']},
    scripts = ["scripts/rfiflagtest.py"],
    url = "http://ska.ac.za",
    install_requires = [
        "numpy", "scipy", "decorator", "mako", "appdirs", "futures"
    ],
    extras_require = {
        "CUDA": ["pycuda>=2015.1.3"],
        "OpenCL": ["pyopencl"],
        "tests": tests_require,
        "doc": ["sphinx>=1.3"]
    },
    tests_require = tests_require,
    zip_safe = False
)
