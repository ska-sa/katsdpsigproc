#!/usr/bin/env python
from setuptools import setup, find_packages

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
        "OpenCL": ["pyopencl"]
    },
    zip_safe = False
)
