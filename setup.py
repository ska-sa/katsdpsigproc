#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name = "katsdpsigproc",
    version = "trunk",
    description = "Karoo Array Telescope accelerated signal processing tools",
    author_email = "spt@ska.ac.za",
    packages = find_packages(),
    package_data = {'': ['*.cu']},
    scripts = ["scripts/rfiflagtest.py"],
    url = "http://ska.ac.za",
    install_requires = [
        "numpy", "scipy"
    ],
    extras_require = {
        "CUDA": ["mako", "pycuda"]
    },
    zip_safe = False
)
