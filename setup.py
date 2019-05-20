#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


# Avoid installing the asyncio modules on Python < 3.4. This can't be done
# just by passing a different list of packages to setup(), because we need
# the asyncio sources to be seen by sdist.
class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = build_py.find_package_modules(self, package, package_dir)
        if sys.version_info < (3, 4) and 'asyncio' in package:
            modules = []
        return modules


tests_require = ['nose', 'mock']

setup(
    name="katsdpsigproc",
    description="Karoo Array Telescope accelerated signal processing tools",
    author="SKA SA Science Processing Team",
    author_email="spt@ska.ac.za",
    packages=find_packages(),
    package_data={'': ['*.mako']},
    scripts=["scripts/rfiflagtest.py"],
    url="https://github.com/ska-sa/katsdpsigproc",
    cmdclass={'build_py': BuildPy},
    setup_requires=["katversion"],
    install_requires=[
        "numpy>=1.10",
        "scipy",
        "pandas",
        "numba>=0.36.1",   # Older versions have bugs in median functions
        "decorator",
        "mako",
        "appdirs",
        "futures; python_version<'3.2'",
        "trollius; python_version<'3.4'",
        "six"
    ],
    extras_require={
        "CUDA": ["pycuda>=2015.1.3"],
        "OpenCL": ["pyopencl>=2017.2.1"],
        "test": tests_require,
        "doc": ["sphinx>=1.3"]
    },
    tests_require=tests_require,
    zip_safe=False,
    use_katversion=True
)
