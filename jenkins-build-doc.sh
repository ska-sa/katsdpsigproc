#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install -r requirements.txt
pip install pycuda 'pyopencl==2015.1'    # 2015.2.1 fails to compile
rm -rf doc/_build
make -C doc html
