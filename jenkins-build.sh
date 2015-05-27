#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install -r requirements.txt
pip install coverage
if [ "$label" = "cuda" ]; then
    pip install pycuda
    export CUDA_DEVICE=0
elif [ "$label" = "opencl" ]; then
    pip install pyopencl
    export PYOPENCL_CTX=0:0
fi
nosetests --with-coverage --cover-package=katsdpsigproc --cover-html
make -C doc html
