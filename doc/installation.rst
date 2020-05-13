Installation
============
katsdpsigproc is available from the Python Package Index, so it can be installed just with::

    pip install katsdpsigproc

The package itself is pure Python, but a number of the dependencies have
compiled code and may require a compiler to be present if you have a
platform that isn't supported by the available binary wheels.

Alternatively, the latest development version can be installed from Github with::

    pip install https://github.com/ska-sa/katsdpsigproc

This is sufficient for running the CPU code, but for acceleration using CUDA or
OpenCL you will also need to install :mod:`pycuda` or :mod:`pyopencl`
respectively.

.. _configuration:

Configuration
-------------

To check that your drivers have been successfully detected, run the following
script:

.. literalinclude:: examples/hello_accel.py

If more than one device is detected, it will list them and prompt you to
select one. This can happen even if you only have a single GPU, if it is
available both via CUDA and OpenCL. Note that for OpenCL the "platform name"
is shown in parentheses; for NVIDIA GPUs this is confusingly "NVIDIA CUDA".
If you want to use the device via the CUDA API, it is the one with just "CUDA"
in parentheses.

An environment variable can be used to specify a device so
that you are not prompted every time:

.. envvar:: KATSDPSIGPROC_DEVICE

  Specify the device to use. The value must be a number, which matches the
  numbering offered by the menu.
