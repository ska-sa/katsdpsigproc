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

.. literalinclude:: user/examples/hello_accel.py

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

Troubleshooting
---------------

CUDA on Linux
_____________

If the script above does not work or does not detect a CUDA device, the
following steps will help isolate the problem.

1. Check that :mod:`pycuda` is installed. If it is not, you won't get an
   error, but CUDA devices will not be available.

2. Run :program:`nvidia-smi`. This checks that the NVIDIA drivers are installed
   and that the userspace component can talk to the kernel component. If not,
   you might need to reinstall the drivers.

3. Run a compiled CUDA program. For example, the CUDA installation will
   usually offer to install samples. The :program:`deviceQuery` sample is a good
   test since it'll show the devices detected. If this fails then there is
   likely something wrong with your CUDA library installation.

4. Run ``python -c 'import pycuda.autoinit'``. If that fails (and the previous
   steps succeed) the error is most likely into your :mod:`pycuda`
   installation. A common problem is that CUDA is upgraded but pycuda is still
   linked against the old version. In this case you need to uninstall it,
   remove any cached binary wheel, and reinstall it.
