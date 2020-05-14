User guide
==========

katsdpsigproc is a Python package aimed at signal processing in radio
astronomy, with a focus on GPU acceleration. It consists of several pieces:

- A framework for writing accelerated code using CUDA or OpenCL;
- A lower-level abstraction layer to paper over some differences between CUDA
  and OpenCL, which is used by the higher-level layer above;
- Implementations of a number of algorithms (some specific to radio astronomy),
  both on the CPU and on accelerators.

.. toctree::
  :maxdepth: 1

  installation
  accel
  builtin-ops
