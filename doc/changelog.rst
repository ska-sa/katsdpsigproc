Changelog
=========

.. rubric:: 1.1

- Use BLOCKING_SYNC for CUDA events, to avoid spinning on the CPU.
- Drop support for Python 3.5.
- Add type annotations for many classes.
- Make some classes abstract base classes (they were already semantically, but
  now it's indicated using :py:mod:`abc`.
- Introduce some abstract base classes to underly the CUDA and OpenCL
  implementations.

.. rubric:: 1.0

This is the first versioned release.
