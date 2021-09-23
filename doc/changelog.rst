Changelog
=========

.. rubric:: 1.3

- Add a pytest plugin.

.. rubric:: 1.2

- Add a user guide.
- Update the dependencies.
- Fix the type annotations to work with numpy 1.20.
- Fix deprecated usage of ``np.bool``.
- Fix handling of Context.compile when no extra flags are passed.
- Fix enqueue_zero_buffer being run on the default CUDA stream.
- Fix sequencing of SVMArray get and set operations with other commands in the
  provided command queue.

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
