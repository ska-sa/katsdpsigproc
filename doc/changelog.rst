Changelog
=========

.. rubric:: 1.6.0

- Add `bind` argument to :meth:`.IOSlotBase.allocate`.
- Add :meth:`.IOSlotBase.allocate_host`.
- Update tests to avoid triggering deprecation warnings.

.. rubric:: 1.5.0

- Add :envvar:`KATSDPSIGPROC_TUNE_MATCH` environment variable to control use of
  partial matches from the autotuning database.

.. rubric:: 1.4.3

- Switching testing from nose to pytest.
- Some packaging modernisation.
- Fix some links to other packages in the docs.
- Fix the return annotation on :meth:`katsdpsigproc.resource.wait_until`.
- Fix a spurious warning about a future exception not being consumed if
  an exception is thrown in the scope of a :class:`.ResourceAllocation`
  context manager.
- Change return type annotations on the abstract base classes to work better
  with mypy.

.. rubric:: 1.4.2

- Change some return type annotations on abstract base classes to work around
  mypy limitations.

.. rubric:: 1.4.1

- Destroy cuFFT plan when FftTemplate is garbage collected. This lack was (for
  unknown reasons) causing segmentation faults when repeatedly creating and
  destroying contexts without cleaning up the plans.
- Fix a potential failure of FftTemplate on 32-bit systems due to an incorrect
  type signature.
- Fix documentation on readthedocs by updating sphinxcontrib-tikz.

.. rubric:: 1.4

- Add a module for FFTs (using cuFFT).

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
