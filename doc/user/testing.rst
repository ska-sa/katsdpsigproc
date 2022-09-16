Testing
=======
Testing GPU code requires a little more setup than regular code. At a minimum
you will normally need a context and a command queue. There are two modules
that provide support for testing, depending on which testing framework is in
use.

.. _testing-pytest:

pytest
------
The module :mod:`katsdpsigproc.pytest_plugin` provides a pytest plugin to
simplify writing tests. It is not enabled by default, because the fixture names
and markers that it defines are very generic and may conflict with other
plugins. To enable it, add the following to your top-level ``conftest.py``:

.. code:: python

   pytest_plugins = ["katsdpsigproc.pytest_plugin"]

.. _pytest-device-selection:

Device selection
^^^^^^^^^^^^^^^^
There is a low-level fixture called ``device``, although you will probably not
need to use it directly. It is parametrized, allowing tests to be repeated on
multiple devices. The devices to use can be controlled by passing
:option:`!--devices` on the pytest command line. The possible values are

``first-per-api`` (default)
    Use the first available CUDA device (if any) and the first available
    OpenCL device (if any).

``all``
    Use all available devices.

``none``
    Skip tests that depend on the ``device`` fixture.

The environment variables that control device selection in
:func:`~.create_some_context` can also be used here to pick a specific
device. Tests can also use marks (see below) to filter out devices that are
not suitable for the test.

If no suitable devices are available, the test is marked as ``xfail`` and not
run.

Fixtures
^^^^^^^^
.. _fixture-patch_autotune:

patch_autotune
    Disable the normal autotuning, instead using the `test` parameter to
    :func:`.tune.autotuner` as the result of autotuning. This behaviour can
    be overridden with the :ref:`force_autotune <mark-force_autotune>` mark.

.. _fixture-device:

device (:class:`~.AbstractDevice`)
    See :ref:`pytest-device-selection`.

.. _fixture-context:

context (:class:`~.AbstractContext`)
    A context created from :ref:`device <pytest-device-selection>`. It
    automatically depends on :ref:`patch_autotune <fixture-patch_autotune>`.
    For CUDA, the context is also made current for the duration of the test.

command_queue (:class:`~.AbstractCommandQueue`)
    A command queue created from :ref:`context <fixture-context>`.

Marks
^^^^^
cuda_only
    Restrict device selection to CUDA devices. An optional
    `min_compute_capability` keyword argument can be set to a 2-tuple of
    integers to set a minimum CUDA compute capability e.g. ``(7, 2)`` will
    limit device selection to devices of compute capability 7.2 or higher.

opencl_only
    Restrict device selection to OpenCL devices.

device_filter(filter)
    Provide an arbitrary predicate which decides whether a device should be
    considered or not. Note that this needs to be invoked as
    ``pytest.mark.device_filter.with_args(filter)`` to make pytest aware
    that the filter is an argument to the mark rather than a function to
    decorate with the mark.

.. _mark-force_autotune:

force_autotune
    When combined with the :ref:`patch_autotune <fixture-patch_autotune>`
    fixture (usually implicitly, as it is used by the :ref:`context
    <fixture-context>` fixture), run the full autotuning unconditionally
    (ignoring any results in the autotuning database).

    See also :ref:`Testing autotuning <autotune-testing>` for information
    about how testing interacts with autotuning.

Example
^^^^^^^

Here's a simple example test for the :class:`!Multiply` operation we've
developed in previous sections.

.. code:: python

    pytest_plugins = ["katsdpsigproc.pytest_plugin"]

    def test_multiply(context, command_queue):
        size = 53
        template = MultiplyTemplate(context)
        op = template.instantiate(command_queue, size, 4.0)
        op.ensure_all_bound()
        src = np.random.uniform(size=size).astype(np.float32)
        op.buffer('data').set(command_queue, src)
        op()
        dst = op.buffer('data').get(command_queue)
        np.testing.assert_array_equal(dst, src * 4.0)

nose
----
The :mod:`katsdpsigproc.test.test_accel` module provides a number of decorators
that can be used to simplify writing test functions. They are designed for use
with `nose`_. Note that nose is no longer maintained and does not work with the
latest versions of Python. As such, this support is kept in katsdpsigproc only
for backwards compatibility, and you are encouraged to convert your tests to
pytest as soon as possible.

.. _nose: https://nose.readthedocs.io

To write a test function, give it two extra arguments, which will be the
context and command queue (you can call them anything as they are passed
positionally). Then decorate the test with :func:`.device_test`. Here's a
simple example test for the :class:`!Multiply` operation we've developed in
previous sections.

.. code:: python

    @device_test
    def test_multiply(context, command_queue):
        size = 53
        template = MultiplyTemplate(context)
        op = template.instantiate(command_queue, size, 4.0)
        op.ensure_all_bound()
        src = np.random.uniform(size=size).astype(np.float32)
        op.buffer('data').set(command_queue, src)
        op()
        dst = op.buffer('data').get(command_queue)
        np.testing.assert_array_equal(dst, src * 4.0)

The device and command queue are created the first time one of the decorated
tests is run; after this they are reused. This can cause problems if there is
an error like an out-of-bounds memory access, because this tends to break the
context and cause all subsequent tests to fail too.

If no devices are found, the test will be skipped. If multiple devices are
found, then the first one will be used. You can use the
:envvar:`KATSDPSIGPROC_DEVICE` environment variable to change which device is
used.
