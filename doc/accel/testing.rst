Testing
=======
Testing GPU code requires a little more setup than regular code. At a minimum
you will normally need a context and a command queue. The
:mod:`katsdpsigproc.test.test_accel` module provides a number of decorators
that can be used to simplify writing test functions. They are designed for use
with `nose`_. You might be able to make them work with `pytest`_ too, but
there is no specific pytest support yet.

.. _nose: https://nose.readthedocs.io
.. _pytest: https://docs.pytest.org

To write a test function, give it two extra arguments, which will be the
context and command queue (you can call them anything as they are passed
positionally). Then decorate the test with :func:`.device_test`. Here's a
simple example test for the :class:`Multiply` operation we've developed in
previous sections.

.. code:: python

    @device_test
    def test_multiply(ctx, queue):
        size = 53
        template = MultiplyTemplate(ctx)
        op = template.instantiate(queue, size, 4.0)
        op.ensure_all_bound()
        src = np.random.uniform(size=size).astype(np.float32)
        op.buffer('data').set(queue, src)
        op()
        dst = op.buffer('data').get(queue)
        np.testing.assert_array_equal(dst, src * 4.0)

The device and command queue are created the first time one of the decorated
tests is run; after this they are reused. This can cause problems if there is
an error like an out-of-bounds memory access, because this tends to break the
context and cause all subsequent tests to fail too.

If no devices are found, the test will be skipped. If multiple devices are
found, then the first one will be used. You can use the
:envvar:`KATSDPSIGPROC_DEVICE` environment variable to change which device is
used (see :ref:`configuration`).
