"""Tests for wg_reduce.mako and reduce.py"""

import numpy as np
from .test_accel import device_test, force_autotune
from nose.tools import assert_equal
from ..accel import DeviceArray, build
builtin_reduce = reduce
from .. import reduce

class Fixture(object):
    def __init__(self, context, rs, size):
        self.data = rs.randint(0, 1000, size=size).astype(np.int32)
        self.program = build(context, 'test/test_reduce.mako', {'size': size})

@device_test
def setup(context, queue):
    global _fixtures
    rs = np.random.RandomState(seed=1)  # Fixed seed to make test repeatable
    _fixtures = [Fixture(context, rs, size) for size in (12, 97, 256)]

def check_reduce(context, queue, kernel_name, op):
    for fixture in _fixtures:
        data = fixture.data
        kernel = fixture.program.get_kernel(kernel_name)
        data_d = DeviceArray(context=context, shape=data.shape, dtype=data.dtype)
        data_d.set(queue, data)
        out_d = DeviceArray(context=context, shape=(1,), dtype=data.dtype)
        queue.enqueue_kernel(
                kernel, [data_d.buffer, out_d.buffer], local_size=data.shape, global_size=data.shape)
        out = out_d.get(queue)[0]
        expected = builtin_reduce(op, data)
        assert_equal(expected, out)

@device_test
def test_reduce_add(context, queue):
    check_reduce(context, queue, 'test_reduce_add', lambda x, y: x + y)

@device_test
def test_reduce_max(context, queue):
    check_reduce(context, queue, 'test_reduce_max', max)

class TestHReduce(object):
    """Tests for :class:`katsdpsigproc.reduce.HReduce`"""

    def check(self, context, queue, rows, columns, column_range):
        template = reduce.HReduceTemplate(context, np.uint32, 'unsigned int', 'a + b', 0)
        fn = template.instantiate(queue, (rows, columns), column_range)
        fn.ensure_all_bound()
        device_src = fn.buffer('src')
        device_dest = fn.buffer('dest')
        src = device_src.empty_like()
        rs = np.random.RandomState(seed=1) # To be reproducible
        src[:] = rs.randint(0, 100000, (rows, columns))
        device_src.set(queue, src)
        fn()
        dest = device_dest.get(queue)
        expected = np.sum(src[:, column_range[0]:column_range[1]], axis=1)
        np.testing.assert_equal(expected, dest)

    @device_test
    def test_normal(self, context, queue):
        """Test the usual case"""
        self.check(context, queue, 129, 173, (67, 128))

    @device_test
    def test_small(self, context, queue):
        """Test that the special case for columns < work group size works"""
        self.check(context, queue, 7, 8, (1, 7))

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Test that autotuner runs successfully"""
        reduce.HReduceTemplate(context, np.uint32, 'unsigned int', 'a + b', '0')
