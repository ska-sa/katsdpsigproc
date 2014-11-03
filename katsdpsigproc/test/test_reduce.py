"""Tests for wg_reduce.mako"""

import numpy as np
from .test_accel import device_test
from nose.tools import assert_equal
from ..accel import DeviceArray, build

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
        expected = reduce(op, data)
        assert_equal(expected, out)

@device_test
def test_reduce_add(context, queue):
    check_reduce(context, queue, 'test_reduce_add', lambda x, y: x + y)

@device_test
def test_reduce_max(context, queue):
    check_reduce(context, queue, 'test_reduce_max', max)
