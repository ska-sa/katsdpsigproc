"""Tests for wg_reduce.mako"""

import numpy as np
from .test_accel import device_test, test_context, test_command_queue
from nose.tools import assert_equal
from ..accel import DeviceArray, build

class Fixture(object):
    def __init__(self, rs, size):
        self.data = rs.randint(0, 1000, size=size).astype(np.int32)
        self.program = build(test_context, 'test/test_reduce.mako', {'size': size})

@device_test
def setup():
    global _fixtures
    rs = np.random.RandomState(seed=1)  # Fixed seed to make test repeatable
    _fixtures = [Fixture(rs, size) for size in (12, 97, 256)]

def check_reduce(kernel_name, op):
    for fixture in _fixtures:
        data = fixture.data
        kernel = fixture.program.get_kernel(kernel_name)
        data_d = DeviceArray(context=test_context, shape=data.shape, dtype=data.dtype)
        data_d.set(test_command_queue, data)
        out_d = DeviceArray(context=test_context, shape=(1,), dtype=data.dtype)
        test_command_queue.enqueue_kernel(
                kernel, [data_d.buffer, out_d.buffer], local_size=data.shape, global_size=data.shape)
        out = out_d.get(test_command_queue)[0]
        expected = reduce(op, data)
        assert_equal(expected, out)

@device_test
def test_reduce_add():
    check_reduce('test_reduce_add', lambda x, y: x + y)

@device_test
def test_reduce_max():
    check_reduce('test_reduce_max', max)
