"""Tests for wg_reduce.mako"""

import numpy as np
from .test_accel import device_test, test_context, test_command_queue
from nose.tools import assert_equal
from ..accel import DeviceArray, build

@device_test
def setup():
    global _program, _data
    rs = np.random.RandomState(seed=1)  # Fixed seed to make test repeatable
    _data = rs.randint(0, 1000, size=97).astype(np.int32)
    _program = build(test_context, 'test/test_reduce.mako', {'size': _data.shape[0]})

def check_reduce(kernel_name, op):
    kernel = _program.get_kernel(kernel_name)
    data_d = DeviceArray(context=test_context, shape=_data.shape, dtype=_data.dtype)
    data_d.set(test_command_queue, _data)
    out_d = DeviceArray(context=test_context, shape=(1,), dtype=_data.dtype)
    test_command_queue.enqueue_kernel(
            kernel, [data_d.buffer, out_d.buffer], local_size=_data.shape, global_size=_data.shape)
    out = out_d.get(test_command_queue)[0]
    expected = reduce(op, _data)
    assert_equal(expected, out)

@device_test
def test_reduce_add():
    check_reduce('test_reduce_add', lambda x, y: x + y)

@device_test
def test_reduce_max():
    check_reduce('test_reduce_max', max)
