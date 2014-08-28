"""Tests for rank.mako"""

import numpy as np
from .test_accel import device_test, test_context, test_command_queue
from nose.tools import assert_equal
if test_context:
    from ..accel import DeviceArray, build

@device_test
def setup():
    global _program, _wgs, _data, _expected, _N, _M
    _wgs = 128
    _M = 1000
    _N = 2000
    _program = build(test_context, 'test/test_rank.mako', {'size': _wgs})
    rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
    _data = rs.randint(0, _M, size=_N).astype(np.int32)
    _expected = np.empty(_M, dtype=np.int32)
    for i in range(_M):
        _expected[i] = np.sum(np.less(_data, i))

def check_rank(kernel_name, wgs):
    data_d = DeviceArray(test_context, shape=_data.shape, dtype=_data.dtype)
    data_d.set(test_command_queue, _data)
    out_d = DeviceArray(test_context, shape=_expected.shape, dtype=_expected.dtype)
    kernel = _program.get_kernel(kernel_name)
    test_command_queue.enqueue_kernel(
            kernel, [
                data_d.buffer,
                out_d.buffer,
                np.int32(_N),
                np.int32(_M)
            ],
            global_size=(wgs,), local_size=(wgs,))
    out = out_d.get(test_command_queue)
    np.testing.assert_equal(_expected, out)

@device_test
def test_rank_serial():
    check_rank('test_rank_serial', 1)

@device_test
def test_rank_parallel():
    check_rank('test_rank_parallel', _wgs)
