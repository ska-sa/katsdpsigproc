"""Tests for rank.mako"""

import numpy as np
from .. import accel
from ..accel import DeviceArray, build
from .test_accel import device_test, test_context, test_command_queue
from nose.tools import assert_equal

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

class TestMedianNonZero(object):
    @device_test
    def setup(self):
        self.size = 128  # Number of workitems
        self.kernel = _program.get_kernel('test_median_non_zero')

    def check_array(self, data):
        data = np.asarray(data, dtype=np.float32)
        N = data.shape[0]
        expected = np.median(data[data > 0.0])
        data_d = accel.DeviceArray(test_context, shape=data.shape, dtype=np.float32)
        data_d.set(test_command_queue, data)
        out_d = accel.DeviceArray(test_context, shape=(2,), dtype=np.float32)
        test_command_queue.enqueue_kernel(
                self.kernel,
                [data_d.buffer, out_d.buffer, np.int32(N)],
                global_size=(self.size,),
                local_size=(self.size,))
        out = out_d.empty_like()
        out_d.get(test_command_queue, out)

        assert_equal(expected, out[0])
        assert_equal(expected, out[1])

    @device_test
    def test_single(self):
        self.check_array([5.3])

    @device_test
    def test_even(self):
        self.check_array([1.2, 2.3, 3.4, 4.5])

    @device_test
    def test_zeros(self):
        self.check_array([0.0, 0.0, 0.0, 1.2, 5.3, 2.4])

    @device_test
    def test_big_even(self):
        data = np.random.random_sample(10001) + 0.5
        data[123] = 0.0
        self.check_array(data)

    @device_test
    def test_big_odd(self):
        data = np.random.random_sample(10000) + 0.5
        data[456] = 0.0
        self.check_array(data)