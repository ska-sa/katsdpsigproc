"""Tests for rank.mako"""

import numpy as np
from .. import accel
from ..accel import DeviceArray, build
from .test_accel import device_test
from nose.tools import assert_equal

@device_test
def setup(context, queue):
    global _program, _wgs, _data, _expected, _N, _M
    _wgs = 128
    _M = 1000
    _N = 2000
    _program = build(context, 'test/test_rank.mako', {'size': _wgs})
    rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
    _data = rs.randint(0, _M, size=_N).astype(np.int32)
    _expected = np.empty(_M, dtype=np.int32)
    for i in range(_M):
        _expected[i] = np.sum(np.less(_data, i))

def check_rank(context, queue, kernel_name, wgs):
    data_d = DeviceArray(context, shape=_data.shape, dtype=_data.dtype)
    data_d.set(queue, _data)
    out_d = DeviceArray(context, shape=_expected.shape, dtype=_expected.dtype)
    kernel = _program.get_kernel(kernel_name)
    queue.enqueue_kernel(
            kernel, [
                data_d.buffer,
                out_d.buffer,
                np.int32(_N),
                np.int32(_M)
            ],
            global_size=(wgs,), local_size=(wgs,))
    out = out_d.get(queue)
    np.testing.assert_equal(_expected, out)

@device_test
def test_rank_serial(context, queue):
    check_rank(context, queue, 'test_rank_serial', 1)

@device_test
def test_rank_parallel(context, queue):
    check_rank(context, queue, 'test_rank_parallel', _wgs)

def run_float_func(context, queue, kernel_name, data, output_size):
    """Common code for testing find_min_float, find_max_float, median_non_zero_float"""
    size = 128  # Number of workitems
    kernel = _program.get_kernel(kernel_name)
    data = np.asarray(data, dtype=np.float32)
    N = data.shape[0]
    data_d = accel.DeviceArray(context, shape=data.shape, dtype=np.float32)
    data_d.set(queue, data)
    out_d = accel.DeviceArray(context, shape=(output_size,), dtype=np.float32)
    queue.enqueue_kernel(
            kernel,
            [data_d.buffer, out_d.buffer, np.int32(N)],
            global_size=(size,),
            local_size=(size,))
    out = out_d.empty_like()
    out_d.get(queue, out)
    return out

class TestFindMinMaxFloat(object):
    def check_array(self, context, queue, data):
        data = np.asarray(data, dtype=np.float32)
        expected = [np.nanmin(data), np.nanmax(data)]
        out = run_float_func(context, queue, 'test_find_min_max_float', data, 2)
        assert_equal(expected[0], out[0])
        assert_equal(expected[1], out[1])

    @device_test
    def test_single(self, context, queue):
        self.check_array(context, queue, [5.3])

    @device_test
    def test_ordered(self, context, queue):
        data = np.sort(np.random.uniform(-10.0, 10.0, 1000))
        self.check_array(context, queue, data)

    @device_test
    def test_nan(self, context, queue):
        self.check_array(context, queue, [-10.0, 5.5, np.nan, -20.0, np.nan])

    @device_test
    def test_all_nan(self, context, queue):
        # Can't use check_array, because np.nanmin warns if all are NaN,
        # and assert_equal doesn't handle NaN
        data = np.array([np.nan, np.nan], dtype=np.float32)
        out = run_float_func(context, queue, 'test_find_min_max_float', data, 2)
        assert np.isnan(out[0])
        assert np.isnan(out[1])

    @device_test
    def test_random(self, context, queue):
        data = np.random.uniform(-10.0, 10.0, 1000)
        self.check_array(context, queue, data)

class TestMedianNonZero(object):
    def check_array(self, context, queue, data):
        data = np.asarray(data, dtype=np.float32)
        expected = np.median(data[data > 0.0])
        out = run_float_func(context, queue, 'test_median_non_zero', data, 2)
        assert_equal(expected, out[0])
        assert_equal(expected, out[1])

    @device_test
    def test_single(self, context, queue):
        self.check_array(context, queue, [5.3])

    @device_test
    def test_even(self, context, queue):
        self.check_array(context, queue, [1.2, 2.3, 3.4, 4.5])

    @device_test
    def test_zeros(self, context, queue):
        self.check_array(context, queue, [0.0, 0.0, 0.0, 1.2, 5.3, 2.4])

    @device_test
    def test_big_even(self, context, queue):
        data = np.random.random_sample(10001) + 0.5
        data[123] = 0.0
        self.check_array(context, queue, data)

    @device_test
    def test_big_odd(self, context, queue):
        data = np.random.random_sample(10000) + 0.5
        data[456] = 0.0
        self.check_array(context, queue, data)
