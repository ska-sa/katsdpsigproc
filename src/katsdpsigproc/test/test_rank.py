"""Tests for rank.mako."""

from typing import Optional, Any

import numpy as np
try:
    from numpy.typing import DTypeLike, ArrayLike
except ImportError:
    DTypeLike = Any     # type: ignore
    ArrayLike = Any   # type: ignore
from nose.tools import assert_equal

from .. import accel
from ..accel import DeviceArray, build
from ..abc import AbstractContext, AbstractCommandQueue, AbstractProgram
from .test_accel import device_test


_wgs = 128
_M = 1000
_N = 2000
_program: Optional[AbstractProgram] = None
_data: np.ndarray
_expected: np.ndarray


@device_test
def setup(context, queue):   # type: (AbstractContext, AbstractCommandQueue) -> None
    # Note: have to use comment-style annotation for the function signature
    # because nosetests chokes if it has Python 3-style annotations.
    global _program, _data, _expected
    _program = build(context, 'test/test_rank.mako', {'size': _wgs})
    rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
    _data = rs.randint(0, _M, size=_N).astype(np.int32)
    _expected = np.empty(_M, dtype=np.int32)
    for i in range(_M):
        _expected[i] = np.sum(np.less(_data, i))


def check_rank(context: AbstractContext, queue: AbstractCommandQueue,
               kernel_name: str, wgs: int) -> None:
    global _data, _expected
    data_d = DeviceArray(context, shape=_data.shape, dtype=_data.dtype)
    data_d.set(queue, _data)
    out_d = DeviceArray(context, shape=_expected.shape, dtype=_expected.dtype)
    assert _program is not None
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
def test_rank_serial(context: AbstractContext, queue: AbstractCommandQueue) -> None:
    check_rank(context, queue, 'test_rank_serial', 1)


@device_test
def test_rank_parallel(context: AbstractContext, queue: AbstractCommandQueue) -> None:
    check_rank(context, queue, 'test_rank_parallel', _wgs)


def run_float_func(context: AbstractContext, queue: AbstractCommandQueue,
                   kernel_name: str, data: np.ndarray, output_size: int) -> np.ndarray:
    """Do common testing for find_min_float, find_max_float, median_non_zero_float."""
    size = 128  # Number of workitems
    assert _program is not None
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


class TestFindMinMaxFloat:
    def check_array(self, context: AbstractContext, queue: AbstractCommandQueue,
                    data: ArrayLike) -> None:
        data = np.asarray(data, dtype=np.float32)
        expected = [np.nanmin(data), np.nanmax(data)]
        out = run_float_func(context, queue, 'test_find_min_max_float', data, 2)
        assert_equal(expected[0], out[0])
        assert_equal(expected[1], out[1])

    @device_test
    def test_single(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.check_array(context, queue, [5.3])

    @device_test
    def test_ordered(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        data = np.sort(np.random.uniform(-10.0, 10.0, 1000))
        self.check_array(context, queue, data)

    @device_test
    def test_nan(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.check_array(context, queue, [-10.0, 5.5, np.nan, -20.0, np.nan])

    @device_test
    def test_all_nan(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        # Can't use check_array, because np.nanmin warns if all are NaN,
        # and assert_equal doesn't handle NaN
        data = np.array([np.nan, np.nan], dtype=np.float32)
        out = run_float_func(context, queue, 'test_find_min_max_float', data, 2)
        assert np.isnan(out[0])
        assert np.isnan(out[1])

    @device_test
    def test_random(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        data = np.random.uniform(-10.0, 10.0, 1000)
        self.check_array(context, queue, data)


class TestMedianNonZero:
    def check_array(self, context: AbstractContext, queue: AbstractCommandQueue,
                    data: ArrayLike) -> None:
        data = np.asarray(data, dtype=np.float32)
        expected = np.median(data[data > 0.0])
        out = run_float_func(context, queue, 'test_median_non_zero', data, 2)
        assert_equal(expected, out[0])
        assert_equal(expected, out[1])

    @device_test
    def test_single(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.check_array(context, queue, [5.3])

    @device_test
    def test_even(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.check_array(context, queue, [1.2, 2.3, 3.4, 4.5])

    @device_test
    def test_zeros(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.check_array(context, queue, [0.0, 0.0, 0.0, 1.2, 5.3, 2.4])

    @device_test
    def test_big_even(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        data = np.random.random_sample(10001) + 0.5
        data[123] = 0.0
        self.check_array(context, queue, data)

    @device_test
    def test_big_odd(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        data = np.random.random_sample(10000) + 0.5
        data[456] = 0.0
        self.check_array(context, queue, data)
