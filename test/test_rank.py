"""Tests for rank.mako."""

import os
from typing import Optional, Any

import numpy as np
try:
    from numpy.typing import DTypeLike, ArrayLike
except ImportError:
    DTypeLike = Any     # type: ignore
    ArrayLike = Any   # type: ignore
import pytest

from katsdpsigproc import accel
from katsdpsigproc.accel import DeviceArray, build
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue, AbstractProgram


_wgs = 128
_M = 1000
_N = 2000
_program: Optional[AbstractProgram] = None
_data: np.ndarray
_expected: np.ndarray


@pytest.fixture(autouse=True)
def setup(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    # TODO: this is extremely messy. Replace with proper pytest fixtures.
    global _program, _data, _expected
    _program = build(
        context, 'test_rank.mako', {'size': _wgs},
        extra_dirs=[os.path.dirname(__file__)])
    rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
    _data = rs.randint(0, _M, size=_N).astype(np.int32)
    _expected = np.empty(_M, dtype=np.int32)
    for i in range(_M):
        _expected[i] = np.sum(np.less(_data, i))


def check_rank(context: AbstractContext, command_queue: AbstractCommandQueue,
               kernel_name: str, wgs: int) -> None:
    global _data, _expected
    data_d = DeviceArray(context, shape=_data.shape, dtype=_data.dtype)
    data_d.set(command_queue, _data)
    out_d = DeviceArray(context, shape=_expected.shape, dtype=_expected.dtype)
    assert _program is not None
    kernel = _program.get_kernel(kernel_name)
    command_queue.enqueue_kernel(
        kernel, [
            data_d.buffer,
            out_d.buffer,
            np.int32(_N),
            np.int32(_M)
        ],
        global_size=(wgs,), local_size=(wgs,))
    out = out_d.get(command_queue)
    np.testing.assert_equal(_expected, out)


def test_rank_serial(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    check_rank(context, command_queue, 'test_rank_serial', 1)


def test_rank_parallel(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    check_rank(context, command_queue, 'test_rank_parallel', _wgs)


def run_float_func(context: AbstractContext, command_queue: AbstractCommandQueue,
                   kernel_name: str, data: np.ndarray, output_size: int) -> np.ndarray:
    """Do common testing for find_min_float, find_max_float, median_non_zero_float."""
    size = 128  # Number of workitems
    assert _program is not None
    kernel = _program.get_kernel(kernel_name)
    data = np.asarray(data, dtype=np.float32)
    N = data.shape[0]
    data_d = accel.DeviceArray(context, shape=data.shape, dtype=np.float32)
    data_d.set(command_queue, data)
    out_d = accel.DeviceArray(context, shape=(output_size,), dtype=np.float32)
    command_queue.enqueue_kernel(
        kernel,
        [data_d.buffer, out_d.buffer, np.int32(N)],
        global_size=(size,),
        local_size=(size,))
    out = out_d.empty_like()
    out_d.get(command_queue, out)
    return out


class TestFindMinMaxFloat:
    def check_array(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    data: ArrayLike) -> None:
        data = np.asarray(data, dtype=np.float32)
        expected = [np.nanmin(data), np.nanmax(data)]
        out = run_float_func(context, command_queue, 'test_find_min_max_float', data, 2)
        assert expected[0] == out[0]
        assert expected[1] == out[1]

    def test_single(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self.check_array(context, command_queue, [5.3])

    def test_ordered(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        data = np.sort(np.random.uniform(-10.0, 10.0, 1000))
        self.check_array(context, command_queue, data)

    def test_nan(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self.check_array(context, command_queue, [-10.0, 5.5, np.nan, -20.0, np.nan])

    def test_all_nan(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        # Can't use check_array, because np.nanmin warns if all are NaN,
        # and asserting == doesn't handle NaN.
        data = np.array([np.nan, np.nan], dtype=np.float32)
        out = run_float_func(context, command_queue, 'test_find_min_max_float', data, 2)
        assert np.isnan(out[0])
        assert np.isnan(out[1])

    def test_random(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        data = np.random.uniform(-10.0, 10.0, 1000)
        self.check_array(context, command_queue, data)


class TestMedianNonZero:
    def check_array(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    data: ArrayLike) -> None:
        data = np.asarray(data, dtype=np.float32)
        expected = np.median(data[data > 0.0])
        out = run_float_func(context, command_queue, 'test_median_non_zero', data, 2)
        assert expected == out[0]
        assert expected == out[1]

    def test_single(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self.check_array(context, command_queue, [5.3])

    def test_even(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self.check_array(context, command_queue, [1.2, 2.3, 3.4, 4.5])

    def test_zeros(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self.check_array(context, command_queue, [0.0, 0.0, 0.0, 1.2, 5.3, 2.4])

    def test_big_even(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        data = np.random.random_sample(10001) + 0.5
        data[123] = 0.0
        self.check_array(context, command_queue, data)

    def test_big_odd(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        data = np.random.random_sample(10000) + 0.5
        data[456] = 0.0
        self.check_array(context, command_queue, data)
