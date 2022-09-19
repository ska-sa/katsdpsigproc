################################################################################
# Copyright (c) 2014-2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Tests for rank.mako."""

import os
from typing import Any

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


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
    return rs.randint(0, _M, size=_N).astype(np.int32)


@pytest.fixture(scope="module")
def expected(data: np.ndarray) -> np.ndarray:
    expected = np.empty(_M, dtype=np.int32)
    for i in range(_M):
        expected[i] = np.sum(np.less(data, i))
    return expected


@pytest.fixture
def program(context: AbstractContext) -> AbstractProgram:
    return build(
        context, 'test_rank.mako', {'size': _wgs},
        extra_dirs=[os.path.dirname(__file__)])


@pytest.mark.parametrize(
    'kernel_name, wgs',
    [
        ('test_rank_serial', 1),
        ('test_rank_parallel', _wgs)
    ]
)
def test_rank(context: AbstractContext, command_queue: AbstractCommandQueue,
              data: np.ndarray, expected: np.ndarray, program: AbstractProgram,
              kernel_name: str, wgs: int) -> None:
    data_d = DeviceArray(context, shape=data.shape, dtype=data.dtype)
    data_d.set(command_queue, data)
    out_d = DeviceArray(context, shape=expected.shape, dtype=expected.dtype)
    kernel = program.get_kernel(kernel_name)
    command_queue.enqueue_kernel(
        kernel, [
            data_d.buffer,
            out_d.buffer,
            np.int32(_N),
            np.int32(_M)
        ],
        global_size=(wgs,), local_size=(wgs,))
    out = out_d.get(command_queue)
    np.testing.assert_equal(expected, out)


def run_float_func(context: AbstractContext, command_queue: AbstractCommandQueue,
                   program: AbstractProgram,
                   kernel_name: str, data: np.ndarray, output_size: int) -> np.ndarray:
    """Do common testing for find_min_float, find_max_float, median_non_zero_float."""
    size = 128  # Number of workitems
    kernel = program.get_kernel(kernel_name)
    data = data.astype(np.float32)
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
                    program: AbstractProgram, data_: ArrayLike) -> None:
        data = np.asarray(data_, dtype=np.float32)
        expected = [np.nanmin(data), np.nanmax(data)]
        out = run_float_func(context, command_queue, program, 'test_find_min_max_float', data, 2)
        assert expected[0] == out[0]
        assert expected[1] == out[1]

    @pytest.mark.parametrize('data', [[5.3], [-10.0, 5.5, np.nan, -20.0, np.nan]])
    def test_simple(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    program: AbstractProgram, data: ArrayLike) -> None:
        self.check_array(context, command_queue, program, data)

    @pytest.mark.parametrize('ordered', [False, True])
    def test_random(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    program: AbstractProgram, ordered: bool) -> None:
        rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
        data = rs.uniform(-10.0, 10.0, 1000)
        if ordered:
            data = np.sort(data)
        self.check_array(context, command_queue, program, data)

    def test_all_nan(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                     program: AbstractProgram) -> None:
        # Can't use check_array, because np.nanmin warns if all are NaN,
        # and asserting == doesn't handle NaN.
        data = np.array([np.nan, np.nan], dtype=np.float32)
        out = run_float_func(context, command_queue, program, 'test_find_min_max_float', data, 2)
        assert np.isnan(out[0])
        assert np.isnan(out[1])


class TestMedianNonZero:
    def check_array(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    program: AbstractProgram, data_: ArrayLike) -> None:
        data = np.asarray(data_, dtype=np.float32)
        expected = np.median(data[data > 0.0])
        out = run_float_func(context, command_queue, program, 'test_median_non_zero', data, 2)
        assert expected == out[0]
        assert expected == out[1]

    @pytest.mark.parametrize(
        'data',
        [
            [5.3],
            [1.2, 2.3, 3.4, 4.5],
            [0.0, 0.0, 0.0, 1.2, 5.3, 2.4]
        ]
    )
    def test_simple(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    program: AbstractProgram, data: ArrayLike) -> None:
        self.check_array(context, command_queue, program, data)

    def test_big_even(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                      program: AbstractProgram) -> None:
        rs = np.random.RandomState(seed=1)   # Fixed seed to make test repeatable
        data = rs.random_sample(10001) + 0.5
        data[123] = 0.0
        self.check_array(context, command_queue, program, data)

    def test_big_odd(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                     program: AbstractProgram) -> None:
        rs = np.random.RandomState(seed=2)   # Fixed seed to make test repeatable
        data = rs.random_sample(10000) + 0.5
        data[456] = 0.0
        self.check_array(context, command_queue, program, data)
