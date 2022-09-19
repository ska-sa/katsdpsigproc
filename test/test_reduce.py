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

"""Tests for wg_reduce.mako and :mod:`katsdpsigproc.reduce`."""

import os
from typing import Tuple

import numpy as np
import pytest

from katsdpsigproc.accel import DeviceArray, build
from katsdpsigproc import reduce
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue


@pytest.mark.parametrize('size', [1, 4, 12, 16, 32, 87, 97, 160, 256])
@pytest.mark.parametrize('allow_shuffle', [True, False])
@pytest.mark.parametrize('broadcast', [True, False])
@pytest.mark.parametrize(
    'kernel_name, op', [('test_reduce_add', np.add), ('test_reduce_max', np.maximum)])
def test_reduce(context: AbstractContext, command_queue: AbstractCommandQueue,
                size: int, allow_shuffle: bool, broadcast: bool,
                kernel_name: str, op: np.ufunc) -> None:
    rs = np.random.RandomState(seed=1)  # Fixed seed to make test repeatable
    rows = 256 // size
    data = rs.randint(0, 1000, size=(rows, size)).astype(np.int32)
    program = build(
        context, 'test_reduce.mako',
        {
            'size': size,
            'allow_shuffle': allow_shuffle,
            'broadcast': broadcast,
            'rows': rows
        },
        extra_dirs=[os.path.dirname(__file__)]
    )

    kernel = program.get_kernel(kernel_name)
    data_d = DeviceArray(context=context, shape=data.shape, dtype=data.dtype)
    data_d.set(command_queue, data)
    out_d = DeviceArray(context=context, shape=data.shape, dtype=data.dtype)
    dims = tuple(reversed(data.shape))
    command_queue.enqueue_kernel(
        kernel, [data_d.buffer, out_d.buffer], local_size=dims, global_size=dims)
    out = out_d.get(command_queue)
    expected = op.reduce(data, axis=1, keepdims=True)
    if broadcast:
        expected = np.repeat(expected, data.shape[1], axis=1)
    else:
        out = out[:, 0:1]
    np.testing.assert_array_equal(expected, out)


class TestHReduce:
    """Tests for :class:`katsdpsigproc.reduce.HReduce`."""

    @pytest.mark.parametrize(
        'rows, columns, column_range',
        [
            (129, 173, (67, 128)),
            (7, 8, (1, 7))  # Test that the special case for columns < work group size works
        ]
    )
    def test(self, context: AbstractContext, command_queue: AbstractCommandQueue,
             rows: int, columns: int, column_range: Tuple[int, int]) -> None:
        template = reduce.HReduceTemplate(context, np.uint32, 'unsigned int', 'a + b', '0')
        fn = template.instantiate(command_queue, (rows, columns), column_range)
        fn.ensure_all_bound()
        device_src = fn.buffer('src')
        device_dest = fn.buffer('dest')
        src = device_src.empty_like()
        rs = np.random.RandomState(seed=1)  # To be reproducible
        src[:] = rs.randint(0, 100000, (rows, columns))
        device_src.set(command_queue, src)
        fn()
        dest = device_dest.get(command_queue)
        expected = np.sum(src[:, column_range[0]:column_range[1]], axis=1)
        np.testing.assert_equal(expected, dest)

    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext) -> None:
        """Test that autotuner runs successfully."""
        reduce.HReduceTemplate(context, np.uint32, 'unsigned int', 'a + b', '0')
