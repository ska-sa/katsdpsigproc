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

"""Tests for :mod:`katsdpsigproc.percentile`."""

from typing import Tuple, Optional, cast

import numpy as np
import pytest

from . import complex_normal
from katsdpsigproc import accel
from katsdpsigproc import percentile
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue


class TestPercentile5:
    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @pytest.mark.parametrize(
        'R, C, is_amplitude, column_range',
        [
            (4096, 1, False, None),
            (4095, 4029, True, (0, 4009)),
            (4094, 4030, False, (100, 4030)),
            (2343, 6031, False, (123, 4001)),
            (4092, 4032, True, None)
        ]
    )
    def test_percentile5(self, R: int, C: int, is_amplitude: bool,
                         column_range: Optional[Tuple[int, int]],
                         context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        template = percentile.Percentile5Template(context, max_columns=5000,
                                                  is_amplitude=is_amplitude)
        fn = template.instantiate(command_queue, (R, C), column_range)
        # Force some padded, to check that stride calculation works
        src_slot = cast(accel.IOSlot, fn.slots['src'])
        dest_slot = cast(accel.IOSlot, fn.slots['dest'])
        self.pad_dimension(src_slot.dimensions[0], 1)
        self.pad_dimension(src_slot.dimensions[1], 4)
        self.pad_dimension(dest_slot.dimensions[0], 2)
        self.pad_dimension(dest_slot.dimensions[1], 3)
        rs = np.random.RandomState(seed=1)
        if is_amplitude:
            ary = np.abs(rs.randn(R, C)).astype(np.float32)  # note positive numbers required
        else:
            ary = complex_normal(rs, size=(R, C)).astype(np.complex64)
        src = fn.slots['src'].allocate(fn.allocator)
        dest = fn.slots['dest'].allocate(fn.allocator)
        src.set_async(command_queue, ary)
        fn()
        out = dest.get(command_queue)
        if column_range is None:
            column_range = (0, C)
        expected = np.percentile(np.abs(ary[:, column_range[0]:column_range[1]]),
                                 [0, 100, 25, 75, 50], axis=1, interpolation='lower')
        expected = expected.astype(dtype=np.float32)
        # When amplitudes are being computed, we won't get a bit-exact match
        if is_amplitude:
            np.testing.assert_equal(expected, out)
        else:
            np.testing.assert_allclose(expected, out, 1e-6)

    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext) -> None:
        """Check that the autotuner runs successfully."""
        percentile.Percentile5Template(context, max_columns=5000)
