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

"""Tests for :mod:`katsdpsigproc.transpose`."""

from typing import cast

import numpy as np
import pytest

from katsdpsigproc import accel
from katsdpsigproc import transpose
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue


class TestTranspose:
    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @pytest.mark.parametrize('R, C', [(4, 5), (53, 7), (53, 81), (32, 64)])
    def test_transpose(self, R: int, C: int,
                       context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        template = transpose.TransposeTemplate(context, np.float32, 'float')
        fn = template.instantiate(command_queue, (R, C))
        # Force some padded, to check that stride calculation works
        src_slot = cast(accel.IOSlot, fn.slots['src'])
        dest_slot = cast(accel.IOSlot, fn.slots['dest'])
        self.pad_dimension(src_slot.dimensions[0], 1)
        self.pad_dimension(src_slot.dimensions[1], 4)
        self.pad_dimension(dest_slot.dimensions[0], 2)
        self.pad_dimension(dest_slot.dimensions[1], 3)

        ary = np.random.randn(R, C).astype(np.float32)
        src = fn.slots['src'].allocate(fn.allocator)
        dest = fn.slots['dest'].allocate(fn.allocator)
        src.set_async(command_queue, ary)
        fn()
        out = dest.get(command_queue)
        np.testing.assert_equal(ary.T, out)

    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext) -> None:
        """Check that the autotuner runs successfully."""
        transpose.TransposeTemplate(context, np.uint8, 'unsigned char')
