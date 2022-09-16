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

"""Tests for :mod:`katsdpsigproc.fill`."""

from typing import cast

import numpy as np
import pytest

from katsdpsigproc import accel
from katsdpsigproc import fill
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue


class TestFill:
    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    def test_fill(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        shape = (75, 63)
        template = fill.FillTemplate(context, np.uint32, 'unsigned int')
        fn = template.instantiate(command_queue, shape)
        data_slot = cast(accel.IOSlot, fn.slots['data'])
        self.pad_dimension(data_slot.dimensions[0], 5)
        self.pad_dimension(data_slot.dimensions[1], 10)
        fn.ensure_all_bound()
        data = fn.buffer('data')
        # Do the fill
        fn.set_value(0xDEADBEEF)
        fn()
        # Check the result, including padding
        ret = data.get(command_queue)
        assert ret.base is not None
        ret = ret.base
        np.testing.assert_equal(ret, 0xDEADBEEF)

    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext) -> None:
        """Test that autotuner runs successfully."""
        fill.FillTemplate(context, np.uint8, 'unsigned char')
