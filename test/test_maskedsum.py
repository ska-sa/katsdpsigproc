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

"""Tests for :mod:`katsdpsigproc.maskedsum`."""

from typing import cast

import numpy as np
import pytest

from katsdpsigproc import accel
from katsdpsigproc import maskedsum
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue


class TestMaskedSum:
    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @pytest.mark.parametrize(
        'R,C',
        [(4096, 2), (4096, 4029), (4096, 4030), (4096, 4031), (4096, 4032)]
    )
    @pytest.mark.parametrize('use_amplitudes', [False, True])
    def test_maskedsum(self, R: int, C: int, use_amplitudes: bool,
                       context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        template = maskedsum.MaskedSumTemplate(context, use_amplitudes)
        fn = template.instantiate(command_queue, (R, C))
        # Force some padding, to check that stride calculation works
        src_slot = cast(accel.IOSlot, fn.slots['src'])
        self.pad_dimension(src_slot.dimensions[0], 1)
        self.pad_dimension(src_slot.dimensions[1], 4)
        ary = np.random.randn(R, C, 2).astype(np.float32).view(dtype=np.complex64)[..., 0]
        msk = np.ones((R,)).astype(np.float32)
        src = fn.slots['src'].allocate(fn.allocator)
        mask = fn.slots['mask'].allocate(fn.allocator)
        dest = fn.slots['dest'].allocate(fn.allocator)
        src.set_async(command_queue, ary)
        mask.set_async(command_queue, msk)
        fn()
        out = dest.get(command_queue).reshape(-1)
        if use_amplitudes:
            use_ary = np.abs(ary)
        else:
            use_ary = ary
        expected = np.sum(use_ary * msk.reshape(ary.shape[0], 1), axis=0)
        np.testing.assert_allclose(expected, out, rtol=1e-6)

    @pytest.mark.parametrize('use_amplitudes', [False, True])
    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext, use_amplitudes: bool) -> None:
        """Check that the autotuner runs successfully."""
        maskedsum.MaskedSumTemplate(context, use_amplitudes)
