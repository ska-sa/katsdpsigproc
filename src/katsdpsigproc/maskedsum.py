################################################################################
# Copyright (c) 2014-2019, National Research Foundation (SARAO)
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

"""Perform on-device percentile calculation of 2D arrays."""
# see scripts/maskedsumtest.py for an example

from typing import Tuple, Mapping, Callable, Optional, Any, cast
from typing_extensions import TypedDict

import numpy as np

from . import accel
from . import tune
from .abc import AbstractContext, AbstractCommandQueue


_TuningDict = TypedDict('_TuningDict', {'size': int})


class MaskedSumTemplate:
    """Kernel for calculating masked sums of a 2D array of data.

    Masked sums are calculated per column (along rows, independently per column).

    Parameters
    ----------
    context
        Context for which kernels will be compiled
    use_amplitudes
        If true, the amplitudes of the inputs rather than the inputs
        themselves will be summed.
    tuning
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - size: number of workitems per workgroup
    """

    autotune_version = 1

    def __init__(self, context: AbstractContext, use_amplitudes: bool = False,
                 tuning: Optional[_TuningDict] = None) -> None:
        self.context = context
        self.use_amplitudes = use_amplitudes
        if tuning is None:
            tuning = self.autotune(context, use_amplitudes)
        self.size = tuning['size']
        self.program = accel.build(context, "maskedsum.mako", {
            'size': self.size,
            'use_amplitudes': use_amplitudes
        })

    @classmethod
    @tune.autotuner(test={'size': 256})
    def autotune(cls, context: AbstractContext, use_amplitudes: bool) -> _TuningDict:
        queue = context.create_tuning_command_queue()
        columns = 5000
        in_shape = (4096, columns)
        rs = np.random.RandomState(seed=1)
        host_data = rs.uniform(size=(in_shape[0], in_shape[1], 2)).astype(np.float32)
        host_data = host_data.view(dtype=np.complex64)[..., 0]
        host_mask = np.ones((in_shape[0],)).astype(np.float32)

        def generate(size: int) -> Callable[[int], float]:
            fn = cls(context, use_amplitudes, {
                'size': size}).instantiate(queue, in_shape)
            inp = fn.slots['src'].allocate(fn.allocator)
            msk = fn.slots['mask'].allocate(fn.allocator)
            fn.slots['dest'].allocate(fn.allocator)
            inp.set(queue, host_data)
            msk.set(queue, host_mask)
            return tune.make_measure(queue, fn)

        return cast(_TuningDict, tune.autotune(generate, size=[32, 64, 128, 256, 512, 1024]))

    def instantiate(self, command_queue: AbstractCommandQueue, shape: Tuple[int, int],
                    allocator: Optional[accel.AbstractAllocator] = None) -> 'MaskedSum':
        return MaskedSum(self, command_queue, shape, allocator)


class MaskedSum(accel.Operation):
    """Concrete instance of :class:`MaskedSumTemplate`.

    .. rubric:: Slots

    **src**
        Input type complex64
        Shape is number of rows by number of columns, masked sum is calculated
        along the rows, independently per column.

    **mask**
        Input type float32
        Shape is (number of rows of input).

    **dest**
        Output type complex64
        Shape is (number of columns of input)
    """

    def __init__(self, template: MaskedSumTemplate, command_queue: AbstractCommandQueue,
                 shape: Tuple[int, int],
                 allocator: Optional[accel.AbstractAllocator] = None) -> None:
        super().__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel("maskedsum_float")
        self.shape = shape
        self.slots['src'] = accel.IOSlot(
            (shape[0], accel.Dimension(shape[1], template.size)),
            np.complex64)
        self.slots['mask'] = accel.IOSlot((shape[0],), np.float32)
        self.slots['dest'] = accel.IOSlot(
            (accel.Dimension(shape[1], template.size),),
            np.float32 if template.use_amplitudes else np.complex64)

    def _run(self) -> None:
        src = self.buffer('src')
        mask = self.buffer('mask')
        dest = self.buffer('dest')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                src.buffer, mask.buffer, dest.buffer,
                np.int32(src.padded_shape[1]), np.int32(src.shape[0])
            ],
            global_size=(accel.roundup(src.shape[1], self.template.size),),
            local_size=(self.template.size, ))

    def parameters(self) -> Mapping[str, Any]:
        return {
            'shape': self.slots['src'].shape,       # type: ignore
            'use_amplitudes': self.template.use_amplitudes
        }
