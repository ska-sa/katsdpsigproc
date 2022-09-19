################################################################################
# Copyright (c) 2014-2021, National Research Foundation (SARAO)
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

"""Transpose 2D arrays on a device."""

from typing import Tuple, Optional, Mapping, Callable, Any, cast
from typing_extensions import TypedDict

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore

from . import accel
from . import tune
from .abc import AbstractContext, AbstractCommandQueue


_TuningDict = TypedDict('_TuningDict', {'block': int, 'vtx': int, 'vty': int})


class TransposeTemplate:
    """Kernel for transposing a 2D array of data.

    Parameters
    ----------
    context
        Context for which kernels will be compiled
    dtype
        Type of data elements
    ctype
        Type (in C/CUDA, not numpy) of data elements
    tuning
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - block: number of workitems per workgroup in each dimension
        - vtx, vty: elements per workitem in each dimension
    """

    autotune_version = 1

    def __init__(self, context: AbstractContext, dtype: DTypeLike, ctype: str,
                 tuning: Optional[_TuningDict] = None) -> None:
        self.context = context
        self.dtype: np.dtype = np.dtype(dtype)
        self.ctype = ctype
        if tuning is None:
            tuning = self.autotune(context, dtype, ctype)
        self._block = tuning['block']
        self._tilex = tuning['block'] * tuning['vtx']
        self._tiley = tuning['block'] * tuning['vty']
        self.program = accel.build(context, "transpose.mako", {
            'block': tuning['block'],
            'vtx': tuning['vtx'],
            'vty': tuning['vty'],
            'ctype': ctype
        })

    @classmethod
    @tune.autotuner(test={'block': 8, 'vtx': 2, 'vty': 3})
    def autotune(cls, context: AbstractContext, dtype: DTypeLike, ctype: str) -> _TuningDict:
        queue = context.create_tuning_command_queue()
        in_shape = (2048, 2048)
        out_shape = (2048, 2048)
        in_data = accel.DeviceArray(context, in_shape, dtype=dtype)
        out_data = accel.DeviceArray(context, out_shape, dtype=dtype)

        def generate(block: int, vtx: int, vty: int) -> Optional[Callable[[int], float]]:
            local_mem = (block * vtx + 1) * (block * vty) * np.dtype(dtype).itemsize
            if local_mem > 32768:
                # Skip configurations using lots of lmem
                raise RuntimeError('too much local memory')
            fn = cls(context, dtype, ctype, {
                'block': block,
                'vtx': vtx,
                'vty': vty}).instantiate(queue, in_shape)
            fn.bind(src=in_data, dest=out_data)
            return tune.make_measure(queue, fn)

        return cast(_TuningDict, tune.autotune(generate,
                                               block=[4, 8, 16, 32],
                                               vtx=[1, 2, 3, 4],
                                               vty=[1, 2, 3, 4]))

    def instantiate(self, command_queue: AbstractCommandQueue, shape: Tuple[int, int],
                    allocator: Optional[accel.AbstractAllocator] = None) -> 'Transpose':
        return Transpose(self, command_queue, shape, allocator)


class Transpose(accel.Operation):
    """Concrete instance of :class:`TransposeTemplate`.

    .. rubric:: Slots

    **src**
        Input

    **dest**
        Output
    """

    def __init__(self, template: TransposeTemplate, command_queue: AbstractCommandQueue,
                 shape: Tuple[int, int], allocator: Optional[accel.AbstractAllocator] = None):
        super().__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel("transpose")
        self.shape = shape
        self.slots['src'] = accel.IOSlot(shape, template.dtype)
        self.slots['dest'] = accel.IOSlot((shape[1], shape[0]), template.dtype)

    def _run(self) -> None:
        src = self.buffer('src')
        dest = self.buffer('dest')
        # Round up to number of blocks in each dimension
        in_row_tiles = accel.divup(src.shape[0], self.template._tiley)
        in_col_tiles = accel.divup(src.shape[1], self.template._tilex)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                dest.buffer, src.buffer,
                np.int32(src.shape[0]), np.int32(src.shape[1]),
                np.int32(dest.padded_shape[1]),
                np.int32(src.padded_shape[1])
            ],
            global_size=(in_col_tiles * self.template._block, in_row_tiles * self.template._block),
            local_size=(self.template._block, self.template._block))

    def parameters(self) -> Mapping[str, Any]:
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.slots['src'].shape      # type: ignore
        }
