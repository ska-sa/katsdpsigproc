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

"""Fill device array with a constant value."""

from typing import Tuple, Mapping, Callable, Optional, Any

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore

from . import accel
from . import tune
from .abc import AbstractContext, AbstractCommandQueue


class FillTemplate:
    """Fill a device array with a constant value.

    The pad elements are also filled with this value.

    .. note::
        To fill with zeros, use :meth:`.DeviceArray.zero`

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

        - wgs: number of workitems per workgroup
    """

    def __init__(self, context: AbstractContext, dtype: DTypeLike, ctype: str,
                 tuning: Optional[Mapping[str, Any]] = None) -> None:
        self.context = context
        self.dtype: np.dtype = np.dtype(dtype)
        self.ctype = ctype
        if tuning is None:
            tuning = self.autotune(context, dtype, ctype)
        self.wgs = tuning['wgs']
        self.program = accel.build(context, "fill.mako", {
            'wgs': self.wgs,
            'ctype': ctype
        })

    @classmethod
    @tune.autotuner(test={'wgs': 128})
    def autotune(cls, context: AbstractContext, dtype: DTypeLike, ctype: str) -> Mapping[str, Any]:
        queue = context.create_tuning_command_queue()
        shape = (1048576,)
        data = accel.DeviceArray(context, shape, dtype=dtype)

        def generate(wgs: int) -> Callable[[int], float]:
            fn = cls(context, dtype, ctype, {'wgs': wgs}).instantiate(queue, shape)
            fn.bind(data=data)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate, wgs=[64, 128, 256, 512])

    def instantiate(self, command_queue: AbstractCommandQueue, shape: Tuple[int, ...],
                    allocator: Optional[accel.AbstractAllocator] = None) -> 'Fill':
        return Fill(self, command_queue, shape, allocator)


class Fill(accel.Operation):
    """Concrete instance of :class:`FillTemplate`.

    .. rubric:: Slots

    **data**
        Array to be filled (padding will be filled too)

    Parameters
    ----------
    template
        Operation template
    command_queue
        Command queue for the operation
    shape
        Shape for the data slot
    allocator
        Allocator used to allocate unbound slots
    """

    def __init__(self, template: FillTemplate, command_queue: AbstractCommandQueue,
                 shape: Tuple[int, ...],
                 allocator: Optional[accel.AbstractAllocator] = None) -> None:
        super().__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel("fill")
        self.shape = shape
        self.slots['data'] = accel.IOSlot(shape, self.template.dtype)
        self.value = self.template.dtype.type()

    def set_value(self, value: Any) -> None:
        self.value = self.template.dtype.type(value)

    def _run(self) -> None:
        data = self.buffer('data')

        elements = int(np.product(data.padded_shape))
        global_size = accel.roundup(elements, self.template.wgs)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [data.buffer, np.uint32(elements), self.value],
            global_size=(global_size,),
            local_size=(self.template.wgs,))

    def parameters(self) -> Mapping[str, Any]:
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.slots['data'].shape,     # type: ignore
            'value': self.value
        }
