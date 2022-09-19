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

"""Perform on-device percentile calculation of 2D arrays."""
# see scripts/percentiletest.py for an example

from typing import Tuple, Mapping, Callable, Optional, Any, cast
from typing_extensions import TypedDict

import numpy as np

from . import accel
from . import tune
from .abc import AbstractContext, AbstractCommandQueue


_TuningDict = TypedDict('_TuningDict', {'size': int, 'wgsy': int})


class Percentile5Template:
    """Kernel for calculating percentiles of a 2D array of data.

    5 percentiles [0,100,25,75,50] are calculated per row (along columns, independently per row).
    The lower percentile element, rather than a linear interpolation is chosen.
    WARNING: assumes all values are positive.

    Parameters
    ----------
    context
        Context for which kernels will be compiled
    max_columns
        Maximum number of columns
    is_amplitude
        If true, the inputs are scalar amplitudes; if false, they are complex
        numbers and the answers are computed on the absolute values
    tuning
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - size: number of workitems per workgroup along each row
        - wgsy: number of workitems per workgroup along each column
    """

    autotune_version = 8

    def __init__(self, context: AbstractContext, max_columns: int,
                 is_amplitude: bool = True, tuning: Optional[_TuningDict] = None) -> None:
        self.context = context
        self.max_columns = max_columns
        self.is_amplitude = is_amplitude

        if tuning is None:
            tuning = self.autotune(context, max_columns, is_amplitude)
        self.size = tuning['size']
        self.wgsy = tuning['wgsy']
        self.vt = accel.divup(max_columns, tuning['size'])
        self.program = accel.build(context, "percentile.mako", {
            'size': self.size,
            'wgsy': self.wgsy,
            'vt': self.vt,
            'is_amplitude': self.is_amplitude
        })

    @classmethod
    @tune.autotuner(test={'size': 64, 'wgsy': 4})
    def autotune(cls, context: AbstractContext, max_columns: int,
                 is_amplitude: bool) -> _TuningDict:
        queue = context.create_tuning_command_queue()
        in_shape = (4096, max_columns)
        rs = np.random.RandomState(seed=1)
        if is_amplitude:
            host_data: np.ndarray = rs.uniform(size=in_shape).astype(np.float32)
        else:
            host_data = rs.standard_normal(in_shape) + 1j * rs.standard_normal(in_shape)
            host_data = host_data.astype(np.complex64)

        def generate(size: int, wgsy: int) -> Callable[[int], float]:
            if size * wgsy < 32 or size * wgsy > 1024:
                raise RuntimeError('work group size is unnecessarily large or small, skipping')
            if max_columns > size * 256:
                raise RuntimeError('too many columns')
            fn = cls(context, max_columns, is_amplitude, {
                'size': size, 'wgsy': wgsy}).instantiate(queue, in_shape)
            inp = fn.slots['src'].allocate(fn.allocator)
            fn.slots['dest'].allocate(fn.allocator)
            inp.set(queue, host_data)
            return tune.make_measure(queue, fn)

        return cast(_TuningDict, tune.autotune(generate,
                                               size=[8, 16, 32, 64, 128, 256, 512, 1024],
                                               wgsy=[1, 2, 4, 8, 16, 32]))

    def instantiate(self, command_queue: AbstractCommandQueue,
                    shape: Tuple[int, int],
                    column_range: Optional[Tuple[int, int]] = None,
                    allocator: Optional[accel.AbstractAllocator] = None) -> 'Percentile5':
        return Percentile5(self, command_queue, shape, column_range, allocator)


class Percentile5(accel.Operation):
    """Concrete instance of :class:`PercentileTemplate`.

    .. warning::

       Assumes all values are positive when `template.is_amplitude` is `True`.

    .. rubric:: Slots

    **src**
        Input type float32 or complex64.
        Shape is number of rows by number of columns, where 5 percentiles
        are computed along the columns, per row.

    **dest**
        Output type float32.
        Shape is (5, number of rows of input)

    Parameters
    ----------
    template
        Operation template
    command_queue
        Command queue for the operation
    shape
        Shape of the source data
    column_range:
        Half-open interval of columns that will be processed. If not specified, all columns are
        processed.
    allocator
        Allocator used to allocate unbound slots
    """

    def __init__(self, template: Percentile5Template, command_queue: AbstractCommandQueue,
                 shape: Tuple[int, int], column_range: Optional[Tuple[int, int]],
                 allocator: Optional[accel.AbstractAllocator] = None) -> None:
        super().__init__(command_queue, allocator)
        if column_range is None:
            column_range = (0, shape[1])
        if column_range[1] <= column_range[0]:
            raise ValueError('column range is empty')
        if column_range[0] < 0 or column_range[1] > shape[1]:
            raise IndexError('column range is out of range')
        if column_range[1] - column_range[0] > template.max_columns:
            raise ValueError('columns exceeds max_columns')
        self.template = template
        self.kernel = template.program.get_kernel("percentile5_float")
        self.shape = shape
        self.column_range = column_range
        src_type = np.float32 if self.template.is_amplitude else np.complex64
        row_dim = accel.Dimension(shape[0], self.template.wgsy)
        col_dim = accel.Dimension(shape[1])
        self.slots['src'] = accel.IOSlot((row_dim, col_dim), src_type)
        self.slots['dest'] = accel.IOSlot((5, row_dim), np.float32)

    def _run(self) -> None:
        src = self.buffer('src')
        dest = self.buffer('dest')
        rows_padded = accel.roundup(src.shape[0], self.template.wgsy)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                src.buffer, dest.buffer,
                np.int32(src.padded_shape[1]),
                np.int32(dest.padded_shape[1]),
                np.int32(self.column_range[0]),
                np.int32(self.column_range[1] - self.column_range[0])
            ],
            global_size=(self.template.size, rows_padded),
            local_size=(self.template.size, self.template.wgsy))

    def parameters(self) -> Mapping[str, Any]:
        return {
            'max_columns': self.template.max_columns,
            'is_amplitude': self.template.is_amplitude,
            'shape': self.slots['src'].shape,      # type: ignore
            'column_range': self.column_range
        }
