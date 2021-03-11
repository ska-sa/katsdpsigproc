"""Reduction algorithms."""

from typing import Tuple, Mapping, Callable, Optional, Any, cast
from typing_extensions import TypedDict

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore

from . import accel
from . import tune
from .abc import AbstractContext, AbstractCommandQueue


_TuningDict = TypedDict('_TuningDict', {'wgsx': int, 'wgsy': int})


class HReduceTemplate:
    """Performs reduction along rows in a 2D array.

    Only commutative reduction operators are supported.

    Parameters
    ----------
    context
        Context for which kernels will be compiled
    dtype
        Type of data elements
    ctype
        Type (in C/CUDA, not numpy) of data elements
    op
        C expression to combine two variables, *a* and *b*
    identity
        C expression for an identity value for *op*
    extra_code
        Arbitrary C code to paste in (for use by *op* or *identity*)
    tuning
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx: number of workitems per workgroup per row
        - wgsy: number of rows to handle per workgroup
    """

    def __init__(self, context: AbstractContext, dtype: DTypeLike, ctype: str,
                 op: str, identity: str, extra_code: str = '',
                 tuning: Optional[_TuningDict] = None) -> None:
        self.context = context
        self.dtype = dtype
        self.ctype = ctype
        if tuning is None:
            tuning = self.autotune(context, dtype, ctype, op, identity, extra_code)
        self.wgsx = tuning['wgsx']
        self.wgsy = tuning['wgsy']
        self.op = op
        self.identity = identity
        self.extra_code = extra_code
        self.program = accel.build(context, 'hreduce.mako', {
            'wgsx': self.wgsx, 'wgsy': self.wgsy, 'type': self.ctype,
            'op': op, 'identity': identity, 'extra_code': extra_code})

    @classmethod
    @tune.autotuner(test={'wgsx': 64, 'wgsy': 4})
    def autotune(cls, context: AbstractContext, dtype: DTypeLike, ctype: str,
                 op: str, identity: str, extra_code: str) -> _TuningDict:
        queue = context.create_tuning_command_queue()
        shape = (2048, 1024)
        src = accel.DeviceArray(context, shape, dtype=dtype)
        dest = accel.DeviceArray(context, (shape[0],), dtype=dtype)

        def generate(wgsx: int, wgsy: int) -> Callable[[int], float]:
            wgs = wgsx * wgsy
            if wgs < 32 or wgs > 1024:
                raise RuntimeError(f'Skipping work group size {wgsx}x{wgsy}')
            template = cls(context, dtype, ctype, op, identity, extra_code,
                           {'wgsx': wgsx, 'wgsy': wgsy})
            fn = template.instantiate(queue, shape)
            fn.bind(src=src, dest=dest)
            fn.ensure_all_bound()
            return tune.make_measure(queue, fn)

        return cast(_TuningDict, tune.autotune(generate,
                                               wgsx=[32, 64, 128],
                                               wgsy=[1, 2, 4, 8, 16]))

    def instantiate(self, command_queue: AbstractCommandQueue, shape: Tuple[int, int],
                    column_range: Optional[Tuple[int, int]] = None,
                    allocator: Optional[accel.AbstractAllocator] = None) -> 'HReduce':
        return HReduce(self, command_queue, shape, column_range, allocator)


class HReduce(accel.Operation):
    """Concrete instance of :class:`HReduceTemplate`.

    In each row, the elements in the specified column range are reduced using
    the reduction operator supplied to the template.

    .. rubric:: Slots

    **src** : *rows* × *columns*
        Input data
    **dest** : *rows*
        Output reductions

    Parameters
    ----------
    template
        Operation template
    command_queue
        Command queue for the operation
    shape
        Shape for the source slot
    column_range
        Half-open range of columns to reduce (defaults to the entire array)
    allocator
        Allocator used to allocate unbound slots
    """

    def __init__(self, template: HReduceTemplate, command_queue: AbstractCommandQueue,
                 shape: Tuple[int, int], column_range: Optional[Tuple[int, int]] = None,
                 allocator: Optional[accel.AbstractAllocator] = None) -> None:
        if len(shape) != 2:
            raise ValueError('shape must be 2-dimensional')
        if column_range is None:
            column_range = (0, shape[1])
        if column_range[0] < 0 or column_range[1] > shape[1]:
            raise ValueError('column range overflows the array')
        if column_range[0] >= column_range[1]:
            raise ValueError('column range is empty')

        super().__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('hreduce')
        self.column_range = column_range
        self.slots['src'] = accel.IOSlot(
            (accel.Dimension(shape[0], self.template.wgsy), shape[1]),
            self.template.dtype)
        self.slots['dest'] = accel.IOSlot(
            (accel.Dimension(shape[0], self.template.wgsy),),
            self.template.dtype)

    def _run(self) -> None:
        src = self.buffer('src')
        dest = self.buffer('dest')
        rows_padded = accel.roundup(src.shape[0], self.template.wgsy)
        n_columns = self.column_range[1] - self.column_range[0]
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                src.buffer, dest.buffer,
                np.int32(self.column_range[0]), np.int32(n_columns),
                np.int32(src.padded_shape[1])
            ],
            global_size=(self.template.wgsx, rows_padded),
            local_size=(self.template.wgsx, self.template.wgsy))

    def parameters(self) -> Mapping[str, Any]:
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.slots['src'].shape,    # type: ignore
            'column_range': self.column_range,
            'op': self.template.op,
            'identity': self.template.identity,
            'extra_code': self.template.extra_code
        }
