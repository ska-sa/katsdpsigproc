#coding: utf-8
"""Reduction algorithms"""

from __future__ import division, print_function, absolute_import
import numpy as np
from . import accel
from . import tune

class HReduceTemplate(object):
    """Performs reduction along rows in a 2D array. Only commutative reduction
    operators are supported.

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context for which kernels will be compiled
    dtype : numpy dtype
        Type of data elements
    ctype : str
        Type (in C/CUDA, not numpy) of data elements
    op : str
        C expression to combine two variables, *a* and *b*
    identity : str
        C expression for an identity value for *op*
    extra_code : str, optional
        Arbitrary C code to paste in (for use by *op* or *identity*)

    tuning : dict, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx: number of workitems per workgroup per row
        - wgsy: number of rows to handle per workgroup
    """
    def __init__(self, context, dtype, ctype, op, identity, extra_code='', tuning=None):
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
    def autotune(cls, context, dtype, ctype, op, identity, extra_code):
        queue = context.create_tuning_command_queue()
        shape = (2048, 1024)
        src = accel.DeviceArray(context, shape, dtype=dtype)
        dest = accel.DeviceArray(context, (shape[0],), dtype=dtype)
        def generate(**kwargs):
            wgs = kwargs['wgsx'] * kwargs['wgsy']
            if wgs < 32 or wgs > 1024:
                raise RuntimeError('Skipping work group size {wgsx}x{wgsy}'.format(**kwargs))
            fn = cls(context, dtype, ctype, op, identity, extra_code, kwargs).instantiate(queue, shape)
            fn.bind(src=src, dest=dest)
            fn.ensure_all_bound()
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                wgsx=[32, 64, 128],
                wgsy=[1, 2, 4, 8, 16])

    def instantiate(self, *args, **kwargs):
        return HReduce(self, *args, **kwargs)

class HReduce(accel.Operation):
    """
    Concrete instance of :class:`HReduceTemplate`. In each row, the
    elements in the specified column range are reduced using the reduction
    operator supplied to the template.

    .. rubric:: Slots

    **src** : *rows* × *columns*
        Input data
    **dest** : *rows*
        Output reductions

    Parameters
    ----------
    template : :class:`FillTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : 2-tuple of int
        Shape for the source slot
    column_range : 2-tuple of int, optional
        Half-open range of columns to reduce (defaults to the entire array)
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, shape, column_range=None, allocator=None):
        if len(shape) != 2:
            raise ValueError('shape must be 2-dimensional')
        if column_range is None:
            column_range = (0, shape[1])
        if column_range[0] < 0 or column_range[1] > shape[1]:
            raise ValueError('column range overflows the array')
        if column_range[0] >= column_range[1]:
            raise ValueError('column range is empty')

        super(HReduce, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('hreduce')
        self.column_range = column_range
        self.slots['src'] = accel.IOSlot(
                (accel.Dimension(shape[0], self.template.wgsy), shape[1]),
                self.template.dtype)
        self.slots['dest'] = accel.IOSlot(
                (accel.Dimension(shape[0], self.template.wgsy),),
                self.template.dtype)

    def _run(self):
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

    def parameters(self):
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.slots['src'].shape,
            'column_range': self.column_range,
            'op': self.template.op,
            'identity': self.template.identity,
            'extra_code': self.template.extra_code
        }
