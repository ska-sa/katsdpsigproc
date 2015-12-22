"""On-device percentile calculation of 2D arrays"""
# see scripts/percentiletest.py for an example

from __future__ import division
import numpy as np
from . import accel
from . import tune

class Percentile5Template(object):
    """Kernel for calculating percentiles of a 2D array of data.
    5 percentiles [0,100,25,75,50] are calculated per row (along columns, independently per row).
    The lower percentile element, rather than a linear interpolation is chosen.
    WARNING: assumes all values are positive.

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context for which kernels will be compiled
    max_columns : int
        Maximum number of columns
    is_amplitude : bool
        If true, the inputs are scalar amplitudes; if false, they are complex
        numbers and the answers are computed on the absolute values
    tuning : dict, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - size: number of workitems per workgroup
    """

    autotune_version = 8

    def __init__(self, context, max_columns, is_amplitude=True, tuning=None):
        self.context = context
        self.max_columns = max_columns
        self.is_amplitude = is_amplitude

        if tuning is None:
            tuning = self.autotune(context, max_columns, is_amplitude)
        self.size = tuning['size']
        self.wgsy = tuning['wgsy']
        self.vt = accel.divup(max_columns, tuning['size'])
        program = accel.build(context, "percentile.mako", {
                'size': self.size,
                'wgsy': self.wgsy,
                'vt': self.vt,
                'is_amplitude': self.is_amplitude})
        self.kernel = program.get_kernel("percentile5_float")

    @classmethod
    @tune.autotuner(test={'size': 64, 'wgsy': 4})
    def autotune(cls, context, max_columns, is_amplitude):
        queue = context.create_tuning_command_queue()
        in_shape = (4096, max_columns)
        out_shape = (5, 4096)
        rs = np.random.RandomState(seed=1)
        if is_amplitude:
            host_data = rs.uniform(size=in_shape).astype(np.float32)
        else:
            host_data = (rs.standard_normal(in_shape) + 1j * rs.standard_normal(in_shape)).astype(np.complex64)
        def generate(size, wgsy):
            if size * wgsy < 32 or size * wgsy > 1024:
                raise RuntimeError('work group size is unnecessarily large or small, skipping')
            if max_columns > size*256:
                raise RuntimeError('too many columns')
            fn = cls(context, max_columns, is_amplitude, {
                'size': size, 'wgsy': wgsy}).instantiate(queue, in_shape)
            inp = fn.slots['src'].allocate(fn.allocator)
            fn.slots['dest'].allocate(fn.allocator)
            inp.set(queue,host_data)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                size=[8, 16, 32, 64, 128, 256, 512, 1024],
                wgsy=[1, 2, 4, 8, 16, 32])

    def instantiate(self, *args, **kwargs):
        return Percentile5(self, *args, **kwargs)

class Percentile5(accel.Operation):
    """Concrete instance of :class:`PercentileTemplate`.
    WARNING: assumes all values are positive when `template.is_amplitude` is `True`.

    .. rubric:: Slots

    **src**
        Input type float32 or complex64
        Shape is number of rows by number of columns, where 5 percentiles are computed along the columns, per row.

    **dest**
        Output type float32
        Shape is (5, number of rows of input)

    Parameters
    ----------
    template : :class:`Percentile5Template`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : 2-element tuple of int
        Shape of the source data
    column_range: 2-element tuple of int, optional
        Half-open interval of columns that will be processed. If not specified, all columns are
        processed.
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, shape, column_range=None, allocator=None):
        super(Percentile5, self).__init__(command_queue, allocator)
        if column_range is None:
            column_range = (0, shape[1])
        if column_range[1] <= column_range[0]:
            raise ValueError('column range is empty')
        if column_range[0] < 0 or column_range[1] > shape[1]:
            raise IndexError('column range is out of range')
        if column_range[1] - column_range[0] > template.max_columns:
            raise ValueError('columns exceeds max_columns')
        self.template = template
        self.shape = shape
        self.column_range = column_range
        src_type = np.float32 if self.template.is_amplitude else np.complex64
        row_dim = accel.Dimension(shape[0], self.template.wgsy)
        col_dim = accel.Dimension(shape[1])
        self.slots['src'] = accel.IOSlot((row_dim, col_dim), src_type)
        self.slots['dest'] = accel.IOSlot((5,row_dim), np.float32)

    def _run(self):
        src = self.buffer('src')
        dest = self.buffer('dest')
        rows_padded = accel.roundup(src.shape[0], self.template.wgsy)
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    src.buffer, dest.buffer,
                    np.int32(src.padded_shape[1]),
                    np.int32(dest.padded_shape[1]),
                    np.int32(self.column_range[0]),
                    np.int32(self.column_range[1] - self.column_range[0])
                ],
                global_size=(self.template.size, rows_padded),
                local_size=(self.template.size, self.template.wgsy))

    def parameters(self):
        return {
            'max_columns': self.template.max_columns,
            'is_amplitude': self.template.is_amplitude,
            'shape': self.slots['src'].shape,
            'column_range': self.column_range
        }
