"""On-device percentile calculation of 2D arrays"""
# see scripts/maskedsumtest.py for an example

from __future__ import division, print_function, absolute_import
import numpy as np
from . import accel
from . import tune


class MaskedSumTemplate(object):
    """Kernel for calculating masked sums of a 2D array of data.
    masked sums are calculated per column (along rows, independently per column).

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context for which kernels will be compiled
    use_amplitudes : bool, optional
        If true, the amplitudes of the inputs rather than the inputs
        themselves will be summed.
    tuning : dict, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - size: number of workitems per workgroup
    """

    autotune_version = 1

    def __init__(self, context, use_amplitudes=False, tuning=None):
        self.context = context
        self.use_amplitudes = use_amplitudes
        if tuning is None:
            tuning = self.autotune(context, use_amplitudes)
        self.size = tuning['size']
        self.program = accel.build(context, "maskedsum.mako", {
                'size': self.size,
                'use_amplitudes': use_amplitudes})

    @classmethod
    @tune.autotuner(test={'size': 256})
    def autotune(cls, context, use_amplitudes):
        queue = context.create_tuning_command_queue()
        columns = 5000
        in_shape = (4096, columns)
        out_shape = (columns,)
        rs = np.random.RandomState(seed=1)
        host_data = rs.uniform(size=(in_shape[0], in_shape[1], 2)).astype(np.float32).view(dtype=np.complex64)[...,0]
        host_mask = np.ones((in_shape[0],)).astype(np.float32)
        def generate(size):
            fn = cls(context, use_amplitudes, {
                'size': size}).instantiate(queue, in_shape)
            inp = fn.slots['src'].allocate(fn.allocator)
            msk = fn.slots['mask'].allocate(fn.allocator)
            fn.slots['dest'].allocate(fn.allocator)
            inp.set(queue, host_data)
            msk.set(queue, host_mask)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                size=[32, 64, 128, 256, 512, 1024])

    def instantiate(self, *args, **kwargs):
        return MaskedSum(self, *args, **kwargs)

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
    def __init__(self, template, command_queue, shape, allocator=None):
        super(MaskedSum, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel("maskedsum_float")
        self.shape = shape
        self.slots['src'] = accel.IOSlot((shape[0], accel.Dimension(shape[1], template.size)), np.complex64)
        self.slots['mask'] = accel.IOSlot((shape[0],), np.float32)
        self.slots['dest'] = accel.IOSlot(
            (accel.Dimension(shape[1], template.size),),
            np.float32 if template.use_amplitudes else np.complex64)

    def _run(self):
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

    def parameters(self):
        return {
            'shape': self.slots['src'].shape,
            'use_amplitudes': self.template.use_amplitudes
        }
