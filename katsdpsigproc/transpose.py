"""On-device transposition of 2D arrays"""

from __future__ import division
import numpy as np
from . import accel
from . import tune

class TransposeTemplate(object):
    """Kernel for transposing a 2D array of data

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context for which kernels will be compiled
    dtype : numpy dtype
        Type of data elements
    ctype : str
        Type (in C/CUDA, not numpy) of data elements
    tuning : dict, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - block: number of workitems per workgroup in each dimension
        - vtx, vty: elements per workitem in each dimension
    """

    autotune_version = 1

    def __init__(self, context, dtype, ctype, tuning=None):
        self.context = context
        self.dtype = np.dtype(dtype)
        self.ctype = ctype
        if tuning is None:
            tuning = self.autotune(context, dtype, ctype)
        self._block = tuning['block']
        self._tilex = tuning['block'] * tuning['vtx']
        self._tiley = tuning['block'] * tuning['vty']
        program = accel.build(context, "transpose.mako", {
                'block': tuning['block'],
                'vtx': tuning['vtx'],
                'vty': tuning['vty'],
                'ctype': ctype})
        self.kernel = program.get_kernel("transpose")

    @classmethod
    @tune.autotuner(test={'block': 8, 'vtx': 2, 'vty': 3})
    def autotune(cls, context, dtype, ctype):
        queue = context.create_tuning_command_queue()
        in_shape = (2048, 2048)
        out_shape = (2048, 2048)
        in_data = accel.DeviceArray(context, in_shape, dtype=dtype)
        out_data = accel.DeviceArray(context, out_shape, dtype=dtype)
        def generate(block, vtx, vty):
            local_mem = (block * vtx + 1) * (block * vty) * np.dtype(dtype).itemsize
            if local_mem > 32768:
                raise RuntimeError('too much local memory') # Skip configurations using lots of lmem
            fn = cls(context, dtype, ctype, {
                'block': block,
                'vtx': vtx,
                'vty': vty}).instantiate(queue, in_shape)
            fn.bind(src=in_data, dest=out_data)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                block=[4, 8, 16, 32],
                vtx=[1, 2, 3, 4],
                vty=[1, 2, 3, 4])

    def instantiate(self, command_queue, shape):
        return Transpose(self, command_queue, shape)

class Transpose(accel.Operation):
    """Concrete instance of :class:`TransposeTemplate`.

    .. rubric:: Slots

    **src**
        Input

    **dest**
        Output
    """
    def __init__(self, template, command_queue, shape):
        super(Transpose, self).__init__(command_queue)
        self.template = template
        self.shape = shape
        self.slots['src'] = accel.IOSlot(shape, template.dtype)
        self.slots['dest'] = accel.IOSlot((shape[1], shape[0]), template.dtype)

    def _run(self):
        src = self.slots['src'].buffer
        dest = self.slots['dest'].buffer
        # Round up to number of blocks in each dimension
        in_row_tiles = accel.divup(src.shape[0], self.template._tiley)
        in_col_tiles = accel.divup(src.shape[1], self.template._tilex)
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    dest.buffer, src.buffer,
                    np.int32(src.shape[0]), np.int32(src.shape[1]),
                    np.int32(dest.padded_shape[1]),
                    np.int32(src.padded_shape[1])
                ],
                global_size=(in_col_tiles * self.template._block, in_row_tiles * self.template._block),
                local_size=(self.template._block, self.template._block))

    def parameters(self):
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.slots['src'].shape
        }
