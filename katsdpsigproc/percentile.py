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
        maximum number of columns
    tuning : dict, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - size: number of workitems per workgroup
    """

    autotune_version = 4

    def __init__(self, context, max_columns, tuning=None):
        self.context = context
        self.max_columns=max_columns
        
        if tuning is None:
            tuning = self.autotune(context, max_columns)
        self.size = tuning['size']
        self.vt =  accel.divup(max_columns, tuning['size'])
        program = accel.build(context, "percentile.mako", {
                'size': self.size,
                'vt': self.vt})
        self.kernel = program.get_kernel("percentile5_float")

    @classmethod
    @tune.autotuner(test={'size': 256})
    def autotune(cls, context, max_columns):
        queue = context.create_tuning_command_queue()
        in_shape = (4096, max_columns)
        out_shape = (5, 4096)
        rs = np.random.RandomState(seed=1)
        host_data = rs.uniform(size=in_shape).astype(np.float32)
        def generate(size):
            if max_columns > size*256:
                raise RuntimeError('too many columns')
            fn = cls(context, max_columns, {
                'size': size}).instantiate(queue, in_shape)
            inp = fn.slots['src'].allocate(context)
            fn.slots['dest'].allocate(context)
            inp.set(queue,host_data)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                size=[32, 64, 128, 256, 512, 1024])

    def instantiate(self, command_queue, shape):
        return Percentile5(self, command_queue, shape)

class Percentile5(accel.Operation):
    """Concrete instance of :class:`PercentileTemplate`.
    WARNING: assumes all values are positive.

    .. rubric:: Slots

    **src**
        Input type float32
        Shape is number of rows by number of columns, where 5 percentiles are computed along the columns, per row.

    **dest**
        Output type float32
        Shape is (5, number of rows of input)
        
    """
    def __init__(self, template, command_queue, shape):
        super(Percentile5, self).__init__(command_queue)
        if shape[1] > template.max_columns:
            raise ValueError('columns exceeds max_columns')
        self.template = template
        self.shape = shape
        self.slots['src'] = accel.IOSlot(shape, np.float32)
        self.slots['dest'] = accel.IOSlot((5,shape[0]), np.float32)

    def _run(self):
        src = self.buffer('src')
        dest = self.buffer('dest')
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    src.buffer, dest.buffer,
                    np.int32(src.padded_shape[1]), np.int32(dest.padded_shape[1]), np.int32(src.shape[1])
                ],
                global_size=(self.template.size, np.int32(src.shape[0])),
                local_size=(self.template.size, 1))

    def parameters(self):
        return {
            'max_columns': self.template.max_columns,
            'shape': self.slots['src'].shape
        }
