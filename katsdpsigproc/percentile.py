"""On-device transposition of 2D arrays"""

from __future__ import division
import numpy as np
from . import accel
from . import tune

# #example code to test:
# import numpy as np
# from katsdpsigproc import accel
# from katsdpsigproc.accel import DeviceArray, build
# from katsdpsigproc import percentile as perc5
#
# context = accel.create_some_context(True)
# queue = context.create_command_queue(profile=True)
#
# data=np.abs(np.random.randn(4000,5000)).astype(np.float32)
#
# template=perc5.Percentile5Template(context,max_baselines=5000)
# perc=template.instantiate(queue,data.shape)
# perc.ensure_all_bound()
# perc.buffer('src').set(queue,data)
# perc()
# out=perc.buffer('dest').get(queue)
# np.array_equal(out,np.percentile(data,[0,100,25,75,50],axis=1,interpolation='lower'))

class Percentile5Template(object):
    """Kernel for calculating percentiles of a 2D array of data. 
    5 percentiles [0,100,25,75,50] are calculated per row.

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context for which kernels will be compiled
    max_baselines : maximum number of baselines
    tuning : dict, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - size: number of workitems per workgroup
    """

    autotune_version = 1

    def __init__(self, context, max_baselines, tuning=None):
        self.context = context
        self.max_baselines=max_baselines
        if tuning is None:
            tuning = self.autotune(context, max_baselines)
        self._size = tuning['size']
        self._vt =  accel.divup(max_baselines,tuning['size'])
        program = accel.build(context, "percentile.mako", {
                'size': self._size,
                'vt': self._vt})
        self.kernel = program.get_kernel("percentile5_float")

    @classmethod
    @tune.autotuner(test={'size': 256})
    def autotune(cls, context, max_baselines):
        queue = context.create_tuning_command_queue()
        in_shape = (4096, 4032)
        out_shape = (5, 4096)
        in_data = accel.DeviceArray(context, in_shape, dtype=np.float32)
        out_data = accel.DeviceArray(context, out_shape, dtype=np.float32)
        def generate(size):
            if max_baselines > size*256:
                raise RuntimeError('too many baselines')
            fn = cls(context, max_baselines, {
                'size': size}).instantiate(queue, in_shape)
            fn.bind(src=in_data, dest=out_data)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                size=[32,64,128,256,512,1024])

    def instantiate(self, command_queue, shape):
        return Percentile5(self, command_queue, shape)

class Percentile5(accel.Operation):
    """Concrete instance of :class:`PercentileTemplate`.

    .. rubric:: Slots

    **src**
        Input

    **dest**
        Output
    """
    def __init__(self, template, command_queue, shape):
        super(Percentile5, self).__init__(command_queue)
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
                global_size=(self.template._size, np.int32(src.shape[0])),
                local_size=(self.template._size, 1))

    def parameters(self):
        return {
            'max_baselines': self.template.max_baselines,
            'shape': self.slots['src'].shape
        }
