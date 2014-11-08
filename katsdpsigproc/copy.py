"""Copy one device array to another."""

import numpy as np
from . import accel
from . import tune

class CopyTemplate(object):
    """
    Copies one device array to another. Any padding elements are also copied.

    This class should *not* be used for production code. It uses a kernel to
    do the copy, rather than hardware copy engines, and so is slower. It is
    intended only for benchmarking kernel memory access patterns.

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

        - wgs: number of workitems per workgroup
    """

    def __init__(self, context, dtype, ctype, tuning=None):
        self.context = context
        self.dtype = np.dtype(dtype)
        self.ctype = ctype
        if tuning is None:
            tuning = self.autotune(context, dtype, ctype)
        self.wgs = tuning['wgs']
        program = accel.build(context, "copy.mako", {
                'wgs': self.wgs,
                'ctype': ctype})
        self.kernel = program.get_kernel("copy")

    @classmethod
    @tune.autotuner(test={'wgs': 128})
    def autotune(cls, context, dtype, ctype):
        queue = context.create_tuning_command_queue()
        shape = (1048576,)
        src = accel.DeviceArray(context, shape, dtype=dtype)
        dest = accel.DeviceArray(context, shape, dtype=dtype)
        def generate(wgs):
            fn = cls(context, dtype, ctype, {'wgs': wgs}).instantiate(queue, shape)
            fn.bind(src=src, dest=dest)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate, wgs=[64, 128, 256, 512])

    def instantiate(self, command_queue, shape):
        return Copy(self, command_queue, shape)

class Copy(accel.Operation):
    """Concrete instance of :class:`CopyTemplate`.

    This class should *not* be used for production code. See :class:`CopyTemplate` for
    details.

    .. rubric:: Slots

    **src**
        Source array
    **dest**
        Destination array

    Parameters
    ----------
    template : :class:`CopyTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    shape : tuple of int
        Shape for the data slot
    """

    def __init__(self, template, command_queue, shape):
        super(Copy, self).__init__(command_queue)
        self.template = template
        self.shape = shape
        dims = [accel.Dimension(size) for size in shape]
        self.slots['src'] = accel.IOSlot(dims, self.template.dtype)
        self.slots['dest'] = accel.IOSlot(dims, self.template.dtype)

    def _run(self):
        src = self.buffer('src')
        dest = self.buffer('dest')

        elements = np.product(src.padded_shape)
        global_size = accel.roundup(elements, self.template.wgs)
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [dest.buffer, src.buffer, np.uint32(elements)],
                global_size=(global_size,),
                local_size=(self.template.wgs,))

    def parameters(self):
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.shape
        }
