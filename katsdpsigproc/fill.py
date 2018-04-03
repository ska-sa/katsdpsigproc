"""Fill device array with a constant value

.. include:: macros.rst
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from . import accel
from . import tune


class FillTemplate(object):
    """
    Fills a device array with a constant value. The pad elements are also
    filled with this value.

    .. note::
        To fill with zeros, use :meth:`katsdpsigproc.DeviceArray.zero`

    Parameters
    ----------
    context : |Context|
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
        self.program = accel.build(context, "fill.mako", {
            'wgs': self.wgs,
            'ctype': ctype
        })

    @classmethod
    @tune.autotuner(test={'wgs': 128})
    def autotune(cls, context, dtype, ctype):
        queue = context.create_tuning_command_queue()
        shape = (1048576,)
        data = accel.DeviceArray(context, shape, dtype=dtype)

        def generate(wgs):
            fn = cls(context, dtype, ctype, {'wgs': wgs}).instantiate(queue, shape)
            fn.bind(data=data)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate, wgs=[64, 128, 256, 512])

    def instantiate(self, *args, **kwargs):
        return Fill(self, *args, **kwargs)


class Fill(accel.Operation):
    """Concrete instance of :class:`FillTemplate`.

    .. rubric:: Slots

    **data**
        Array to be filled (padding will be filled too)

    Parameters
    ----------
    template : :class:`FillTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    shape : tuple of int
        Shape for the data slot
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    def __init__(self, template, command_queue, shape, allocator=None):
        super(Fill, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel("fill")
        self.shape = shape
        self.slots['data'] = accel.IOSlot(shape, self.template.dtype)
        self.value = self.template.dtype.type()

    def set_value(self, value):
        self.value = self.template.dtype.type(value)

    def _run(self):
        data = self.buffer('data')

        elements = int(np.product(data.padded_shape))
        global_size = accel.roundup(elements, self.template.wgs)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [data.buffer, np.uint32(elements), self.value],
            global_size=(global_size,),
            local_size=(self.template.wgs,))

    def parameters(self):
        return {
            'dtype': self.template.dtype,
            'ctype': self.template.ctype,
            'shape': self.slots['data'].shape,
            'value': self.value
        }
