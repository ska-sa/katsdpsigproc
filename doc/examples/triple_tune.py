import logging
import numpy as np
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup
import katsdpsigproc.tune

SOURCE = """
<%include file="/port.mako"/>

KERNEL void multiply(GLOBAL float *data, float scale) {
    data[get_global_id(0)] *= scale;
}
"""

class MultiplyTemplate:
    def __init__(self, context, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.wgs = tuning['wgs']
        self.program = build(context, '', source=SOURCE)

    @classmethod
    @katsdpsigproc.tune.autotuner(test={'wgs': 32})
    def autotune(cls, context):
        queue = context.create_tuning_command_queue()
        size = 1048576

        def generate(wgs):
            fn = cls(context, {'wgs': wgs}).instantiate(queue, size, 1)
            fn.ensure_all_bound()
            fn.buffer('data').zero(queue)
            return katsdpsigproc.tune.make_measure(queue, fn)

        return katsdpsigproc.tune.autotune(generate, wgs=[32, 64, 128, 256])

    def instantiate(self, queue, size, scale):
        return Multiply(self, queue, size, scale)


class Multiply(Operation):
    def __init__(self, template, queue, size, scale):
        super().__init__(queue)
        self.template = template
        self.kernel = template.program.get_kernel('multiply')
        self.scale = np.float32(scale)
        self.slots['data'] = IOSlot((Dimension(size, self.template.wgs),), np.float32)

    def _run(self):
        data = self.buffer('data')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [data.buffer, self.scale],
            global_size=(roundup(data.shape[0], self.template.wgs),),
            local_size=(self.template.wgs,)
        )


logging.basicConfig(level='DEBUG')
ctx = katsdpsigproc.accel.create_some_context()
queue = ctx.create_command_queue()
op_template = MultiplyTemplate(ctx)
op = op_template.instantiate(queue, 50, 3.0)
op.ensure_all_bound()
buf = op.buffer('data')
host = buf.empty_like()
host[:] = np.random.uniform(size=host.shape)
buf.set(queue, host)
op()
buf.get(queue, host)
print(host)
