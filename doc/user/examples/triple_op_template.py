import numpy as np
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup

SOURCE = """
<%include file="/port.mako"/>

KERNEL void multiply(GLOBAL float *data, float scale) {
    data[get_global_id(0)] *= scale;
}
"""

class MultiplyTemplate:
    WGS = 32

    def __init__(self, context):
        self.program = build(context, '', source=SOURCE)

    def instantiate(self, queue, size, scale):
        return Multiply(self, queue, size, scale)


class Multiply(Operation):
    def __init__(self, template, queue, size, scale):
        super().__init__(queue)
        self.template = template
        self.kernel = template.program.get_kernel('multiply')
        self.scale = np.float32(scale)
        self.slots['data'] = IOSlot((Dimension(size, template.WGS),), np.float32)

    def _run(self):
        data = self.buffer('data')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [data.buffer, self.scale],
            global_size=(roundup(data.shape[0], self.template.WGS),),
            local_size=(self.template.WGS,)
        )


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
