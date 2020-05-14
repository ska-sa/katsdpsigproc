import numpy as np
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup

SOURCE = """
<%include file="/port.mako"/>

KERNEL void multiply(GLOBAL float *data, float scale) {
    data[get_global_id(0)] *= scale;
}
"""

class Multiply(Operation):
    WGS = 32

    def __init__(self, queue, size, scale):
        super().__init__(queue)
        program = build(queue.context, '', source=SOURCE)
        self.kernel = program.get_kernel('multiply')
        self.scale = np.float32(scale)
        self.slots['data'] = IOSlot((Dimension(size, self.WGS),), np.float32)

    def _run(self):
        data = self.buffer('data')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [data.buffer, self.scale],
            global_size=(roundup(data.shape[0], self.WGS),),
            local_size=(self.WGS,)
        )


ctx = katsdpsigproc.accel.create_some_context()
queue = ctx.create_command_queue()
op = Multiply(queue, 50, 3.0)
op.ensure_all_bound()
buf = op.buffer('data')
host = buf.empty_like()
host[:] = np.random.uniform(size=host.shape)
buf.set(queue, host)
op()
buf.get(queue, host)
print(host)
