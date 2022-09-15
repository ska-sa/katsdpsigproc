import numpy as np
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup

pytest_plugins = ["katsdpsigproc.pytest_plugin"]

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


def test_multiply(context, command_queue):
    size = 53
    template = MultiplyTemplate(context)
    op = template.instantiate(command_queue, size, 4.0)
    op.ensure_all_bound()
    src = np.random.uniform(size=size).astype(np.float32)
    op.buffer('data').set(command_queue, src)
    op()
    dst = op.buffer('data').get(command_queue)
    np.testing.assert_array_equal(dst, src * 4.0)
