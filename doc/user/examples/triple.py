import numpy as np
import katsdpsigproc.accel

SOURCE = """
__kernel void multiply(__global float *data, float scale) {
    data[get_global_id(0)] *= scale;
}
"""

ctx = katsdpsigproc.accel.create_some_context(device_filter=lambda x: not x.is_cuda)
queue = ctx.create_command_queue()
buf = katsdpsigproc.accel.DeviceArray(ctx, (50,), np.float32, (64,))
host = buf.empty_like()
host[:] = np.random.uniform(size=host.shape)
buf.set(queue, host)

program = ctx.compile(SOURCE)
kernel = program.get_kernel('multiply')
queue.enqueue_kernel(
    kernel,
    [buf.buffer, np.float32(3.0)],
    global_size=buf.padded_shape,
    local_size=(32,)
)
buf.get(queue, host)
print(host)
