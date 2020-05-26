import numpy as np
import katsdpsigproc.accel

ctx = katsdpsigproc.accel.create_some_context()
queue = ctx.create_command_queue()
dev = katsdpsigproc.accel.DeviceArray(ctx, (10,), np.float32)
host = dev.empty_like()
host[:] = 1
dev.set(queue, host)
