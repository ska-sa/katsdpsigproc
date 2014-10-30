#!/usr/bin/env python
from katsdpsigproc import accel
import numpy as np

# dtype = np.complex64
# ctype = 'float2'
dtype = np.float32
ctype = 'float'

R = 10240
C = 8256
ctx = accel.create_some_context(True)
queue = ctx.create_tuning_command_queue()
rm = accel.DeviceArray(ctx, (R, C), dtype)
cm = accel.DeviceArray(ctx, (C, R), dtype)
transpose = accel.Transpose(queue, dtype, ctype)
transpose(rm, cm)
queue.start_tuning()
transpose(rm, cm)
print queue.stop_tuning()
queue.start_tuning()
transpose(cm, rm)
print queue.stop_tuning()
