#!/usr/bin/env python
from katsdpsigproc import accel, transpose
import numpy as np

dtype = np.complex64
ctype = 'float2'
# dtype = np.float32
# ctype = 'float'

R = 3072
C = 8320
ctx = accel.create_some_context(True)
template = transpose.TransposeTemplate(ctx, dtype, ctype)

queue = ctx.create_tuning_command_queue()
proc = template.instantiate(queue, (R, C))
proc.ensure_all_bound()
rm = proc.buffer('src')
cm = proc.buffer('dest')
proc()  # Warmup
for i in range(4):
    queue.start_tuning()
    proc()
    print(queue.stop_tuning())
