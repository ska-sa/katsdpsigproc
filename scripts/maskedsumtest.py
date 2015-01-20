#!/usr/bin/env python
#for nosetest: nosetests katsdpsigproc.test.test_maskedsum
import time
import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.accel import DeviceArray, build
from katsdpsigproc import maskedsum as msum

context = accel.create_some_context(True)
queue = context.create_command_queue(profile=True)

data = np.abs(np.random.randn(4000, 5000)).astype(np.float32)
mask=np.ones((4000,1)).astype(np.float32)

template = msum.MaskedSumTemplate(context)
msum = template.instantiate(queue, data.shape)
msum.ensure_all_bound()
msum.buffer('src').set(queue,data)
msum.buffer('mask').set(queue,mask)
start_event = queue.enqueue_marker()
msum()
end_event = queue.enqueue_marker()
out = msum.buffer('dest').get(queue)

t0 = time.time()
expected=np.sum(data*mask,axis=0).astype(np.float32)
t1 = time.time()
print 'gpu:', end_event.time_since(start_event), 'cpu:', t1-t0
np.testing.assert_equal(out.reshape(-1), expected)
