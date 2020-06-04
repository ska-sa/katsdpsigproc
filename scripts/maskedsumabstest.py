#!/usr/bin/env python
# for nosetest: nosetests katsdpsigproc.test.test_maskedsumabs

import time
import numpy as np
from katsdpsigproc import accel, maskedsum

context = accel.create_some_context(True)
queue = context.create_command_queue(profile=True)

data = np.random.randn(4000, 5000, 2).astype(np.float32).view(dtype=np.complex64)[..., 0]
mask = np.ones((4000,)).astype(np.float32)

template = maskedsum.MaskedSumTemplate(context, use_amplitudes=True)
msum = template.instantiate(queue, data.shape)
msum.ensure_all_bound()
msum.buffer('src').set(queue, data)
msum.buffer('mask').set(queue, mask)
start_event = queue.enqueue_marker()
msum()
end_event = queue.enqueue_marker()
out = msum.buffer('dest').get(queue)

t0 = time.time()
expected = np.sum(np.abs(data) * mask.reshape(data.shape[0], 1), axis=0).astype(np.float32)
t1 = time.time()
print('gpu:', end_event.time_since(start_event), 'cpu:', t1 - t0)
np.testing.assert_allclose(expected, out.reshape(-1), rtol=1e-6)
