#!/usr/bin/env python
# for nosetest: nosetests katsdpsigproc.test.test_percentile

import time
import numpy as np
from katsdpsigproc import accel
from katsdpsigproc import percentile as perc5

context = accel.create_some_context(True)
queue = context.create_command_queue(profile=True)

data = np.abs(np.random.randn(4000, 5000)).astype(np.float32)

template = perc5.Percentile5Template(context, max_columns=5000)
perc = template.instantiate(queue, data.shape)  # type: ignore [arg-type]
perc.ensure_all_bound()
perc.buffer('src').set(queue, data)
start_event = queue.enqueue_marker()
perc()
end_event = queue.enqueue_marker()
out = perc.buffer('dest').get(queue)

t0 = time.time()
expected = np.percentile(data, [0, 100, 25, 75, 50], axis=1, interpolation='lower')
t1 = time.time()
print('gpu:', end_event.time_since(start_event), 'cpu:', t1 - t0)
np.testing.assert_equal(out, expected)
