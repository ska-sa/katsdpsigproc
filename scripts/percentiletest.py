#!/usr/bin/env python
#for nosetest: nosetests katsdpsigproc.test.test_percentile
import time
import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.accel import DeviceArray, build
from katsdpsigproc import percentile as perc5

context = accel.create_some_context(True)
queue = context.create_command_queue(profile = True)

data = np.abs(np.random.randn(4000, 5000)).astype(np.float32)

template = perc5.Percentile5Template(context, max_columns = 5000)
perc = template.instantiate(queue, data.shape)
perc.ensure_all_bound()
perc.buffer('src').set(queue,data)
perc()
t0 = time.time()
out = perc.buffer('dest').get(queue)
t1 = time.time()
expected = np.percentile(data, [0, 100, 25, 75, 50], axis = 1, interpolation = 'lower')
t2 = time.time()
print 'gpu:', t1-t0, 'cpu:', t2-t1
np.testing.assert_equal(out, expected)



