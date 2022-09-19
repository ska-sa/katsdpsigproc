#!/usr/bin/env python

################################################################################
# Copyright (c) 2014-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

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
