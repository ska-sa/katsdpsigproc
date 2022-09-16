#!/usr/bin/env python

################################################################################
# Copyright (c) 2015-2020, National Research Foundation (SARAO)
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
