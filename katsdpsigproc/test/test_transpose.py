#!/usr/bin/env python
import numpy as np
from .test_accel import test_context, test_command_queue, device_test
from .. import accel
from .. import transpose

class TestTranspose(object):
    def test_transpose(self):
        yield self.check_transpose, 4, 5
        yield self.check_transpose, 53, 7
        yield self.check_transpose, 53, 81
        yield self.check_transpose, 32, 64

    @device_test
    def setup(self):
        self.template = transpose.TransposeTemplate(test_context, np.float32, 'float')

    @device_test
    def check_transpose(self, R, C):
        fn = self.template.instantiate(test_command_queue, (R, C))

        ary = np.random.randn(R, C).astype(np.float32)
        src = accel.DeviceArray(test_context, (R, C), dtype=np.float32, padded_shape=(R + 1, C + 4))
        dest = accel.DeviceArray(test_context, (C, R), dtype=np.float32, padded_shape=(C + 2, R + 3))
        src.set_async(test_command_queue, ary)
        fn(src=src, dest=dest)
        out = dest.get(test_command_queue)
        np.testing.assert_equal(ary.T, out)
