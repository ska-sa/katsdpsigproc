#!/usr/bin/env python
import numpy as np
from nose.tools import assert_equal
from .test_accel import test_context, test_command_queue, device_test
from .. import accel
from .. import fill

class TestFill(object):
    @device_test
    def test_fill(self):
        shape = (75, 63)
        padded_shape = (128, 96)
        data = accel.DeviceArray(test_context, shape, np.uint32, padded_shape=padded_shape)
        template = fill.FillTemplate(test_context, np.uint32, 'unsigned int')
        fn = template.instantiate(test_command_queue, shape)
        fn.slots['data'].bind(data)
        # Do the fill
        fn(0xDEADBEEF)
        # Check the result, including padding
        ret = data.get(test_command_queue)
        ret = ret.base
        assert_equal(padded_shape, ret.shape)
        np.testing.assert_equal(ret, 0xDEADBEEF)
