#!/usr/bin/env python
import numpy as np
from nose.tools import assert_equal
from .test_accel import test_context, test_command_queue, device_test
from .. import accel
from .. import fill

class TestFill(object):
    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def test_fill(self):
        shape = (75, 63)
        template = fill.FillTemplate(test_context, np.uint32, 'unsigned int')
        fn = template.instantiate(test_command_queue, shape)
        self.pad_dimension(fn.slots['data'].dimensions[0], 5)
        self.pad_dimension(fn.slots['data'].dimensions[1], 10)
        fn.ensure_all_bound()
        data = fn.slots['data'].buffer
        # Do the fill
        fn.set_value(0xDEADBEEF)
        fn()
        # Check the result, including padding
        ret = data.get(test_command_queue)
        ret = ret.base
        np.testing.assert_equal(ret, 0xDEADBEEF)
