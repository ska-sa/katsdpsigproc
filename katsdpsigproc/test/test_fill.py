#!/usr/bin/env python
import numpy as np
from nose.tools import assert_equal
from .test_accel import device_test, force_autotune
from .. import accel
from .. import fill

class TestFill(object):
    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def test_fill(self, context, queue):
        shape = (75, 63)
        template = fill.FillTemplate(context, np.uint32, 'unsigned int')
        fn = template.instantiate(queue, shape)
        self.pad_dimension(fn.slots['data'].dimensions[0], 5)
        self.pad_dimension(fn.slots['data'].dimensions[1], 10)
        fn.ensure_all_bound()
        data = fn.slots['data'].buffer
        # Do the fill
        fn.set_value(0xDEADBEEF)
        fn()
        # Check the result, including padding
        ret = data.get(queue)
        ret = ret.base
        np.testing.assert_equal(ret, 0xDEADBEEF)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Test that autotuner runs successfully"""
        fill.FillTemplate(context, np.uint8, 'unsigned char')
