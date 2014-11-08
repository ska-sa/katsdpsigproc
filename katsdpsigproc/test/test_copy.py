#!/usr/bin/env python
import numpy as np
from nose.tools import assert_equal
from .test_accel import device_test, force_autotune
from .. import accel
from .. import copy

class TestCopy(object):
    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def test_fill(self, context, queue):
        shape = (75, 63)
        template = copy.CopyTemplate(context, np.uint32, 'unsigned int')
        fn = template.instantiate(queue, shape)
        self.pad_dimension(fn.slots['src'].dimensions[0], 5)
        self.pad_dimension(fn.slots['src'].dimensions[1], 10)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        # Create random but reproducible data
        rs = np.random.RandomState(1)
        data = src.empty_like()
        data.base[:] = rs.randint(0, 100000, size=data.base.shape)
        src.set(queue, data)
        # Do the copy
        fn()
        # Check the result, including padding
        ret = dest.get(queue)
        ret = ret.base
        np.testing.assert_equal(ret, data.base)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Test that autotuner runs successfully"""
        copy.CopyTemplate(context, np.uint8, 'unsigned char')
