#!/usr/bin/env python
import numpy as np
from . import test_accel
from .test_accel import device_test, force_autotune
from .. import accel
from .. import transpose

class TestTranspose(object):
    def test_transpose(self):
        yield self.check_transpose, 4, 5
        yield self.check_transpose, 53, 7
        yield self.check_transpose, 53, 81
        yield self.check_transpose, 32, 64

    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_transpose(self, R, C, context, queue):
        template = transpose.TransposeTemplate(context, np.float32, 'float')
        fn = template.instantiate(queue, (R, C))
        # Force some padded, to check that stride calculation works
        self.pad_dimension(fn.slots['src'].dimensions[0], 1)
        self.pad_dimension(fn.slots['src'].dimensions[1], 4)
        self.pad_dimension(fn.slots['dest'].dimensions[0], 2)
        self.pad_dimension(fn.slots['dest'].dimensions[1], 3)

        ary = np.random.randn(R, C).astype(np.float32)
        src = fn.slots['src'].allocate(context)
        dest = fn.slots['dest'].allocate(context)
        src.set_async(queue, ary)
        fn()
        out = dest.get(queue)
        np.testing.assert_equal(ary.T, out)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that the autotuner runs successfully"""
        transpose.TransposeTemplate(context, np.uint8, 'unsigned char')
