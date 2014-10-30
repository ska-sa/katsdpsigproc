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

    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_transpose(self, R, C):
        fn = self.template.instantiate(test_command_queue, (R, C))
        # Force some padded, to check that stride calculation works
        self.pad_dimension(fn.slots['src'].dimensions[0], 1)
        self.pad_dimension(fn.slots['src'].dimensions[1], 4)
        self.pad_dimension(fn.slots['dest'].dimensions[0], 2)
        self.pad_dimension(fn.slots['dest'].dimensions[1], 3)

        ary = np.random.randn(R, C).astype(np.float32)
        src = fn.slots['src'].allocate(test_context)
        dest = fn.slots['dest'].allocate(test_context)
        src.set_async(test_command_queue, ary)
        fn(src=src, dest=dest)
        out = dest.get(test_command_queue)
        np.testing.assert_equal(ary.T, out)
