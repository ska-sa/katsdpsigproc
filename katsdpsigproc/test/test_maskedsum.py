#!/usr/bin/env python
import numpy as np
from . import test_accel
from .test_accel import device_test, force_autotune
from .. import accel
from .. import maskedsum

class TestMaskedSum(object):
    def test_maskedsum(self):
        yield self.check_maskedsum, 4096, 2
        yield self.check_maskedsum, 4096, 4029
        yield self.check_maskedsum, 4096, 4030
        yield self.check_maskedsum, 4096, 4031
        yield self.check_maskedsum, 4096, 4032

    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_maskedsum(self, R, C, context, queue):
        template = maskedsum.MaskedSumTemplate(context)
        fn = template.instantiate(queue, (R, C))
        # Force some padded, to check that stride calculation works
        self.pad_dimension(fn.slots['src'].dimensions[0], 1)
        self.pad_dimension(fn.slots['src'].dimensions[1], 4)
        ary = np.random.randn(R, C).astype(np.float32)
        msk = np.ones((R,1)).astype(np.float32)
        src = fn.slots['src'].allocate(context)
        mask = fn.slots['mask'].allocate(context)
        dest = fn.slots['dest'].allocate(context)
        src.set_async(queue, ary)
        mask.set_async(queue, msk)
        fn()
        out = dest.get(queue).reshape(-1)
        expected=np.sum(ary*msk,axis=0).astype(np.float32).reshape(-1)
        np.testing.assert_equal(expected, out)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that the autotuner runs successfully"""
        maskedsum.MaskedSumTemplate(context)
