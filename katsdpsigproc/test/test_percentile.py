#!/usr/bin/env python
import numpy as np
from . import test_accel
from .test_accel import device_test, force_autotune
from .. import accel
from .. import percentile

class TestPercentile5(object):
    def test_percentile(self):
        yield self.check_percentile5, 4096, 4032

    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_percentile5(self, R, C, context, queue):
        template = percentile.Percentile5Template(context, max_baselines=5000)
        fn = template.instantiate(queue, (R, C))
        # Force some padded, to check that stride calculation works
        self.pad_dimension(fn.slots['src'].dimensions[0], 1)
        self.pad_dimension(fn.slots['src'].dimensions[1], 4)
        self.pad_dimension(fn.slots['dest'].dimensions[0], 2)
        self.pad_dimension(fn.slots['dest'].dimensions[1], 3)
        ary = np.abs(np.random.randn(R, C)).astype(np.float32)
        src = fn.slots['src'].allocate(context)
        dest = fn.slots['dest'].allocate(context)
        src.set_async(queue, ary)
        fn()
        out = dest.get(queue)
        np.testing.assert_equal(np.percentile(ary,[0,100,25,75,50],axis=1,interpolation='lower').astype(dtype=np.float32), out)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that the autotuner runs successfully"""
        percentile.Percentile5Template(context, max_baselines=5000)
