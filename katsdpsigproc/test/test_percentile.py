#!/usr/bin/env python
import numpy as np
from . import test_accel
from .test_accel import device_test, force_autotune
from .. import accel
from .. import percentile

class TestPercentile5(object):
    def test_percentile(self):
        yield self.check_percentile5, 4096, 1, False, None
        yield self.check_percentile5, 4096, 4029, True, (0, 4009)
        yield self.check_percentile5, 4096, 4030, False, (100, 4030)
        yield self.check_percentile5, 2345, 6031, False, (123, 4001)
        yield self.check_percentile5, 4096, 4032, True, None

    @classmethod
    def pad_dimension(cls, dim, extra):
        """Modifies `dim` to have at least `extra` padding"""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_percentile5(self, R, C, is_amplitude, column_range, context, queue):
        template = percentile.Percentile5Template(context, max_columns=5000, is_amplitude=is_amplitude)
        fn = template.instantiate(queue, (R, C), column_range)
        # Force some padded, to check that stride calculation works
        self.pad_dimension(fn.slots['src'].dimensions[0], 1)
        self.pad_dimension(fn.slots['src'].dimensions[1], 4)
        self.pad_dimension(fn.slots['dest'].dimensions[0], 2)
        self.pad_dimension(fn.slots['dest'].dimensions[1], 3)
        rs = np.random.RandomState(seed=1)
        if is_amplitude:
            ary = np.abs(rs.randn(R, C)).astype(np.float32) #note positive numbers required
        else:
            ary = (rs.randn(R, C) + 1j * rs.randn(R, C)).astype(np.complex64)
        src = fn.slots['src'].allocate(context)
        dest = fn.slots['dest'].allocate(context)
        src.set_async(queue, ary)
        fn()
        out = dest.get(queue)
        if column_range is None:
            column_range = (0, C)
        expected=np.percentile(np.abs(ary[:, column_range[0]:column_range[1]]),[0,100,25,75,50],axis=1,interpolation='lower').astype(dtype=np.float32)
        # When amplitudes are being computed, we won't get a bit-exact match
        if is_amplitude:
            np.testing.assert_equal(expected, out)
        else:
            np.testing.assert_allclose(expected, out, 1e-6)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Check that the autotuner runs successfully"""
        percentile.Percentile5Template(context, max_columns=5000)
