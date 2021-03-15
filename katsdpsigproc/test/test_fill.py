"""Tests for :mod:`katsdpsigproc.fill`."""

from typing import cast

import numpy as np

from .test_accel import device_test, force_autotune
from .. import accel
from .. import fill
from ..abc import AbstractContext, AbstractCommandQueue


class TestFill:
    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def test_fill(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        shape = (75, 63)
        template = fill.FillTemplate(context, np.uint32, 'unsigned int')
        fn = template.instantiate(queue, shape)
        data_slot = cast(accel.IOSlot, fn.slots['data'])
        self.pad_dimension(data_slot.dimensions[0], 5)
        self.pad_dimension(data_slot.dimensions[1], 10)
        fn.ensure_all_bound()
        data = fn.buffer('data')
        # Do the fill
        fn.set_value(0xDEADBEEF)
        fn()
        # Check the result, including padding
        ret = data.get(queue)
        assert ret.base is not None
        ret = ret.base
        np.testing.assert_equal(ret, 0xDEADBEEF)

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Test that autotuner runs successfully."""
        fill.FillTemplate(context, np.uint8, 'unsigned char')
