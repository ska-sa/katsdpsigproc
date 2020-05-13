"""Tests for :mod:`katsdpsigproc.transpose`."""

from typing import Tuple, Callable, Generator, cast

import numpy as np

from .test_accel import device_test, force_autotune
from .. import accel
from .. import transpose
from ..abc import AbstractContext, AbstractCommandQueue


class TestTranspose:
    def test_transpose(self) -> Generator[Tuple[Callable[[int, int], None], int, int], None, None]:
        yield self.check_transpose, 4, 5
        yield self.check_transpose, 53, 7
        yield self.check_transpose, 53, 81
        yield self.check_transpose, 32, 64

    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_transpose(self, R: int, C: int,
                        context: AbstractContext, queue: AbstractCommandQueue) -> None:
        template = transpose.TransposeTemplate(context, np.float32, 'float')
        fn = template.instantiate(queue, (R, C))
        # Force some padded, to check that stride calculation works
        src_slot = cast(accel.IOSlot, fn.slots['src'])
        dest_slot = cast(accel.IOSlot, fn.slots['dest'])
        self.pad_dimension(src_slot.dimensions[0], 1)
        self.pad_dimension(src_slot.dimensions[1], 4)
        self.pad_dimension(dest_slot.dimensions[0], 2)
        self.pad_dimension(dest_slot.dimensions[1], 3)

        ary = np.random.randn(R, C).astype(np.float32)
        src = fn.slots['src'].allocate(fn.allocator)
        dest = fn.slots['dest'].allocate(fn.allocator)
        src.set_async(queue, ary)
        fn()
        out = dest.get(queue)
        np.testing.assert_equal(ary.T, out)

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Check that the autotuner runs successfully."""
        transpose.TransposeTemplate(context, np.uint8, 'unsigned char')
