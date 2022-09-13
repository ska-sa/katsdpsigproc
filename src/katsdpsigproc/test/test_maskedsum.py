"""Tests for :mod:`katsdpsigproc.maskedsum`."""

from typing import Tuple, Generator, Callable, cast

import numpy as np

from .test_accel import device_test, force_autotune
from .. import accel
from .. import maskedsum
from ..abc import AbstractContext, AbstractCommandQueue


class TestMaskedSum:
    def test_maskedsum(self) -> Generator[
            Tuple[Callable[[int, int, bool], None], int, int, bool],
            None, None]:
        for use_amplitudes in (False, True):
            yield self.check_maskedsum, 4096, 2, use_amplitudes
            yield self.check_maskedsum, 4096, 4029, use_amplitudes
            yield self.check_maskedsum, 4096, 4030, use_amplitudes
            yield self.check_maskedsum, 4096, 4031, use_amplitudes
            yield self.check_maskedsum, 4096, 4032, use_amplitudes

    @classmethod
    def pad_dimension(cls, dim: accel.Dimension, extra: int) -> None:
        """Modify `dim` to have at least `extra` padding."""
        newdim = accel.Dimension(dim.size, min_padded_size=dim.size + extra)
        newdim.link(dim)

    @device_test
    def check_maskedsum(self, R: int, C: int, use_amplitudes: bool,
                        context: AbstractContext, queue: AbstractCommandQueue) -> None:
        template = maskedsum.MaskedSumTemplate(context, use_amplitudes)
        fn = template.instantiate(queue, (R, C))
        # Force some padding, to check that stride calculation works
        src_slot = cast(accel.IOSlot, fn.slots['src'])
        self.pad_dimension(src_slot.dimensions[0], 1)
        self.pad_dimension(src_slot.dimensions[1], 4)
        ary = np.random.randn(R, C, 2).astype(np.float32).view(dtype=np.complex64)[..., 0]
        msk = np.ones((R,)).astype(np.float32)
        src = fn.slots['src'].allocate(fn.allocator)
        mask = fn.slots['mask'].allocate(fn.allocator)
        dest = fn.slots['dest'].allocate(fn.allocator)
        src.set_async(queue, ary)
        mask.set_async(queue, msk)
        fn()
        out = dest.get(queue).reshape(-1)
        if use_amplitudes:
            use_ary = np.abs(ary)
        else:
            use_ary = ary
        expected = np.sum(use_ary * msk.reshape(ary.shape[0], 1), axis=0)
        np.testing.assert_allclose(expected, out, rtol=1e-6)

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Check that the autotuner runs successfully."""
        maskedsum.MaskedSumTemplate(context, False)
        maskedsum.MaskedSumTemplate(context, True)
