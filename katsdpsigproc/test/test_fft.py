"""Tests for :mod:`katsdpsigproc.fft`."""

from typing import Tuple

import numpy as np

from . import complex_normal
from .test_accel import device_test, cuda_test
from ..abc import AbstractContext, AbstractCommandQueue
from .. import fft


class TestFft:
    @device_test
    @cuda_test
    def test_forward(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, (3, 2, 16, 48), np.complex64, np.complex64,
            (3, 2, 24, 64), (3, 2, 20, 48))
        fn = template.instantiate(command_queue, fft.FftMode.FORWARD)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src_host = complex_normal(rs, size=src.shape).astype(np.complex64)
        src.set(command_queue, src_host)
        fn()
        expected = np.fft.fftn(src_host, axes=(2, 3))
        np.testing.assert_allclose(expected, dest.get(command_queue), rtol=1e-4)

    def _test_r2c(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        N: int,
        shape: Tuple[int, ...],
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...]
    ) -> None:
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, N, shape, np.float32, np.complex64, padded_shape_src, padded_shape_dest)
        fn = template.instantiate(command_queue, fft.FftMode.FORWARD)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src_host = rs.standard_normal(src.shape).astype(np.float32)
        src.set(command_queue, src_host)
        fn()
        expected = np.fft.rfftn(src_host, axes=(2, 3))
        np.testing.assert_allclose(expected, dest.get(command_queue), rtol=1e-4)

    def _test_c2r(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        N: int,
        shape: Tuple[int, ...],
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...]
    ) -> None:
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, N, shape, np.complex64, np.float32, padded_shape_src, padded_shape_dest)
        fn = template.instantiate(command_queue, fft.FftMode.INVERSE)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        signal = rs.standard_normal(shape).astype(np.float32) + 3
        src.set(command_queue, np.fft.rfftn(signal, axes=(2, 3)).astype(np.complex64))
        fn()
        expected = signal * shape[2] * shape[3]
        np.testing.assert_allclose(expected, dest.get(command_queue), rtol=1e-4)

    @device_test
    @cuda_test
    def test_r2c_even(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_r2c(context, command_queue, 2, (3, 2, 16, 48), (3, 2, 24, 64), (3, 2, 20, 27))

    @device_test
    @cuda_test
    def test_r2c_odd(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_r2c(context, command_queue, 2, (3, 2, 15, 47), (3, 2, 23, 63), (3, 2, 17, 24))

    @device_test
    @cuda_test
    def test_c2r_even(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_c2r(context, command_queue, 2, (3, 2, 16, 48), (3, 2, 20, 27), (3, 2, 24, 64))

    @device_test
    @cuda_test
    def test_c2r_odd(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_c2r(context, command_queue, 2, (3, 2, 15, 47), (3, 2, 17, 24), (3, 2, 23, 63))

    @device_test
    @cuda_test
    def test_inverse(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, (3, 2, 16, 48), np.complex64, np.complex64,
            (3, 2, 24, 64), (3, 2, 20, 48))
        fn = template.instantiate(command_queue, fft.FftMode.INVERSE)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src_host = complex_normal(rs, size=src.shape).astype(np.complex64)
        src.set(command_queue, src_host)
        fn()
        expected = np.fft.ifftn(src_host, axes=(2, 3)) * (src.shape[2] * src.shape[3])
        np.testing.assert_allclose(expected, dest.get(command_queue), rtol=1e-4)
