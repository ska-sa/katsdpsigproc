"""Tests for :mod:`katsdpsigproc.fft`."""

import ctypes
from typing import Any, Tuple

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any  # type: ignore
from nose.tools import assert_raises, assert_raises_regex

from . import complex_normal
from .test_accel import device_test, cuda_test
from ..abc import AbstractContext, AbstractCommandQueue
from .. import fft


class TestCufft:
    """Test error handling in the :class:`._Cufft` helper class."""

    def setUp(self):
        self.cufft = fft._Cufft()

    def test_invalid_plan(self):
        plan = self.cufft.cufftHandle(123)  # An invalid plan
        with assert_raises_regex(fft._Cufft.CufftError, "CUFFT_INVALID_PLAN"):
            self.cufft.cufftSetAutoAllocation(plan, False)

    def test_unknown_error(self):
        with assert_raises_regex(fft._Cufft.CufftError, "cuFFT error 0x7b"):
            raise fft._Cufft.CufftError(123)

    def test_bad_cenum(self):
        with assert_raises(ctypes.ArgumentError):
            x = ctypes.c_longlong()
            work_size = ctypes.c_size_t()
            self.cufft.cufftMakePlanMany64(
                self.cufft.cufftHandle(), x, x, x, x, x, x, x, x, "bad", x, work_size
            )


class TestFft:
    def _test_c2c_forward(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dst: Tuple[int, ...]
    ):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, shape, dtype, dtype, padded_shape_src, padded_shape_dst)
        fn = template.instantiate(command_queue, fft.FftMode.FORWARD)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src_host = complex_normal(rs, size=src.shape).astype(dtype)
        src.set(command_queue, src_host)
        fn()
        expected = np.fft.fftn(src_host, axes=(-2, -1))
        tol = np.finfo(dtype).resolution * np.max(np.abs(expected))
        np.testing.assert_allclose(expected, dest.get(command_queue), atol=tol)

    def _test_c2c_inverse(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dst: Tuple[int, ...]
    ):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, shape, dtype, dtype, padded_shape_src, padded_shape_dst)
        fn = template.instantiate(command_queue, fft.FftMode.INVERSE)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src_host = complex_normal(rs, size=src.shape).astype(dtype)
        src.set(command_queue, src_host)
        fn()
        expected = np.fft.ifftn(src_host, axes=(-2, -1)) * (src.shape[-2] * src.shape[-1])
        tol = np.finfo(dtype).resolution * np.max(np.abs(expected))
        np.testing.assert_allclose(expected, dest.get(command_queue), atol=tol)

    def _test_r2c(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        shape: Tuple[int, ...],
        dtype_src: DTypeLike,
        dtype_dest: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...]
    ) -> None:
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, shape, dtype_src, dtype_dest, padded_shape_src, padded_shape_dest)
        fn = template.instantiate(command_queue, fft.FftMode.FORWARD)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        src_host = rs.standard_normal(src.shape).astype(dtype_src)
        src.set(command_queue, src_host)
        fn()
        expected = np.fft.rfftn(src_host, axes=(-2, -1))
        tol = np.finfo(dtype_src).resolution * np.max(np.abs(expected))
        np.testing.assert_allclose(expected, dest.get(command_queue), atol=tol)

    def _test_c2r(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        shape: Tuple[int, ...],
        dtype_src: DTypeLike,
        dtype_dest: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...]
    ) -> None:
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, shape, dtype_src, dtype_dest, padded_shape_src, padded_shape_dest)
        fn = template.instantiate(command_queue, fft.FftMode.INVERSE)
        fn.ensure_all_bound()
        src = fn.buffer('src')
        dest = fn.buffer('dest')
        signal = rs.standard_normal(shape).astype(np.float32) + 3
        src.set(command_queue, np.fft.rfftn(signal, axes=(-2, -1)).astype(dtype_src))
        fn()
        expected = signal * shape[2] * shape[3]  # CUFFT does unnormalised FFTs
        tol = np.finfo(dtype_src).resolution * np.max(np.abs(expected))
        np.testing.assert_allclose(expected, dest.get(command_queue), atol=tol)

    @device_test
    @cuda_test
    def test_r2c_even(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_r2c(context, command_queue,
                       (3, 2, 16, 48), np.float32, np.complex64, (3, 2, 24, 64), (3, 2, 20, 27))

    @device_test
    @cuda_test
    def test_r2c_odd(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_r2c(context, command_queue,
                       (3, 2, 15, 47), np.float32, np.complex64, (3, 2, 23, 63), (3, 2, 17, 24))

    @device_test
    @cuda_test
    def test_c2r_even(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_c2r(context, command_queue,
                       (3, 2, 16, 48), np.complex64, np.float32, (3, 2, 20, 27), (3, 2, 24, 64))

    @device_test
    @cuda_test
    def test_c2r_odd(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_c2r(context, command_queue,
                       (3, 2, 15, 47), np.complex64, np.float32, (3, 2, 17, 24), (3, 2, 23, 63))

    @device_test
    @cuda_test
    def test_d2z(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_r2c(context, command_queue,
                       (3, 2, 16, 48), np.float64, np.complex128, (3, 2, 24, 64), (3, 2, 20, 27))

    @device_test
    @cuda_test
    def test_z2d(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        self._test_c2r(context, command_queue,
                       (3, 2, 15, 47), np.complex128, np.float64, (3, 2, 17, 24), (3, 2, 23, 63))

    @device_test
    @cuda_test
    def test_c2c_forward(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        self._test_c2c_forward(context, command_queue,
                               (7, 16, 48), np.complex64, (7, 16, 48), (7, 18, 51))

    @device_test
    @cuda_test
    def test_c2c_inverse(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        self._test_c2c_inverse(context, command_queue,
                               (7, 16, 48), np.complex64, (7, 16, 48), (7, 18, 51))

    @device_test
    @cuda_test
    def test_z2z_forward(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        self._test_c2c_forward(context, command_queue,
                               (7, 16, 48), np.complex128, (7, 16, 48), (7, 18, 51))

    @device_test
    @cuda_test
    def test_z2z_inverse(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        self._test_c2c_inverse(context, command_queue,
                               (7, 16, 48), np.complex128, (7, 16, 48), (7, 18, 51))

    @device_test
    @cuda_test
    def test_wrong_direction(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        template = fft.FftTemplate(context, 1, (16,), np.float32, np.complex64, (16,), (9,),)
        with assert_raises_regex(ValueError, 'R2C transform must use FftMode.FORWARD'):
            template.instantiate(command_queue, fft.FftMode.INVERSE)
        template = fft.FftTemplate(context, 1, (16,), np.complex64, np.float32, (9,), (16,),)
        with assert_raises_regex(ValueError, 'C2R transform must use FftMode.INVERSE'):
            template.instantiate(command_queue, fft.FftMode.FORWARD)

    @device_test
    @cuda_test
    def test_bad_dtype_combination(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        with assert_raises_regex(ValueError, 'Invalid combination of dtypes'):
            fft.FftTemplate(context, 1, (16,), np.float32, np.complex128, (16,), (16,))
        with assert_raises_regex(ValueError, 'Invalid combination of dtypes'):
            fft.FftTemplate(context, 1, (16,), np.int32, np.int32, (16,), (16,))

    @device_test
    @cuda_test
    def test_length_mismatch(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        with assert_raises_regex(ValueError, 'padded_shape_src and shape'):
            fft.FftTemplate(context, 1, (16, 16), np.float32, np.complex64, (16,), (16, 16))
        with assert_raises_regex(ValueError, 'padded_shape_dest and shape'):
            fft.FftTemplate(context, 1, (16, 16), np.float32, np.complex64, (16, 16), (16,))

    @device_test
    @cuda_test
    def test_bad_padding(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        with assert_raises_regex(ValueError, 'Source must not be padded'):
            fft.FftTemplate(context, 1, (16, 16), np.complex64, np.complex64, (17, 16), (16, 16))
        with assert_raises_regex(ValueError, 'Destination must not be padded'):
            fft.FftTemplate(context, 1, (16, 16), np.complex64, np.complex64, (16, 16), (17, 16))
