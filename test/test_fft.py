################################################################################
# Copyright (c) 2011-2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Tests for :mod:`katsdpsigproc.fft`."""

import ctypes
from typing import Any, Tuple

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any  # type: ignore
import pytest

from . import complex_normal
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue
from katsdpsigproc import fft


class TestCufft:
    """Test error handling in the :class:`._Cufft` helper class."""

    @pytest.fixture
    def cufft(self):
        return fft._Cufft()

    def test_invalid_plan(self, cufft: fft._Cufft) -> None:
        plan = cufft.cufftHandle(123)  # An invalid plan
        with pytest.raises(fft._Cufft.CufftError, match="CUFFT_INVALID_PLAN"):
            cufft.cufftSetAutoAllocation(plan, False)

    def test_unknown_error(self) -> None:
        with pytest.raises(fft._Cufft.CufftError, match="cuFFT error 0x7b"):
            raise fft._Cufft.CufftError(123)

    def test_bad_cenum(self, cufft: fft._Cufft) -> None:
        with pytest.raises(ctypes.ArgumentError):
            x = ctypes.c_longlong()
            work_size = ctypes.c_size_t()
            cufft.cufftMakePlanMany64(
                cufft.cufftHandle(), x, x, x, x, x, x, x, x, "bad", x, work_size
            )


@pytest.mark.cuda_only
class TestFft:
    @pytest.mark.parametrize(
        'shape, dtype, padded_shape_src, padded_shape_dest',
        [
            ((7, 16, 48), np.complex64, (7, 16, 48), (7, 18, 51)),
            ((7, 16, 48), np.complex128, (7, 16, 48), (7, 18, 51))
        ]
    )
    def test_c2c_forward(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...]
    ):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, shape, dtype, dtype, padded_shape_src, padded_shape_dest)
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

    @pytest.mark.parametrize(
        'shape, dtype, padded_shape_src, padded_shape_dest',
        [
            ((7, 16, 48), np.complex64, (7, 16, 48), (7, 18, 51)),
            ((7, 16, 48), np.complex128, (7, 16, 48), (7, 18, 51))
        ]
    )
    def test_c2c_inverse(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...]
    ):
        rs = np.random.RandomState(1)
        template = fft.FftTemplate(
            context, 2, shape, dtype, dtype, padded_shape_src, padded_shape_dest)
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

    @pytest.mark.parametrize(
        'shape, dtype_src, dtype_dest, padded_shape_src, padded_shape_dest',
        [
            ((3, 2, 16, 48), np.float32, np.complex64, (3, 2, 24, 64), (3, 2, 20, 27)),
            ((3, 2, 15, 47), np.float32, np.complex64, (3, 2, 23, 63), (3, 2, 17, 24)),
            ((3, 2, 16, 48), np.float64, np.complex128, (3, 2, 24, 64), (3, 2, 20, 27))
        ]
    )
    def test_r2c(
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

    @pytest.mark.parametrize(
        'shape, dtype_src, dtype_dest, padded_shape_src, padded_shape_dest',
        [
            ((3, 2, 16, 48), np.complex64, np.float32, (3, 2, 20, 27), (3, 2, 24, 64)),
            ((3, 2, 15, 47), np.complex64, np.float32, (3, 2, 17, 24), (3, 2, 23, 63)),
            ((3, 2, 15, 47), np.complex128, np.float64, (3, 2, 17, 24), (3, 2, 23, 63))
        ]
    )
    def test_c2r(
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

    def test_wrong_direction(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        template = fft.FftTemplate(context, 1, (16,), np.float32, np.complex64, (16,), (9,),)
        with pytest.raises(ValueError, match='R2C transform must use FftMode.FORWARD'):
            template.instantiate(command_queue, fft.FftMode.INVERSE)
        template = fft.FftTemplate(context, 1, (16,), np.complex64, np.float32, (9,), (16,),)
        with pytest.raises(ValueError, match='C2R transform must use FftMode.INVERSE'):
            template.instantiate(command_queue, fft.FftMode.FORWARD)

    def test_bad_dtype_combination(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        with pytest.raises(ValueError, match='Invalid combination of dtypes'):
            fft.FftTemplate(context, 1, (16,), np.float32, np.complex128, (16,), (16,))
        with pytest.raises(ValueError, match='Invalid combination of dtypes'):
            fft.FftTemplate(context, 1, (16,), np.int32, np.int32, (16,), (16,))

    def test_length_mismatch(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        with pytest.raises(ValueError, match='padded_shape_src and shape'):
            fft.FftTemplate(context, 1, (16, 16), np.float32, np.complex64, (16,), (16, 16))
        with pytest.raises(ValueError, match='padded_shape_dest and shape'):
            fft.FftTemplate(context, 1, (16, 16), np.float32, np.complex64, (16, 16), (16,))

    def test_bad_padding(
        self,
        context: AbstractContext,
        command_queue: AbstractCommandQueue
    ) -> None:
        with pytest.raises(ValueError, match='Source must not be padded'):
            fft.FftTemplate(context, 1, (16, 16), np.complex64, np.complex64, (17, 16), (16, 16))
        with pytest.raises(ValueError, match='Destination must not be padded'):
            fft.FftTemplate(context, 1, (16, 16), np.complex64, np.complex64, (16, 16), (17, 16))
