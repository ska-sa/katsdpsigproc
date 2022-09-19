################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Fast Fourier Transforms.

Currently this module only supports CUDA, using cuFFT.

It does not use scikit-cuda, because that insists on creating a cublas context
to obtain a version number, which then permanently consumes GPU memory on some
arbitrary GPU. It also seems the maintainer `no longer has time
<https://github.com/lebedov/scikit-cuda/issues/319>`_.

Only Linux is supported. Windows and MacOS support would require changing the
code for locating the library.
"""

from enum import Enum
import ctypes.util
import threading
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any  # type: ignore

from . import accel, cuda
from .accel import AbstractAllocator
from .abc import AbstractContext, AbstractCommandQueue


class FftMode(Enum):
    FORWARD = 0
    INVERSE = 1


class _CEnum(Enum):
    """Enum type that can be used with ctypes."""

    @classmethod
    def from_param(cls, value):
        if not isinstance(value, cls):
            raise TypeError
        # TODO: assumes that enum and int have the size in the ABI. But this
        # seems to be what scikit-cuda assumed anyway.
        return ctypes.c_int(value.value)


class _Cufft:
    """Wraps up the low-level ctypes access to cuFFT.

    This is not intended to be a complete wrapper for cuFFT, nor to be
    user-friendly. It provides just enough for the implementation of this
    module.
    """

    cufftHandle = ctypes.c_int
    cudaStream_t = ctypes.c_void_p  # It's a pointer to an opaque struct

    class cufftType(_CEnum):
        CUFFT_R2C = 0x2a
        CUFFT_C2R = 0x2c
        CUFFT_C2C = 0x29
        CUFFT_D2Z = 0x6a
        CUFFT_Z2D = 0x6c
        CUFFT_Z2Z = 0x69

    class cufftResult(_CEnum):
        CUFFT_SUCCESS = 0
        CUFFT_INVALID_PLAN = 1
        CUFFT_ALLOC_FAILED = 2
        CUFFT_INVALID_TYPE = 3
        CUFFT_INVALID_VALUE = 4
        CUFFT_INTERNAL_ERROR = 5
        CUFFT_EXEC_FAILED = 6
        CUFFT_SETUP_FAILED = 7
        CUFFT_INVALID_SIZE = 8
        CUFFT_UNALIGNED_DATA = 9
        CUFFT_INCOMPLETE_PARAMETER_LIST = 10
        CUFFT_INVALID_DEVICE = 11
        CUFFT_PARSE_ERROR = 12
        CUFFT_NO_WORKSPACE = 13
        CUFFT_NOT_IMPLEMENTED = 14
        CUFFT_LICENSE_ERROR = 15
        CUFFT_NOT_SUPPORTED = 16

    # cufft.h has no such type (these are just #defines), but it is convenient
    # to use an enum in Python.
    class cufftDirection(_CEnum):
        CUFFT_FORWARD = -1
        CUFFT_INVERSE = 1

    class CufftError(Exception):
        """Error raised from cuFFT"""

        def __init__(self, error_code: int):
            self.error_code = error_code
            try:
                name = _Cufft.cufftResult(error_code).name
            except ValueError:
                name = f"cuFFT error {error_code:#x}"
            super().__init__(name)

    @staticmethod
    def _errcheck(result, func, args) -> None:
        """Turn cufftResult into an exception."""
        if result:
            raise _Cufft.CufftError(result)

    def _get_function(self, name: str, argtypes: list) -> Callable[..., None]:
        """Shortcut to build a ctypes function wrapper.

        The function must return a :class:`cufftResult`.
        """
        func = getattr(self.lib, name)
        func.argtypes = argtypes
        func.restype = ctypes.c_int
        func.errcheck = self._errcheck
        return func

    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("cufft"))
        self.cufftGetVersion = self._get_function("cufftGetVersion", [ctypes.POINTER(ctypes.c_int)])
        self.cufftCreate = self._get_function("cufftCreate", [ctypes.POINTER(self.cufftHandle)])
        self.cufftDestroy = self._get_function("cufftDestroy", [self.cufftHandle])
        self.cufftMakePlanMany64 = self._get_function(
            "cufftMakePlanMany64",
            [
                self.cufftHandle,
                ctypes.c_int,                           # rank
                ctypes.POINTER(ctypes.c_longlong),      # n
                ctypes.POINTER(ctypes.c_longlong),      # inembed
                ctypes.c_longlong,                      # istride,
                ctypes.c_longlong,                      # idist
                ctypes.POINTER(ctypes.c_longlong),      # onembed
                ctypes.c_longlong,                      # ostride
                ctypes.c_longlong,                      # odist
                self.cufftType,                         # type
                ctypes.c_longlong,                      # batch
                ctypes.POINTER(ctypes.c_size_t)         # workSize
            ]
        )
        self.cufftSetAutoAllocation = self._get_function(
            "cufftSetAutoAllocation", [self.cufftHandle, ctypes.c_int]
        )
        self.cufftSetStream = self._get_function(
            "cufftSetStream", [self.cufftHandle, self.cudaStream_t]
        )
        self.cufftSetWorkArea = self._get_function(
            "cufftSetWorkArea", [self.cufftHandle, ctypes.c_void_p]
        )

        # The actual function signatures have typed pointers rather than void*, but
        # specifying that just making invocation more complicated, and requires
        # defining types like cufftComplex.
        self.cufftExec = {
            self.cufftType.CUFFT_R2C: self._get_function(
                "cufftExecR2C", [self.cufftHandle, ctypes.c_void_p, ctypes.c_void_p]
            ),
            self.cufftType.CUFFT_C2R: self._get_function(
                "cufftExecC2R", [self.cufftHandle, ctypes.c_void_p, ctypes.c_void_p]
            ),
            self.cufftType.CUFFT_D2Z: self._get_function(
                "cufftExecD2Z", [self.cufftHandle, ctypes.c_void_p, ctypes.c_void_p]
            ),
            self.cufftType.CUFFT_Z2D: self._get_function(
                "cufftExecZ2D", [self.cufftHandle, ctypes.c_void_p, ctypes.c_void_p]
            ),
            self.cufftType.CUFFT_C2C: self._get_function(
                "cufftExecC2C",
                [self.cufftHandle, ctypes.c_void_p, ctypes.c_void_p, self.cufftDirection]
            ),
            self.cufftType.CUFFT_Z2Z: self._get_function(
                "cufftExecZ2Z",
                [self.cufftHandle, ctypes.c_void_p, ctypes.c_void_p, self.cufftDirection]
            )
        }


class FftTemplate:
    r"""Operation template for a forward or reverse FFT.

    The transformation is done over the last N dimensions, with the remaining
    dimensions for batching multiple arrays to be transformed. Dimensions
    before the first N must not be padded.

    This template bakes in more information than most (data shapes), which is
    due to constraints in CUFFT.

    The template can specify real->complex, complex->real, or
    complex->complex. In the last case, the same template can be used to
    instantiate forward or inverse transforms. Otherwise, real->complex
    transforms must be forward, and complex->real transforms must be inverse.

    For real<->complex transforms, the final dimension of the padded shape
    need only be :math:`\lfloor\frac{L}{2}\rfloor + 1`, where :math:`L` is the
    last element of `shape`.

    The transform is unnormalised: performing a forward followed by a reverse
    transform will scale the result by the number of elements.

    Parameters
    ----------
    context
        Context for the operation
    N
        Number of dimensions for the transform
    shape
        Shape of the data (N or more dimensions). For real->complex or
        complex->real transformation, this is the size of the real side of the
        transform.
    dtype_src : {`np.float32`, `np.float64`, `np.complex64`, `np.complex128`}
        Data type for input
    dtype_dest : {`np.float32`, `np.float64`, `np.complex64`, `np.complex128`}
        Data type for output
    padded_shape_src
        Padded shape of the input
    padded_shape_dest
        Padded shape of the output
    tuning
        Tuning parameters (currently unused)
    """

    def __init__(
        self,
        context: AbstractContext,
        N: int,
        shape: Tuple[int, ...],
        dtype_src: DTypeLike,
        dtype_dest: DTypeLike,
        padded_shape_src: Tuple[int, ...],
        padded_shape_dest: Tuple[int, ...],
        tuning: Optional[Dict[str, Any]] = None
    ) -> None:
        if not isinstance(context, cuda.Context):
            raise TypeError('Only CUDA contexts are supported')
        if len(padded_shape_src) != len(shape):
            raise ValueError('padded_shape_src and shape must have same length')
        if len(padded_shape_dest) != len(shape):
            raise ValueError('padded_shape_dest and shape must have same length')
        if padded_shape_src[:-N] != shape[:-N]:
            raise ValueError('Source must not be padded on batch dimensions')
        if padded_shape_dest[:-N] != shape[:-N]:
            raise ValueError('Destination must not be padded on batch dimensions')

        if dtype_src == np.float32 and dtype_dest == np.complex64:
            fft_type = _Cufft.cufftType.CUFFT_R2C
        elif dtype_src == np.complex64 and dtype_dest == np.float32:
            fft_type = _Cufft.cufftType.CUFFT_C2R
        elif dtype_src == np.complex64 and dtype_dest == np.complex64:
            fft_type = _Cufft.cufftType.CUFFT_C2C
        elif dtype_src == np.float64 and dtype_dest == np.complex128:
            fft_type = _Cufft.cufftType.CUFFT_D2Z
        elif dtype_src == np.complex128 and dtype_dest == np.float64:
            fft_type = _Cufft.cufftType.CUFFT_Z2D
        elif dtype_src == np.complex128 and dtype_dest == np.complex128:
            fft_type = _Cufft.cufftType.CUFFT_Z2Z
        else:
            raise ValueError("Invalid combination of dtypes")

        cufft = _Cufft()
        self._cufft = cufft
        self.context: cuda.Context = context
        self.shape = shape
        # TODO: the type annotation is necessary with numpy 1.20.1, but
        # is no longer needed for newer versions.
        self.dtype_src: np.dtype = np.dtype(dtype_src)
        self.dtype_dest: np.dtype = np.dtype(dtype_dest)
        self.padded_shape_src = padded_shape_src
        self.padded_shape_dest = padded_shape_dest
        # CUDA 7.0 CUFFT has a bug where kernels are run in the default stream
        # instead of the requested one, if dimensions are up to 1920. There is
        # a patch, but there is no query to detect whether it has been
        # applied.
        version = ctypes.c_int()
        cufft.cufftGetVersion(ctypes.byref(version))
        self._needs_synchronize_workaround = (
            version.value < 8000 and any(x <= 1920 for x in shape[:N])
        )
        batches = int(np.product(padded_shape_src[:-N]))
        with context:
            arr_type = ctypes.c_longlong * N
            work_size = ctypes.c_size_t()
            self._plan = cufft.cufftHandle()
            cufft.cufftCreate(ctypes.byref(self._plan))
            cufft.cufftSetAutoAllocation(self._plan, False)
            cufft.cufftMakePlanMany64(
                self._plan,                               # handle
                N,                                        # rank
                arr_type(*shape[-N:]),                    # n
                arr_type(*padded_shape_src[-N:]),         # inembed
                1,                                        # istride
                int(np.product(padded_shape_src[-N:])),   # idist
                arr_type(*padded_shape_dest[-N:]),        # onembed
                1,                                        # ostride
                int(np.product(padded_shape_dest[-N:])),  # odist
                fft_type,                                 # type
                batches,                                  # batch
                ctypes.byref(work_size)                   # workSize
            )
        self._work_size = work_size.value
        self._cufft = cufft
        self._fft_type = fft_type
        # The stream and work area are associated with the plan rather than
        # the execution, so we need to serialise executions.
        self._lock = threading.RLock()

    def instantiate(self, *args, **kwargs):
        return Fft(self, *args, **kwargs)

    def __del__(self):
        if hasattr(self, '_plan'):
            with self.context:
                self._cufft.cufftDestroy(self._plan)


class Fft(accel.Operation):
    """Forward or inverse Fourier transformation.

    .. rubric:: Slots

    **src**
        Input data
    **dest**
        Output data
    **work_area**
        Scratch area for work. The contents should not be used; it is made
        available so that it can be aliased with other scratch areas.

    Parameters
    ----------
    template
        Operation template
    command_queue
        Command queue for the operation
    mode
        FFT direction
    allocator
        Allocator used to allocate unbound slots
    """

    command_queue: cuda.CommandQueue  # refine the type for mypy

    def __init__(
        self,
        template: FftTemplate,
        command_queue: AbstractCommandQueue,
        mode: FftMode,
        allocator: Optional[AbstractAllocator] = None
    ) -> None:
        if not isinstance(command_queue, cuda.CommandQueue):
            raise TypeError('Only CUDA command queues are supported')
        super().__init__(command_queue, allocator)
        self.template = template
        src_shape = list(template.shape)
        dest_shape = list(template.shape)
        if template.dtype_src.kind != 'c':
            if mode != FftMode.FORWARD:
                raise ValueError('R2C transform must use FftMode.FORWARD')
            dest_shape[-1] = template.shape[-1] // 2 + 1
        if template.dtype_dest.kind != 'c':
            if mode != FftMode.INVERSE:
                raise ValueError('C2R transform must use FftMode.INVERSE')
            src_shape[-1] = template.shape[-1] // 2 + 1
        src_dims = tuple(accel.Dimension(d[0], min_padded_size=d[1], exact=True)
                         for d in zip(src_shape, template.padded_shape_src))
        dest_dims = tuple(accel.Dimension(d[0], min_padded_size=d[1], exact=True)
                          for d in zip(dest_shape, template.padded_shape_dest))
        self.slots['src'] = accel.IOSlot(src_dims, template.dtype_src)
        self.slots['dest'] = accel.IOSlot(dest_dims, template.dtype_dest)
        self.slots['work_area'] = accel.IOSlot((template._work_size,), np.uint8)
        self.mode = mode

    def _run(self) -> None:
        src_buffer = self.buffer('src')
        dest_buffer = self.buffer('dest')
        work_area_buffer = self.buffer('work_area')
        context = self.command_queue.context
        cufft = self.template._cufft
        with context, self.template._lock:
            cufft.cufftSetStream(
                self.template._plan,
                self.command_queue._pycuda_stream.handle
            )
            cufft.cufftSetWorkArea(
                self.template._plan,
                work_area_buffer.buffer.ptr
            )
            func = cufft.cufftExec[self.template._fft_type]
            args = [self.template._plan, src_buffer.buffer.ptr, dest_buffer.buffer.ptr]
            if len(func.argtypes) == 4:
                args.append(_Cufft.cufftDirection['CUFFT_' + self.mode.name])
            func(*args)
            if self.template._needs_synchronize_workaround:
                context._pycuda_context.synchronize()
