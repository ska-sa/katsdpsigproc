"""Fast Fourier Transforms.

Currently this module only supports CUDA, using CUFFT.
"""

from enum import Enum
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any  # type: ignore
import skcuda.fft

from . import accel, cuda
from .accel import AbstractAllocator
from .abc import AbstractContext, AbstractCommandQueue


class FftMode(Enum):
    FORWARD = 0
    INVERSE = 1


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

    Parameters
    ----------
    context
        Context for the operation
    N
        Number of dimensions for the transform
    shape
        Shape of the input (N or more dimensions)
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
        self.context: cuda.Context = context
        self.shape = shape
        self.dtype_src = np.dtype(dtype_src)
        self.dtype_dest = np.dtype(dtype_dest)
        self.padded_shape_src = padded_shape_src
        self.padded_shape_dest = padded_shape_dest
        # CUDA 7.0 CUFFT has a bug where kernels are run in the default stream
        # instead of the requested one, if dimensions are up to 1920. There is
        # a patch, but there is no query to detect whether it has been
        # applied.
        self._needs_synchronize_workaround = (
            skcuda.cufft.cufftGetVersion() < 8000 and any(x <= 1920 for x in shape[:N])
        )
        batches = int(np.product(padded_shape_src[:-N]))
        with context:
            self.plan = skcuda.fft.Plan(
                shape[-N:], dtype_src, dtype_dest, batches,
                inembed=np.array(padded_shape_src[-N:], np.int32),
                istride=1,
                idist=int(np.product(padded_shape_src[-N:])),
                onembed=np.array(padded_shape_dest[-N:], np.int32),
                ostride=1,
                odist=int(np.product(padded_shape_dest[-N:])),
                auto_allocate=False)
        # The stream and work area are associated with the plan rather than
        # the execution, so we need to serialise executions.
        self.lock = threading.RLock()

    def instantiate(self, *args, **kwargs):
        return Fft(self, *args, **kwargs)


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
        self.slots['work_area'] = accel.IOSlot((template.plan.worksize,), np.uint8)
        self.mode = mode

    def _run(self) -> None:
        src_buffer = self.buffer('src')
        dest_buffer = self.buffer('dest')
        work_area_buffer = self.buffer('work_area')
        context = self.command_queue.context
        with context, self.template.lock:
            skcuda.cufft.cufftSetStream(self.template.plan.handle,
                                        self.command_queue._pycuda_stream.handle)
            self.template.plan.set_work_area(work_area_buffer.buffer)
            if self.mode == FftMode.FORWARD:
                skcuda.fft.fft(src_buffer.buffer, dest_buffer.buffer, self.template.plan)
            else:
                skcuda.fft.ifft(src_buffer.buffer, dest_buffer.buffer, self.template.plan)
            if self.template._needs_synchronize_workaround:
                context._pycuda_context.synchronize()
