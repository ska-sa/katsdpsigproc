################################################################################
# Copyright (c) 2014-2021, National Research Foundation (SARAO)
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

"""Abstraction layer over PyCUDA.

It implements the abstract interfaces defined in :mod:`katsdpsigproc.abc`.
"""

from typing import List, Tuple, Sequence, Optional, Type, TypeVar, Union, Any
from types import TracebackType

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore
import pycuda.driver
import pycuda.compiler
import pycuda.gpuarray
import pycuda.characterize

from .abc import (AbstractProgram, AbstractKernel, AbstractDevice, AbstractContext,
                  AbstractEvent, AbstractCommandQueue, AbstractTuningCommandQueue)


# When building with autodoc_mock_imports, DEFAULT_NVCC_FLAGS is a mock that
# doesn't have a + operator.
if isinstance(pycuda.compiler.DEFAULT_NVCC_FLAGS, list):
    NVCC_FLAGS = pycuda.compiler.DEFAULT_NVCC_FLAGS + ['-lineinfo']
else:
    NVCC_FLAGS = ['-lineinfo']
_T = TypeVar('_T')
_C = TypeVar('_C', bound='Context')
_D = TypeVar('_D', bound='Device')
_AnyBuffer = Union[pycuda.gpuarray.GPUArray, np.ndarray]


class Program(AbstractProgram['Kernel']):
    def __init__(self, pycuda_module: pycuda.driver.Module) -> None:
        self._pycuda_program = pycuda_module

    def get_kernel(self, name: str) -> 'Kernel':
        return Kernel(self, name)


class Kernel(AbstractKernel[Program]):
    def __init__(self, program: Program, name: str) -> None:
        self._pycuda_kernel = program._pycuda_program.get_function(name)


class Event(AbstractEvent):
    def __init__(self, pycuda_event: pycuda.driver.Event) -> None:
        self._pycuda_event = pycuda_event

    def wait(self) -> None:
        self._pycuda_event.synchronize()

    def time_since(self, prior_event: 'Event') -> float:
        prior_event.wait()
        self.wait()
        return 1e-3 * self._pycuda_event.time_since(prior_event._pycuda_event)

    def time_till(self, next_event: 'Event') -> float:
        return next_event.time_since(self)


class Device(AbstractDevice['Context']):
    def __init__(self, pycuda_device: pycuda.driver.Device) -> None:
        self._pycuda_device = pycuda_device

    def make_context(self) -> 'Context':
        pycuda_context = self._pycuda_device.make_context()
        # The above also makes the context current, which we do not
        # want (it leads to errors on termination).
        pycuda_context.pop()
        return Context(pycuda_context)

    @property
    def name(self) -> str:
        return self._pycuda_device.name()

    @property
    def platform_name(self) -> str:
        return 'CUDA'

    @property
    def driver_version(self) -> str:
        return 'CUDA:{0[0]}{0[1]}{0[2]} Driver:{1}'.format(
            pycuda.driver.get_version(), pycuda.driver.get_driver_version())

    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def is_gpu(self) -> bool:
        return True

    @property
    def is_accelerator(self) -> bool:
        return False

    @property
    def is_cpu(self) -> bool:
        return False

    @property
    def simd_group_size(self) -> int:
        return self._pycuda_device.warp_size

    @property
    def compute_capability(self) -> Tuple[int, int]:
        return self._pycuda_device.compute_capability()

    @classmethod
    def get_devices(cls: Type[_D]) -> Sequence[_D]:
        num_devices = pycuda.driver.Device.count()
        return [cls(pycuda.driver.Device(i)) for i in range(num_devices)]

    @classmethod
    def get_devices_by_platform(cls: Type[_D]) -> Sequence[Sequence[_D]]:
        return [cls.get_devices()]


class _RawManaged:
    """Wrap a PyCUDA managed allocation into an opaque object.

    The managed memory API is different to the device memory API, in that one
    cannot allocate a raw pointer. Thus, "raw" allocations are wrapped in this
    class so that they are not accidentally used as numpy arrays.
    """

    def __init__(self, wrapped: np.ndarray) -> None:
        self._wrapped = wrapped

    def get_array(self, shape: Tuple[int, ...], dtype: DTypeLike) -> np.ndarray:
        """Return a view of (a prefix of) the memory, with the given shape and type."""
        size = int(np.product(shape)) * np.dtype(dtype).itemsize
        return self._wrapped[:size].view(dtype).reshape(shape)


class Context(AbstractContext[pycuda.gpuarray.GPUArray,
                              pycuda.driver.DeviceAllocation,
                              _RawManaged,
                              Device, Program, 'CommandQueue', 'TuningCommandQueue']):
    def __init__(self, pycuda_context: pycuda.driver.Context) -> None:
        self._pycuda_context = pycuda_context

    @property
    def device(self) -> Device:
        with self:
            return Device(pycuda.driver.Context.get_device())

    def compile(self, source: str, extra_flags: Optional[List[str]] = None) -> Program:
        if extra_flags is None:
            extra_flags = []
        with self:
            module = pycuda.compiler.SourceModule(source, options=NVCC_FLAGS + extra_flags)
            return Program(module)

    def allocate_raw(self, n_bytes: int) -> pycuda.driver.DeviceAllocation:
        with self:
            return pycuda.driver.mem_alloc(n_bytes)

    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 raw: Optional[pycuda.driver.DeviceAllocation] = None) -> pycuda.gpuarray.GPUArray:
        with self:
            return pycuda.gpuarray.GPUArray(shape, dtype, gpudata=raw)

    def allocate_pinned(self, shape: Tuple[int, ...], dtype: DTypeLike) -> np.ndarray:
        with self:
            return pycuda.driver.pagelocked_empty(shape, dtype)

    def allocate_svm_raw(self, n_bytes: int) -> _RawManaged:
        with self:
            return _RawManaged(pycuda.driver.managed_empty(
                (n_bytes,), np.uint8, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL))

    def allocate_svm(self, shape: Tuple[int, ...], dtype: DTypeLike,
                     raw: Optional[_RawManaged] = None) -> np.ndarray:
        with self:
            if raw is None:
                return pycuda.driver.managed_empty(
                    shape, dtype, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL)
            else:
                return raw.get_array(shape, dtype)

    def create_command_queue(self, profile: bool = False) -> 'CommandQueue':
        return CommandQueue(self, profile=profile)

    def create_tuning_command_queue(self) -> 'TuningCommandQueue':
        return TuningCommandQueue(self)

    def __enter__(self: _C) -> _C:
        self._pycuda_context.push()
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self._pycuda_context.pop()


class CommandQueue(AbstractCommandQueue[pycuda.gpuarray.GPUArray, Context, Event, Kernel]):
    def __init__(self, context: Context, pycuda_stream: Optional[pycuda.driver.Stream] = None,
                 profile: bool = False) -> None:
        self.context = context
        if pycuda_stream is None:
            with context:
                self._pycuda_stream = pycuda.driver.Stream()
        else:
            self._pycuda_stream = pycuda_stream

    def enqueue_read_buffer(self, buffer: pycuda.gpuarray.GPUArray, data: Any,
                            blocking: bool = True) -> None:
        with self.context:
            # CUDA doesn't support synchronous transfers sequenced in a stream
            # (PyCUDA simply doesn't pass on the stream argument), so use an
            # async transfer and then block if necessary.
            buffer.get_async(self._pycuda_stream, data)
            if blocking:
                self._pycuda_stream.synchronize()

    def enqueue_write_buffer(self, buffer: pycuda.gpuarray.GPUArray, data: Any,
                             blocking: bool = True) -> None:
        with self.context:
            # See comment in enqueue_read_buffer
            buffer.set_async(data, self._pycuda_stream)
            if blocking:
                self._pycuda_stream.synchronize()

    @staticmethod
    def _get_device_pointer(buffer: _AnyBuffer) -> int:
        """Retrieve the device pointer from either a GPUArray or a managed memory allocation."""
        if isinstance(buffer, pycuda.gpuarray.GPUArray):
            return int(buffer.gpudata)
        else:
            return buffer.base.get_device_pointer()  # type: ignore

    def enqueue_copy_buffer_rect(
            self, src_buffer: _AnyBuffer,
            dest_buffer: _AnyBuffer,
            src_origin: int, dest_origin: int,
            shape: Sequence[int], src_strides: Sequence[int], dest_strides: Sequence[int]) -> None:
        with self.context:
            assert src_strides[0] == 1
            assert dest_strides[0] == 1
            assert len(shape) > 0 and len(shape) <= 3
            if len(shape) == 1:
                pycuda.driver.memcpy_dtod_async(
                    self._get_device_pointer(dest_buffer) + dest_origin,
                    self._get_device_pointer(src_buffer) + src_origin,
                    shape[0],
                    self._pycuda_stream)
            else:
                if len(shape) == 3:
                    copy = pycuda.driver.Memcpy3D()
                else:
                    copy = pycuda.driver.Memcpy2D()
                copy.src_pitch = src_strides[1]
                copy.set_src_device(self._get_device_pointer(src_buffer) + src_origin)
                copy.dst_pitch = dest_strides[1]
                copy.set_dst_device(self._get_device_pointer(dest_buffer) + dest_origin)
                copy.width_in_bytes = shape[0]
                copy.height = shape[1]
                if len(shape) >= 3:
                    assert src_strides[2] % src_strides[1] == 0
                    assert dest_strides[2] % dest_strides[1] == 0
                    copy.src_height = src_strides[2] // src_strides[1]
                    copy.dst_height = dest_strides[2] // dest_strides[1]
                    copy.depth = shape[2]
                copy(self._pycuda_stream)

    @classmethod
    def _byte_buffer(cls, data: np.ndarray) -> np.ndarray:
        """Reinterpret a contiguous array as an array of bytes."""
        view = data.view()
        view.shape = (data.size,)   # Reshape while disallowing copy
        return view.view(np.uint8)

    def enqueue_read_buffer_rect(
            self, buffer: _AnyBuffer, data: Any,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        with self.context:
            assert buffer_strides[0] == 1
            assert data_strides[0] == 1
            assert len(shape) > 0 and len(shape) <= 3
            data = self._byte_buffer(data)
            if len(shape) == 1:
                pycuda.driver.memcpy_dtoh_async(
                    data[data_origin : data_origin + shape[0]],
                    self._get_device_pointer(buffer) + buffer_origin,
                    self._pycuda_stream)
            else:
                if len(shape) == 3:
                    copy = pycuda.driver.Memcpy3D()
                else:
                    copy = pycuda.driver.Memcpy2D()
                copy.src_pitch = buffer_strides[1]
                copy.set_src_device(self._get_device_pointer(buffer) + buffer_origin)
                copy.dst_pitch = data_strides[1]
                copy.set_dst_host(data[data_origin:])
                copy.width_in_bytes = shape[0]
                copy.height = shape[1]
                if len(shape) >= 3:
                    assert data_strides[2] % data_strides[1] == 0
                    assert buffer_strides[2] % buffer_strides[1] == 0
                    copy.src_height = buffer_strides[2] // buffer_strides[1]
                    copy.dst_height = data_strides[2] // data_strides[1]
                    copy.depth = shape[2]
                copy(self._pycuda_stream)
            if blocking:
                self._pycuda_stream.synchronize()

    def enqueue_write_buffer_rect(
            self, buffer: _AnyBuffer, data: Any,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        with self.context:
            assert buffer_strides[0] == 1
            assert data_strides[0] == 1
            assert len(shape) > 0 and len(shape) <= 3
            data = self._byte_buffer(data)
            if len(shape) == 1:
                pycuda.driver.memcpy_htod_async(
                    self._get_device_pointer(buffer) + buffer_origin,
                    data[data_origin : data_origin + shape[0]],
                    self._pycuda_stream)
            else:
                if len(shape) == 3:
                    copy = pycuda.driver.Memcpy3D()
                else:
                    copy = pycuda.driver.Memcpy2D()
                copy.src_pitch = data_strides[1]
                copy.set_src_host(data[data_origin:])
                copy.dst_pitch = buffer_strides[1]
                copy.set_dst_device(self._get_device_pointer(buffer) + buffer_origin)
                copy.width_in_bytes = shape[0]
                copy.height = shape[1]
                if len(shape) >= 3:
                    assert data_strides[2] % data_strides[1] == 0
                    assert buffer_strides[2] % buffer_strides[1] == 0
                    copy.src_height = data_strides[2] // data_strides[1]
                    copy.dst_height = buffer_strides[2] // buffer_strides[1]
                    copy.depth = shape[2]
                copy(self._pycuda_stream)
            if blocking:
                self._pycuda_stream.synchronize()

    def enqueue_zero_buffer(self, buffer: _AnyBuffer) -> None:
        with self.context:
            if isinstance(buffer, pycuda.gpuarray.GPUArray):
                pycuda.driver.memset_d8_async(
                    buffer.gpudata, 0, buffer.mem_size * buffer.dtype.itemsize,
                    stream=self._pycuda_stream)
            else:
                # managed memory
                pycuda.driver.memset_d8_async(
                    buffer.base.get_device_pointer(),     # type: ignore
                    0,
                    buffer.size * buffer.dtype.itemsize,
                    stream=self._pycuda_stream)

    def enqueue_kernel(self, kernel: Kernel, args: Sequence[Any],
                       global_size: Tuple[int, ...], local_size: Tuple[int, ...]) -> None:
        assert len(global_size) == len(local_size)
        block = [1, 1, 1]
        grid = [1, 1, 1]
        for i in range(len(global_size)):
            assert global_size[i] % local_size[i] == 0
            block[i] = local_size[i]
            grid[i] = global_size[i] // local_size[i]
        with self.context:
            kernel._pycuda_kernel(*args, block=tuple(block), grid=tuple(grid),
                                  stream=self._pycuda_stream)

    def enqueue_marker(self) -> Event:
        with self.context:
            event = pycuda.driver.Event(pycuda.driver.event_flags.BLOCKING_SYNC)
            event.record(self._pycuda_stream)
        return Event(event)

    def enqueue_wait_for_events(self, events: Sequence[Event]) -> None:
        with self.context:
            for event in events:
                self._pycuda_stream.wait_for_event(event._pycuda_event)

    def flush(self) -> None:
        with self.context:
            # This anecdotally flushes, but isn't tested
            self._pycuda_stream.is_done()

    def finish(self) -> None:
        with self.context:
            self._pycuda_stream.synchronize()


class TuningCommandQueue(CommandQueue,
                         AbstractTuningCommandQueue[pycuda.gpuarray.GPUArray,
                                                    Context, Event, Kernel]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_tuning = False
        self._start_event = None    # type: Optional[Event]

    def start_tuning(self) -> None:
        self.is_tuning = True
        self._start_event = self.enqueue_marker()

    def stop_tuning(self) -> float:
        elapsed = 0.0
        if self._start_event is not None:
            end_event = self.enqueue_marker()
            elapsed = self._start_event.time_till(end_event)
        self._start_event = None
        self.is_tuning = False
        return elapsed
