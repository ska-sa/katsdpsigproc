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

"""Abstraction layer over PyOpenCL.

It implements the abstract interfaces defined by :mod:`katsdpsigproc.abc`.
"""

from typing import List, Tuple, Sequence, Optional, Callable, Type, TypeVar, Union, Any
from types import TracebackType

import pyopencl
import pyopencl.array
import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore

from .abc import (AbstractProgram, AbstractKernel, AbstractDevice, AbstractContext,
                  AbstractEvent, AbstractCommandQueue, AbstractTuningCommandQueue)


_T = TypeVar('_T')
_D = TypeVar('_D', bound='Device')
_AnyBuffer = Union['_DummyArray', pyopencl.array.Array]


class _DummyArray:
    """Trivial dummy for :class:`pyopencl.array.Array`, that just has a `data` attribute."""

    def __init__(self, data: pyopencl.Buffer) -> None:
        self.data = data


class _PinnedAMD(np.ndarray):
    """Pinned memory abstraction for AMD GPUs.

    Copies to the device
    are done by unmapping the buffer, enqueuing the copy, and remapping
    the buffer. This is based on AMD's optimization guide.
    """

    _mapping: pyopencl.MemoryMap = None
    _buffer: pyopencl.Buffer
    _array: _DummyArray

    def __new__(cls, context: 'Context', queue: 'CommandQueue',
                shape: Tuple[int, ...], dtype: DTypeLike) -> '_PinnedAMD':
        dtype = np.dtype(dtype)
        n_bytes = int(np.product(shape)) * dtype.itemsize
        # Do not add READ or WRITE to the flags: doing so seems to cause AMD
        # drivers to allocate GPU memory.
        buffer = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.ALLOC_HOST_PTR,
            n_bytes)
        array = pyopencl.enqueue_map_buffer(
            queue,
            buffer,
            pyopencl.map_flags.READ | pyopencl.map_flags.WRITE,
            0, shape, dtype)[0]
        mapping = array.base               # type: pyopencl.MemoryMap
        array = array.view(_PinnedAMD)
        array._buffer = buffer
        array._mapping = mapping
        array._array = _DummyArray(buffer)
        return array

    def _enqueue_transfer(self, queue: 'CommandQueue', blocking: bool,
                          transfer: Callable[[], None]) -> None:
        if not hasattr(self, '_buffer'):
            raise ValueError('cannot copy to/from a _PinnedAMD view')
        self._mapping.release(queue._pyopencl_command_queue)
        transfer()
        # The shape and dtype don't matter here as long as we map the entire buffer,
        # because we're going to throw away the array wrapper and just keep the
        # MemoryMap object.
        array = pyopencl.enqueue_map_buffer(
            queue._pyopencl_command_queue,
            self._buffer, pyopencl.map_flags.READ | pyopencl.map_flags.WRITE,
            0, (self._buffer.size,), np.uint8, is_blocking=blocking)[0]
        self._mapping = array.base

    def enqueue_write_buffer(self, queue: 'CommandQueue', buffer: pyopencl.array.Array,
                             blocking: bool) -> None:
        """Enqueue a copy from the host buffer to device memory.

        The host memory must not be touched while the copy is in progress.

        Parameters
        ----------
        queue
            Queue for the copy
        buffer
            Target buffer
        blocking
            If `True`, wait for the copy to complete before returning
        """
        def transfer():
            queue.enqueue_copy_buffer(self._array, buffer)
        self._enqueue_transfer(queue, blocking, transfer)

    def enqueue_read_buffer(self, queue: 'CommandQueue', buffer: pyopencl.array.Array,
                            blocking: bool) -> None:
        """Enqueue a copy to the host buffer from device memory.

        The host memory must not be touched while the copy is in progress.

        Parameters
        ----------
        queue
            Queue for the copy
        buffer
            Source buffer
        blocking
            If `True`, wait for the copy to complete before returning
        """
        def transfer():
            queue.enqueue_copy_buffer(buffer, self._array)
        self._enqueue_transfer(queue, blocking, transfer)

    def enqueue_write_buffer_rect(
            self, queue: 'CommandQueue', buffer: pyopencl.array.Array,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        def transfer():
            queue.enqueue_copy_buffer_rect(
                self._array, buffer, data_origin, buffer_origin, shape,
                data_strides, buffer_strides)
        self._enqueue_transfer(queue, blocking, transfer)

    def enqueue_read_buffer_rect(
            self, queue: 'CommandQueue', buffer: pyopencl.array.Array,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        def transfer():
            queue.enqueue_copy_buffer_rect(
                buffer, self._array, buffer_origin, data_origin, shape,
                buffer_strides, data_strides)
        self._enqueue_transfer(queue, blocking, transfer)


class Program(AbstractProgram['Kernel']):
    def __init__(self, pyopencl_program: pyopencl.Program) -> None:
        self._pyopencl_program = pyopencl_program

    def get_kernel(self, name: str) -> 'Kernel':
        return Kernel(self, name)


class Kernel(AbstractKernel[Program]):
    def __init__(self, program: Program, name: str) -> None:
        self._pyopencl_kernel = pyopencl.Kernel(program._pyopencl_program, name)


class Event(AbstractEvent):
    def __init__(self, pyopencl_event: pyopencl.Event) -> None:
        self._pyopencl_event = pyopencl_event

    def wait(self) -> None:
        self._pyopencl_event.wait()

    def time_since(self, prior_event: 'Event') -> float:
        prior_event.wait()
        self.wait()
        return 1e-9 * (self._pyopencl_event.profile.start
                       - prior_event._pyopencl_event.profile.end)

    def time_till(self, next_event: 'Event') -> float:
        return next_event.time_since(self)


class Device(AbstractDevice['Context']):
    """Abstraction of an OpenCL device."""

    def __init__(self, pyopencl_device: pyopencl.Device) -> None:
        self._pyopencl_device = pyopencl_device

    def make_context(self) -> 'Context':
        return Context(pyopencl.Context([self._pyopencl_device]))

    @property
    def name(self) -> str:
        return self._pyopencl_device.name

    @property
    def platform_name(self) -> str:
        return self._pyopencl_device.platform.name

    @property
    def driver_version(self) -> str:
        return self._pyopencl_device.driver_version

    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def is_gpu(self) -> bool:
        return self._pyopencl_device.type & pyopencl.device_type.GPU

    @property
    def is_accelerator(self) -> bool:
        return self._pyopencl_device.type & pyopencl.device_type.ACCELERATOR

    @property
    def is_cpu(self) -> bool:
        return self._pyopencl_device.type & pyopencl.device_type.CPU

    @property
    def simd_group_size(self) -> int:
        return pyopencl.characterize.get_simd_group_size(self._pyopencl_device, 4)

    @classmethod
    def _get_platforms(cls) -> List[pyopencl.Platform]:
        """Return all platforms."""
        try:
            return pyopencl.get_platforms()
        except pyopencl.LogicError:
            # OpenCL considers it an error if there are no platforms
            # available.
            return []

    @classmethod
    def get_devices(cls: Type[_D]) -> Sequence[_D]:
        ans = []
        for platform in cls._get_platforms():
            for device in platform.get_devices():
                ans.append(cls(device))
        return ans

    @classmethod
    def get_devices_by_platform(cls: Type[_D]) -> Sequence[Sequence[_D]]:
        ans = []
        for platform in cls._get_platforms():
            ans.append([cls(device) for device in platform.get_devices()])
        return ans


class Context(AbstractContext[pyopencl.array.Array, pyopencl.Buffer, None,
                              Device, Program, 'CommandQueue', 'TuningCommandQueue']):
    """Abstraction of an OpenCL context."""

    def __init__(self, pyopencl_context: pyopencl.Context) -> None:
        self._pyopencl_context = pyopencl_context
        device = pyopencl_context.devices[0]
        self._internal_queue = pyopencl.CommandQueue(pyopencl_context, device)
        self._force_pinned_amd = False   # Used for unit tests

    @property
    def device(self) -> Device:
        return Device(self._pyopencl_context.devices[0])

    def compile(self, source: str, extra_flags: Optional[List[str]] = None) -> Program:
        # source is passed through str because it might arrive as Unicode,
        # triggering a warning.
        # TODO: can remove this now that it's Python 3-only.
        program = pyopencl.Program(self._pyopencl_context, str(source))
        program.build(extra_flags if extra_flags is not None else [])
        return Program(program)

    def allocate_raw(self, n_bytes: int) -> pyopencl.Buffer:
        return pyopencl.Buffer(self._pyopencl_context, pyopencl.mem_flags.READ_WRITE, n_bytes)

    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 raw: Optional[pyopencl.Buffer] = None) -> pyopencl.array.Array:
        return pyopencl.array.Array(self._pyopencl_context, shape, dtype, data=raw)

    def allocate_pinned(self, shape: Tuple[int, ...], dtype: DTypeLike) -> np.ndarray:
        dtype = np.dtype(dtype)
        device = self._pyopencl_context.devices[0]
        if (self._force_pinned_amd
            or ((device.type & pyopencl.device_type.GPU)
                and device.platform.name == 'AMD Accelerated Parallel Processing')):
            return _PinnedAMD(self._pyopencl_context, self._internal_queue, shape, dtype)
        elif device.platform.name == 'NVIDIA CUDA':
            # Based on NVIDIA's recommendation: create a device buffer and
            # leave it mapped permanently.
            n_bytes = int(np.product(shape)) * dtype.itemsize
            buf = pyopencl.Buffer(
                self._pyopencl_context,
                pyopencl.mem_flags.ALLOC_HOST_PTR | pyopencl.mem_flags.READ_ONLY,
                n_bytes)
            return pyopencl.enqueue_map_buffer(
                self._internal_queue,
                buf,
                pyopencl.map_flags.READ | pyopencl.map_flags.WRITE,
                0, shape, dtype)[0]
        else:
            return np.empty(shape, dtype)

    def allocate_svm_raw(self, n_bytes: int) -> None:
        raise NotImplementedError("PyOpenCL does not support OpenCL Shared Virtual Memory")

    def allocate_svm(self, shape: Tuple[int, ...], dtype: DTypeLike,
                     raw: Optional[None] = None) -> np.ndarray:
        raise NotImplementedError("PyOpenCL does not support OpenCL Shared Virtual Memory")

    def create_command_queue(self, profile: bool = False) -> 'CommandQueue':
        return CommandQueue(self, profile=profile)

    def create_tuning_command_queue(self) -> 'TuningCommandQueue':
        return TuningCommandQueue(self)

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        pass


class CommandQueue(AbstractCommandQueue[pyopencl.array.Array, Context, Event, Kernel]):
    """Abstraction of a command queue.

    If no existing command queue is passed to the constructor, a new one is
    created.

    Parameters
    ----------
    context
        Context owning the queue
    pyopencl_command_queue
        Existing command queue to wrap
    profile
        If true and `pyopencl_command_queue` is omitted, enabling profiling
        (timing) on the queue.
    """

    def __init__(self, context: Context,
                 pyopencl_command_queue: Optional[pyopencl.CommandQueue] = None,
                 profile: bool = False) -> None:
        self.context = context
        if pyopencl_command_queue is None:
            pyopencl_device = context._pyopencl_context.devices[0]
            properties = 0
            if profile:
                properties |= pyopencl.command_queue_properties.PROFILING_ENABLE
            pyopencl_command_queue = pyopencl.CommandQueue(
                context._pyopencl_context, pyopencl_device,
                properties)
        self._pyopencl_command_queue = pyopencl_command_queue

    def enqueue_read_buffer(self, buffer: pyopencl.array.Array, data: Any,
                            blocking: bool = True) -> None:
        if hasattr(data, 'enqueue_read_buffer'):
            data.enqueue_read_buffer(self, buffer, blocking)
        else:
            buffer.get(ary=data, queue=self._pyopencl_command_queue, async_=not blocking)

    def enqueue_write_buffer(self, buffer: pyopencl.array.Array, data: Any,
                             blocking: bool = True) -> None:
        if hasattr(data, 'enqueue_write_buffer'):
            data.enqueue_write_buffer(self, buffer, blocking)
        else:
            buffer.set(ary=data, queue=self._pyopencl_command_queue, async_=not blocking)

    def enqueue_copy_buffer(
            self, src_buffer: _AnyBuffer, dest_buffer: _AnyBuffer) -> None:
        pyopencl.enqueue_copy(
            self._pyopencl_command_queue,
            src=src_buffer.data, dest=dest_buffer.data)

    def enqueue_copy_buffer_rect(
            self, src_buffer: _AnyBuffer, dest_buffer: _AnyBuffer,
            src_origin: int, dest_origin: int,
            shape: Sequence[int], src_strides: Sequence[int], dest_strides: Sequence[int]) -> None:
        assert src_strides[0] == 1
        assert dest_strides[0] == 1
        pyopencl.enqueue_copy(
            self._pyopencl_command_queue,
            src=src_buffer.data, dest=dest_buffer.data,
            src_origin=(src_origin,), dst_origin=(dest_origin,),
            region=shape,
            src_pitches=src_strides[1:], dst_pitches=dest_strides[1:])

    def enqueue_read_buffer_rect(
            self, buffer: pyopencl.array.Array, data: Any,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        assert buffer_strides[0] == 1
        assert data_strides[0] == 1
        if hasattr(data, 'enqueue_read_buffer_rect'):
            data.enqueue_read_buffer_rect(self, buffer, buffer_origin, data_origin, shape,
                                          buffer_strides, data_strides, blocking)
        else:
            pyopencl.enqueue_copy(
                self._pyopencl_command_queue,
                src=buffer.data, dest=data,
                host_origin=(data_origin,), buffer_origin=(buffer_origin,),
                region=shape,
                host_pitches=data_strides[1:], buffer_pitches=buffer_strides[1:])

    def enqueue_write_buffer_rect(
            self, buffer: pyopencl.array.Array, data: Any,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        assert buffer_strides[0] == 1
        assert data_strides[0] == 1
        if hasattr(data, 'enqueue_write_buffer_rect'):
            data.enqueue_write_buffer_rect(self, buffer, buffer_origin, data_origin, shape,
                                           buffer_strides, data_strides, blocking)
        else:
            pyopencl.enqueue_copy(
                self._pyopencl_command_queue,
                src=data, dest=buffer.data,
                host_origin=(data_origin,), buffer_origin=(buffer_origin,),
                region=shape,
                host_pitches=data_strides[1:], buffer_pitches=buffer_strides[1:])

    def enqueue_zero_buffer(self, buffer: pyopencl.array.Array) -> None:
        pyopencl.enqueue_fill_buffer(self._pyopencl_command_queue, buffer.data,
                                     np.uint8(0), 0, buffer.data.get_info(pyopencl.mem_info.SIZE))

    @classmethod
    def _raw_arg(cls, arg: Any) -> Any:
        if isinstance(arg, pyopencl.array.Array):
            return arg.data
        else:
            return arg

    def _enqueue_kernel(self, kernel: Kernel, args: Sequence[Any],
                        global_size: Tuple[int, ...],
                        local_size: Tuple[int, ...]) -> pyopencl.Event:
        """Enqueue a kernel to the command queue.

        This version returns the OpenCL event object. Refer to
        :meth:`enqueue_queue` for the public interface.
        """
        # PyOpenCL doesn't allow Array objects to be passed through
        args = [self._raw_arg(x) for x in args]
        # OpenCL allows local_size to be None, but this is not portable to
        # CUDA
        assert local_size is not None
        assert len(local_size) == len(global_size)
        return kernel._pyopencl_kernel(self._pyopencl_command_queue,
                                       global_size, local_size, *args)

    def enqueue_kernel(self, kernel: Kernel, args: Sequence[Any],
                       global_size: Tuple[int, ...], local_size: Tuple[int, ...]) -> None:
        self._enqueue_kernel(kernel, args, global_size, local_size)

    def enqueue_marker(self) -> Event:
        return Event(pyopencl.enqueue_marker(self._pyopencl_command_queue))

    def enqueue_wait_for_events(self, events: Sequence[Event]) -> None:
        # OpenCL has some odd semantics for an empty wait list, hence the check
        if events:
            pyopencl.enqueue_barrier(self._pyopencl_command_queue,
                                     [x._pyopencl_event for x in events])

    def flush(self) -> None:
        self._pyopencl_command_queue.flush()

    def finish(self) -> None:
        self._pyopencl_command_queue.finish()


class TuningCommandQueue(CommandQueue,
                         AbstractTuningCommandQueue[pyopencl.array.Array, Context, Event, Kernel]):
    """Command queue with extra facilities for autotuning.

    It keeps track of kernels that are enqueued since the last call to
    :meth:`start_tuning`, and reports the total time they consume when
    :meth:`stop_tuning` is called.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs['profile'] = True
        super().__init__(*args, **kwargs)
        self.is_tuning = False
        self.events = []        # type: List[pyopencl.Event]

    def start_tuning(self) -> None:
        self.is_tuning = True
        self.events = []

    def enqueue_kernel(self, kernel: Kernel, args: Sequence[Any],
                       global_size: Tuple[int, ...], local_size: Tuple[int, ...]) -> None:
        if not self.is_tuning:
            super().enqueue_kernel(kernel, args, global_size, local_size)
        else:
            event = self._enqueue_kernel(kernel, args, global_size, local_size)
            self.events.append(event)

    def stop_tuning(self) -> float:
        self.finish()
        elapsed = 0.0
        if self.events:
            start = self.events[0].profile.start
            end = self.events[-1].profile.end
            elapsed = (end - start) * 1e-9
        self.is_tuning = False
        self.events = []
        return elapsed
