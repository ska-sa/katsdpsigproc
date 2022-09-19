################################################################################
# Copyright (c) 2019-2021, National Research Foundation (SARAO)
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

"""Abstract base classes for :mod:`.opencl` and :mod:`.cuda`."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence, Optional, Any, Type, TypeVar, Generic
from types import TracebackType

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore


_B = TypeVar('_B')     # buffer type
_RB = TypeVar('_RB')   # raw buffer type
_RS = TypeVar('_RS')   # raw buffer for SVM
_C = TypeVar('_C', bound='AbstractContext')
_D = TypeVar('_D', bound='AbstractDevice')
_E = TypeVar('_E', bound='AbstractEvent')
_K = TypeVar('_K', bound='AbstractKernel')
_P = TypeVar('_P', bound='AbstractProgram')
_Q = TypeVar('_Q', bound='AbstractCommandQueue')
_TQ = TypeVar('_TQ', bound='AbstractTuningCommandQueue')


class AbstractProgram(ABC, Generic[_K]):
    """Abstraction of a program object."""

    @abstractmethod
    def get_kernel(self, name: str) -> 'AbstractKernel':
        """Create a new kernel.

        Parameters
        ----------
        name
            Name of the kernel function
        """


class AbstractKernel(ABC, Generic[_P]):
    """Abstraction of a kernel object.

    The object can be enqueued using :meth:`AbstractCommandQueue.enqueue_kernel`.

    The recommended way to create this object is via
    :meth:`AbstractProgram.get_kernel`.
    """

    @abstractmethod
    def __init__(self, program: _P, name: str) -> None:
        pass


class AbstractEvent(ABC):
    """Abstraction of an event.

    This is more akin to a CUDA event than an OpenCL event, in that it is a
    marker in a command queue rather than associated with a specific command.
    """

    @abstractmethod
    def wait(self) -> None:
        """Block until the event has completed."""

    @abstractmethod
    def time_since(self: _E, prior_event: _E) -> float:
        """Return the time in seconds from `prior_event` to self.

        Unlike the PyCUDA method of the same name, this will wait for the
        events to complete if they have not already.
        """

    @abstractmethod
    def time_till(self: _E, next_event: _E) -> float:
        """Return the time in seconds from this event to `next_event`.

        See  :meth:`time_since`.
        """


class AbstractDevice(ABC, Generic[_C]):
    """Abstraction of a device."""

    @abstractmethod
    def make_context(self) -> 'AbstractContext':
        """Create a new context associated with this device."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name for the device."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return human-readable name for the platform owning the device."""

    @property
    @abstractmethod
    def driver_version(self) -> str:
        """Return human-readable name for the driver version."""

    @property
    @abstractmethod
    def is_cuda(self) -> bool:
        """Whether the device is a CUDA device."""

    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        """Whether device is a GPU."""

    @property
    @abstractmethod
    def is_accelerator(self) -> bool:
        """Whether device is an accelerator (as defined by OpenCL device types)."""

    @property
    @abstractmethod
    def is_cpu(self) -> bool:
        """Whether the device is a CPU."""

    @property
    @abstractmethod
    def simd_group_size(self) -> int:
        """Return the number of workitems that run in lock-step.

        This must only be used to tune performance parameters; there are no
        guarantees about memory coherency, forward progress etc.
        """

    @classmethod
    @abstractmethod
    def get_devices(cls: Type[_D]) -> Sequence[_D]:
        """Return a list of all devices on all platforms."""

    @classmethod
    @abstractmethod
    def get_devices_by_platform(cls: Type[_D]) -> Sequence[Sequence[_D]]:
        """Return a list of all devices, with a sub-list per platform."""


class AbstractContext(ABC, Generic[_B, _RB, _RS, _D, _P, _Q, _TQ]):
    """Abstraction of an OpenCL/CUDA context."""

    @property
    @abstractmethod
    def device(self) -> AbstractDevice:
        """Return the device associated with the context (or the first device, if multiple)."""

    @abstractmethod
    def compile(self, source: str, extra_flags: Optional[List[str]] = None) -> AbstractProgram:
        """Build a program object from source.

        Parameters
        ----------
        source
            Source code
        extra_flags
            Extra parameters to pass to the compiler
        """

    @abstractmethod
    def allocate_raw(self, n_bytes: int) -> _RB:
        """Create an untyped buffer on the device."""

    @abstractmethod
    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike, raw: Optional[_RB] = None) -> _B:
        """Create a typed buffer on the device.

        Parameters
        ----------
        shape
            Shape for the array
        dtype
            Type for the data
        raw
            Memory backing the array (automatically allocated if ``None``)
        """

    @abstractmethod
    def allocate_pinned(self, shape: Tuple[int, ...], dtype: DTypeLike) -> np.ndarray:
        """Create a buffer in host memory that can be efficiently copied to and from the device.

        Parameters
        ----------
        shape
            Shape for the array
        dtype
            Type for the data
        """

    @abstractmethod
    def allocate_svm_raw(self, n_bytes: int) -> _RS:
        """Allocate raw storage that can be passed to :meth:`allocate_svm`."""

    @abstractmethod
    def allocate_svm(self, shape: Tuple[int, ...], dtype: DTypeLike,
                     raw: Optional[_RS] = None) -> np.ndarray:
        """Allocate shared virtual memory."""

    @abstractmethod
    def create_command_queue(self, profile: bool = False) -> 'AbstractCommandQueue':
        """Create a new command queue associated with this context.

        Parameters
        ----------
        profile
            If true, the command queue will support timing kernels
        """

    @abstractmethod
    def create_tuning_command_queue(self) -> 'AbstractTuningCommandQueue':
        """Create a new command queue for doing autotuning."""

    @abstractmethod
    def __enter__(self: _C) -> _C:
        pass

    @abstractmethod
    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        pass


class AbstractCommandQueue(ABC, Generic[_B, _C, _E, _K]):
    """Abstraction of a command queue."""

    context = None     # type: _C

    @abstractmethod
    def enqueue_read_buffer(self, buffer: _B, data: Any, blocking: bool = True) -> None:
        """Copy data from the device to the host.

        Only whole-buffer copies are supported, and the shape and type must
        match. In general, one should use the convenience functions in
        :class:`accel.DeviceArray`.

        Parameters
        ----------
        buffer
            Source
        data
            Target
        blocking
            If true (default) the call blocks until the copy is complete
        """

    @abstractmethod
    def enqueue_write_buffer(self, buffer: _B, data: Any, blocking=True) -> None:
        """Copy data from the host to the device.

        Only whole-buffer copies are supported, and the shape and type must
        match. In general, one should use the convenience functions in
        :class:`accel.DeviceArray`.

        Parameters
        ----------
        buffer
            Target
        data : array-like
            Source
        blocking
            If true (default), the call blocks until the source has been fully
            read (it has not necessarily reached the device).
        """

    @abstractmethod
    def enqueue_copy_buffer_rect(
            self, src_buffer: _B, dest_buffer: _B, src_origin: int, dest_origin: int,
            shape: Sequence[int], src_strides: Sequence[int], dest_strides: Sequence[int]) -> None:
        """Copy a subregion of one buffer to another.

        This is a low-level interface that ignores the shape, strides etc of
        the buffers, and treats them as byte arrays. It also only supports 3 or
        fewer dimensions. Use
        :py:meth:`~katsdpsigproc.accel.DeviceArray.copy_region` for a
        high-level interface.

        Parameters
        ----------
        src_buffer,dest_buffer
            Source and destination buffers
        src_origin,dest_origin
            Offsets for the start of the copy, in bytes
        shape
            Shape of the region to copy (1-3 elements). The first dimension is
            a byte count.
        src_strides,dest_strides
            Strides for the source and destination memory layout, with the same
            length as `shape`. The first element of each must be 1, and each
            element must be a factor of the next element.
        """

    @abstractmethod
    def enqueue_read_buffer_rect(
            self, buffer: _B, data: Any,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        """Copy a region of a buffer to host memory.

        This is a low-level interface that ignores the shape, strides etc of
        the buffers, and treats them as byte arrays. It also only supports 3 or
        fewer dimensions. Use
        :py:meth:`~katsdpsigproc.accel.DeviceArray.set_region` for a high-level
        interface.

        Parameters
        ----------
        buffer
            Source
        data : array-like
            Target
        buffer_origin, data_origin
            Offsets for the start of the copy, in bytes
        shape
            Shape of the region to copy (1-3 elements). The first dimension is
            a byte count.
        buffer_strides,data_strides
            Strides for the destination and source memory layout, with the same
            length as `shape`. The first element of each must be 1, and each
            element must be a factor of the next element.
        blocking
            If true, block until the transfer is complete.
        """

    @abstractmethod
    def enqueue_write_buffer_rect(
            self, buffer: _B, data: Any,
            buffer_origin: int, data_origin: int, shape: Sequence[int],
            buffer_strides: Sequence[int], data_strides: Sequence[int],
            blocking: bool = True) -> None:
        """Copy a region of host memory to a buffer.

        This is a low-level interface that ignores the shape, strides etc of
        the buffers, and treats them as byte arrays. It also only supports 3 or
        fewer dimensions. Use
        :py:meth:`~katsdpsigproc.accel.DeviceArray.set_region` for a high-level
        interface.

        Parameters
        ----------
        buffer
            Target
        data : array-like
            Source
        buffer_origin, data_origin
            Offsets for the start of the copy, in bytes
        shape
            Shape of the region to copy (1-3 elements). The first dimension is
            a byte count.
        buffer_strides,data_strides
            Strides for the destination and source memory layout, with the same
            length as `shape`. The first element of each must be 1, and each
            element must be a factor of the next element.
        blocking
            If true, block until the transfer is complete.
        """

    @abstractmethod
    def enqueue_zero_buffer(self, buffer: _B) -> None:
        """Fill a buffer with zero bytes."""

    @abstractmethod
    def enqueue_kernel(self, kernel: _K, args: Sequence[Any],
                       global_size: Tuple[int, ...], local_size: Tuple[int, ...]) -> None:
        """Enqueue a kernel to the command queue.

        .. warning:: It is not thread-safe to call this function in two threads
            on the same kernel at the same time.

        Parameters
        ----------
        kernel
            Kernel to run
        args
            Arguments to pass to the kernel. Refer to the PyOpenCL/CUDA
            documentation for details. Additionally, this function allows
            a low-level device array to be passed.
        global_size
            Number of work-items in each global dimension
        local_size
            Number of work-items in each local dimension. Must divide
            exactly into `global_size`.
        """

    @abstractmethod
    def enqueue_marker(self) -> AbstractEvent:
        """Create an event at this point in the command queue."""

    @abstractmethod
    def enqueue_wait_for_events(self, events: Sequence[_E]) -> None:
        """Enqueue a barrier to wait for all events in `events`."""

    @abstractmethod
    def flush(self) -> None:
        """Start enqueued work running, but do not wait for it to complete."""

    @abstractmethod
    def finish(self) -> None:
        """Block until all enqueued work has completed."""


class AbstractTuningCommandQueue(AbstractCommandQueue[_B, _C, _E, _K]):
    """Command queue with extra facilities for autotuning.

    It keeps track of kernels that are enqueued since the last call to
    :meth:`start_tuning`, and reports the total time they consume when
    :meth:`stop_tuning` is called.
    """

    @abstractmethod
    def start_tuning(self) -> None:
        pass

    @abstractmethod
    def stop_tuning(self) -> float:
        pass
