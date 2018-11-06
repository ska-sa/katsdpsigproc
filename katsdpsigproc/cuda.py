"""Abstraction layer over PyCUDA to present an interface that is common
between CUDA and OpenCL. For documentation on these classes, refer to
the documentation for :mod:`katsdpsigproc.opencl`, which presents the same
interfaces.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from six.moves import range

import pycuda.driver
import pycuda.compiler
import pycuda.gpuarray
import pycuda.characterize


NVCC_FLAGS = pycuda.compiler.DEFAULT_NVCC_FLAGS + ['-lineinfo']


class Program(object):
    def __init__(self, pycuda_module):
        self._pycuda_program = pycuda_module

    def get_kernel(self, name):
        return Kernel(self, name)


class Kernel(object):
    def __init__(self, program, name):
        self._pycuda_kernel = program._pycuda_program.get_function(name)


class Event(object):
    def __init__(self, pycuda_event):
        self._pycuda_event = pycuda_event

    def wait(self):
        self._pycuda_event.synchronize()

    def time_since(self, prior_event):
        prior_event.wait()
        self.wait()
        return 1e-3 * self._pycuda_event.time_since(prior_event._pycuda_event)

    def time_till(self, next_event):
        return next_event.time_since(self)


class Device(object):
    def __init__(self, pycuda_device):
        self._pycuda_device = pycuda_device

    def make_context(self):
        pycuda_context = self._pycuda_device.make_context()
        # The above also makes the context current, which we do not
        # want (it leads to errors on termination).
        pycuda_context.pop()
        return Context(pycuda_context)

    @property
    def name(self):
        return self._pycuda_device.name()

    @property
    def platform_name(self):
        return 'CUDA'

    @property
    def driver_version(self):
        return 'CUDA:{0[0]}{0[1]}{0[2]} Driver:{1}'.format(
            pycuda.driver.get_version(), pycuda.driver.get_driver_version())

    @property
    def is_cuda(self):
        return True

    @property
    def is_gpu(self):
        return True

    @property
    def is_accelerator(self):
        return False

    @property
    def is_cpu(self):
        return False

    @property
    def simd_group_size(self):
        return self._pycuda_device.warp_size

    @classmethod
    def get_devices(cls):
        num_devices = pycuda.driver.Device.count()
        return [Device(pycuda.driver.Device(i)) for i in range(num_devices)]


class _RawManaged(object):
    """Wraps a PyCUDA managed allocation into an opaque object.

    The managed memory API is different to the device memory API, in that one
    cannot allocate a raw pointer. Thus, "raw" allocations are wrapped in this
    class so that they are not accidentally used as numpy arrays.
    """
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def get_array(self, shape, dtype):
        """Returns a view of (a prefix of) the memory, with the given shape and
        type."""
        size = int(np.product(shape)) * np.dtype(dtype).itemsize
        return self._wrapped[:size].view(dtype).reshape(shape)


class Context(object):
    def __init__(self, pycuda_context):
        self._pycuda_context = pycuda_context

    @property
    def device(self):
        with self:
            return Device(pycuda.driver.Context.get_device())

    def compile(self, source, extra_flags=None):
        with self:
            module = pycuda.compiler.SourceModule(source, options=NVCC_FLAGS + extra_flags)
            return Program(module)

    def allocate_raw(self, n_bytes):
        with self:
            return pycuda.driver.mem_alloc(n_bytes)

    def allocate(self, shape, dtype, raw=None):
        with self:
            return pycuda.gpuarray.GPUArray(shape, dtype, gpudata=raw)

    def allocate_pinned(self, shape, dtype):
        with self:
            return pycuda.driver.pagelocked_empty(shape, dtype)

    def allocate_svm_raw(self, n_bytes):
        with self:
            return _RawManaged(pycuda.driver.managed_empty(
                (n_bytes,), np.uint8, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL))

    def allocate_svm(self, shape, dtype, raw=None):
        with self:
            if raw is None:
                return pycuda.driver.managed_empty(
                    shape, dtype, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL)
            else:
                return raw.get_array(shape, dtype)

    def create_command_queue(self, profile=False):
        return CommandQueue(self, profile=profile)

    def create_tuning_command_queue(self):
        return TuningCommandQueue(self)

    def __enter__(self):
        self._pycuda_context.push()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pycuda_context.pop()
        return False


class CommandQueue(object):
    def __init__(self, context, pycuda_stream=None, profile=False):
        self.context = context
        if pycuda_stream is None:
            with context:
                pycuda_stream = pycuda.driver.Stream()
        self._pycuda_stream = pycuda_stream

    def enqueue_read_buffer(self, buffer, data, blocking=True):
        with self.context:
            # CUDA doesn't support synchronous transfers sequenced in a stream
            # (PyCUDA simply doesn't pass on the stream argument), so use an
            # async transfer and then block if necessary.
            buffer.get_async(self._pycuda_stream, data)
            if blocking:
                self._pycuda_stream.synchronize()

    def enqueue_write_buffer(self, buffer, data, blocking=True):
        with self.context:
            # See comment in enqueue_read_buffer
            buffer.set_async(data, self._pycuda_stream)
            if blocking:
                self._pycuda_stream.synchronize()

    @staticmethod
    def _get_device_pointer(buffer):
        """Retrieves the device pointer from either a GPUArray or a managed
        memory allocation.
        """
        if isinstance(buffer, pycuda.gpuarray.GPUArray):
            return int(buffer.gpudata)
        else:
            return buffer.base.get_device_pointer()

    def enqueue_copy_buffer_rect(
            self, src_buffer, dest_buffer, src_origin, dest_origin,
            shape, src_strides, dest_strides):
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
    def _byte_buffer(cls, data):
        """Reinterpret a contiguous array as an array of bytes"""
        view = data.view()
        view.shape = data.size   # Reshape while disallowing copy
        return view.view(np.uint8)

    def enqueue_read_buffer_rect(
            self, buffer, data, buffer_origin, data_origin, shape,
            buffer_strides, data_strides, blocking=True):
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
            self, buffer, data, buffer_origin, data_origin, shape,
            buffer_strides, data_strides, blocking=True):
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

    def enqueue_zero_buffer(self, buffer):
        with self.context:
            if isinstance(buffer, pycuda.gpuarray.GPUArray):
                pycuda.driver.memset_d8(buffer.gpudata, 0, buffer.mem_size * buffer.dtype.itemsize)
            else:
                # managed memory
                pycuda.driver.memset_d8(buffer.base.get_device_pointer(), 0,
                                        buffer.size * buffer.dtype.itemsize)

    def enqueue_kernel(self, kernel, args, global_size, local_size):
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

    def enqueue_marker(self):
        with self.context:
            event = pycuda.driver.Event()
            event.record(self._pycuda_stream)
        return Event(event)

    def enqueue_wait_for_events(self, events):
        with self.context:
            for event in events:
                self._pycuda_stream.wait_for_event(event._pycuda_event)

    def flush(self):
        with self.context:
            # This anecdotally flushes, but isn't tested
            self._pycuda_stream.is_done()

    def finish(self):
        with self.context:
            self._pycuda_stream.synchronize()


class TuningCommandQueue(CommandQueue):
    def __init__(self, *args, **kwargs):
        super(TuningCommandQueue, self).__init__(*args, **kwargs)
        self.is_tuning = False
        self._start_event = None

    def start_tuning(self):
        self.is_tuning = True
        self._start_event = self.enqueue_marker()

    def stop_tuning(self):
        elapsed = 0.0
        if self._start_event is not None:
            end_event = self.enqueue_marker()
            elapsed = self._start_event.time_till(end_event)
        self._start_event = None
        self.is_tuning = False
        return elapsed
