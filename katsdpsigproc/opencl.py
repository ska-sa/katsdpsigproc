"""Abstraction layer over PyOpenCL to present an interface that is common
between CUDA and OpenCL. This is only the subset that is needed so far. New
functionality can be added, but should be kept in sync with
:mod:`katsdpsigproc.cuda`.
"""

import pyopencl
import pyopencl.array

class Program(object):
    """Abstraction of a program object"""
    def __init__(self, pyopencl_program):
        self._pyopencl_program = pyopencl_program

    def get_kernel(self, name):
        """Create a new kernel

        Parameters
        ----------
        name : str
            Name of the kernel function
        """
        return Kernel(self, name)

class Kernel(object):
    """Abstraction of a kernel object. The object can be enqueued using
    :meth:`CommandQueue.enqueue_kernel`.

    The recommended way to create this object is via
    :meth:`Program.get_kernel`.
    """
    def __init__(self, program, name):
        self._pyopencl_kernel = pyopencl.Kernel(program._pyopencl_program, name)

class Event(object):
    """Abstraction of an event. This is more akin to a CUDA event than an
    OpenCL event, in that it is a marker in a command queue rather than
    associated with a specific command.
    """
    def __init__(self, pyopencl_event):
        self._pyopencl_event = pyopencl_event

    def wait(self):
        """Block until the event has completed"""
        self._pyopencl_event.wait()

    def time_since(self, prior_event):
        """Return the time in seconds from `prior_event` to self. Unlike the
        PyCUDA method of the same name, this will wait for the events to
        complete if they have not already.
        """
        prior_event.wait()
        self.wait()
        return 1e-9 * (self._pyopencl_event.profile.start -
                       prior_event._pyopencl_event.profile.end)

    def time_till(self, next_event):
        """Return the time in seconds from this event to `next_event`. See
        :meth:`time_since`.
        """
        return next_event.time_since(self)

class Device(object):
    """Abstraction of an OpenCL device"""
    def __init__(self, pyopencl_device):
        self._pyopencl_device = pyopencl_device

    def make_context(self):
        """Create a new context associated with this device"""
        return Context(pyopencl.Context([self._pyopencl_device]))

    @property
    def name(self):
        """Return human-readable name for the device"""
        return self._pyopencl_device.name

    @property
    def platform_name(self):
        """Return human-readable name for the platform owning the device"""
        return self._pyopencl_device.platform.name

    @property
    def is_cuda(self):
        """Whether the device is a CUDA device"""
        return False

    @property
    def is_gpu(self):
        """Whether device is a GPU"""
        return self._pyopencl_device.type & pyopencl.device_type.GPU

    @property
    def is_accelerator(self):
        """Whether device is an accelerator (as defined by OpenCL device
        types)"""
        return self._pyopencl_device.type & pyopencl.device_type.ACCELERATOR

    @property
    def is_cpu(self):
        """Whether the device is a CPU"""
        return self._pyopencl_device.type & pyopencl.device_type.CPU

    @classmethod
    def get_devices(cls):
        """Return a list of all devices on all platforms"""
        ans = []
        for platform in pyopencl.get_platforms():
            for device in platform.get_devices():
                ans.append(Device(device))
        return ans

class Context(object):
    """Abstraction of an OpenCL context"""
    def __init__(self, pyopencl_context):
        self._pyopencl_context = pyopencl_context
        device = pyopencl_context.devices[0]
        self._internal_queue = pyopencl.CommandQueue(pyopencl_context, device)

    @property
    def device(self):
        """Return the device associated with the context (or the first device,
        if there are multiple)."""
        return Device(self._pyopencl_context.devices[0])

    def compile(self, source, extra_flags=None):
        """Build a program object from source

        Parameters
        ----------
        source : str
            OpenCL source code
        extra_flags : list
            OpenCL-specific parameters to pass to the compiler
        """
        # source is passed through str because it might arrive as Unicode,
        # triggering a warning
        program = pyopencl.Program(self._pyopencl_context, str(source))
        program.build(extra_flags)
        return Program(program)

    def allocate(self, shape, dtype):
        """Create a typed buffer on the device.

        Parameters
        ----------
        shape : tuple
            Shape for the array
        dtype : numpy dtype
            Type for the data
        """
        return pyopencl.array.Array(self._pyopencl_context, shape, dtype)

    def allocate_pinned(self, shape, dtype):
        """Create a buffer in host memory that can be efficiently copied
        to and from the device.

        Parameters
        ----------
        shape : tuple
            Shape for the array
        dtype : numpy dtype
            Type for the data
        """
        bytes = reduce(lambda x, y: x * y, shape) * dtype.itemsize
        buf = pyopencl.Buffer(
                self._pyopencl_context,
                pyopencl.mem_flags.ALLOC_HOST_PTR | pyopencl.mem_flags.READ_ONLY,
                bytes)
        (ary, event) = pyopencl.enqueue_map_buffer(
                self._internal_queue,
                buf,
                pyopencl.map_flags.READ | pyopencl.map_flags.WRITE,
                0, shape, dtype)
        return ary

    def create_command_queue(self, profile=False):
        """Create a new command queue associated with this context

        Parameters
        ----------
        profile : boolean
            If true, the command queue will support timing kernels
        """
        return CommandQueue(self, profile=profile)

    def create_tuning_command_queue(self):
        """Create a new command queue for doing autotuning"""
        return TuningCommandQueue(self)


class CommandQueue(object):
    """Abstraction of a command queue. If no existing command queue is passed
    to the constructor, a new one is created.

    Parameters
    ----------
    context : :class:`Context`
        Context owning the queue
    pyopencl_command_queue : :class:`pyopencl.CommandQueue` (optional)
        Existing command queue to wrap
    profile : boolean
        If true and `pyopencl_command_queue` is omitted, enabling profiling
        (timing) on the queue.
    """
    def __init__(self, context, pyopencl_command_queue=None, profile=False):
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

    def enqueue_read_buffer(self, buffer, data, blocking=True):
        """Copy data from the device to the host. Only whole-buffer copies are
        supported, and the shape and type must match. In general, one should use
        the convenience functions in :class:`accel.DeviceArray`.

        Parameters
        ----------
        buffer : `pyopencl.array.Array`
            Source
        data : array-like
            Target
        blocking : boolean (optional)
            If true (default) the call blocks until the copy is complete
        """
        buffer.get(ary=data, queue=self._pyopencl_command_queue, async=not blocking)

    def enqueue_write_buffer(self, buffer, data, blocking=True):
        """Copy data from the host to the device. Only whole-buffer copies are
        supported, and the shape and type must match. In general, one should
        use the convenience functions in :class:`accel.DeviceArray`.

        Parameters
        ----------
        buffer : `pyopencl.array.Array`
            Target
        data : array-like
            Source
        blocking : boolean (optional)
            If true (default), the call blocks until the source has been fully
            read (it has no necessarily reached the device).
        """
        buffer.set(ary=data, queue=self._pyopencl_command_queue, async=not blocking)

    @classmethod
    def _raw_arg(cls, arg):
        if isinstance(arg, pyopencl.array.Array):
            return arg.data
        else:
            return arg

    def _enqueue_kernel(self, kernel, args, global_size, local_size):
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

    def enqueue_kernel(self, kernel, args, global_size, local_size):
        """Enqueue a kernel to the command queue.

        .. warning:: It is not thread-safe to call this function in two threads
            on the same kernel at the same time.

        Parameters
        ----------
        kernel : :class:`Kernel`
            Kernel to run
        args : sequence
            Arguments to pass to the kernel. Refer to the PyOpenCL
            documentation for details. Additionally, this function allows
            a :class:`pyopencl.array.Array` to be passed.
        global_size : tuple
            Number of work-items in each global dimension
        local_size : tuple
            Number of work-items in each local dimension. Must divide
            exactly into `global_size`.
        """
        self._enqueue_kernel(kernel, args, global_size, local_size)

    def enqueue_marker(self):
        """Create an event at this point in the command queue"""
        return Event(pyopencl.enqueue_marker(self._pyopencl_command_queue))

    def finish(self):
        """Block until all enqueued work has completed"""
        self._pyopencl_command_queue.finish()

class TuningCommandQueue(CommandQueue):
    """Command queue with extra facilities for autotuning. It keeps
    track of kernels that are enqueued since the last call to
    :meth:`start_tuning`, and reports the total time they consume
    when :meth:`stop_tuning` is called.
    """

    def __init__(self, *args, **kwargs):
        kwargs['profile'] = True
        super(TuningCommandQueue, self).__init__(*args, **kwargs)
        self.is_tuning = False
        self.events = []

    def start_tuning(self):
        self.is_tuning = True
        self.events = []

    def enqueue_kernel(self, kernel, args, global_size, local_size):
        if not self.is_tuning:
            super(TuningCommandQueue, self).enqueue_kernel(kernel, args, global_size, local_size)
        else:
            event = self._enqueue_kernel(kernel, args, global_size, local_size)
            self.events.append(event)

    def stop_tuning(self):
        self.finish()
        elapsed = 0.0
        if self.events:
            start = self.events[0].profile.start
            end = self.events[-1].profile.end
            elapsed = (end - start) * 1e-9
        self.is_tuning = False
        self.events = []
        return elapsed
