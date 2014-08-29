import pyopencl
import pyopencl.array

class Program(object):
    def __init__(self, pyopencl_program):
        self._pyopencl_program = pyopencl_program

    def get_kernel(self, name):
        return Kernel(self, name)

class Kernel(object):
    def __init__(self, program, name):
        self._pyopencl_kernel = pyopencl.Kernel(program._pyopencl_program, name)

class Event(object):
    def __init__(self, pyopencl_event):
        self._pyopencl_event = pyopencl_event

    def wait(self):
        self._pyopencl_event.wait()

    def time_since(self, prior_event):
        prior_event.wait()
        self.wait()
        return 1e-9 * (self._pyopencl_event.profile.start - prior_event._pyopencl_event.profile.end)

    def time_till(self, next_event):
        return next_event.time_since(self)

class Context(object):
    def __init__(self, pyopencl_context):
        self._pyopencl_context = pyopencl_context

    def device_name(self):
        device = self._pyopencl_context.devices[0]
        return '{0} ({1})'.format(device.name, device.platform.name)

    def compile(self, source, extra_flags=None):
        # source is passed through str because it might arrive as Unicode,
        # triggering a warning
        program = pyopencl.Program(self._pyopencl_context, str(source))
        program.build(extra_flags)
        return Program(program)

    def allocate(self, shape, dtype):
        return pyopencl.array.Array(self._pyopencl_context, shape, dtype)

    def create_command_queue(self):
        return CommandQueue(self)

class CommandQueue(object):
    def __init__(self, context, pyopencl_command_queue=None):
        self.context = context
        if pyopencl_command_queue is None:
            pyopencl_device = context._pyopencl_context.devices[0]
            pyopencl_command_queue = pyopencl.CommandQueue(
                    context._pyopencl_context, pyopencl_device,
                    properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        self._pyopencl_command_queue = pyopencl_command_queue

    def enqueue_read_buffer(self, buffer, data, blocking=True):
        buffer.get(ary=data, queue=self._pyopencl_command_queue, async=not blocking)

    def enqueue_write_buffer(self, buffer, data, blocking=True):
        buffer.set(ary=data, queue=self._pyopencl_command_queue, async=not blocking)

    @classmethod
    def _raw_arg(cls, arg):
        if isinstance(arg, pyopencl.array.Array):
            return arg.data
        else:
            return arg

    def enqueue_kernel(self, kernel, args, global_size, local_size):
        # PyOpenCL doesn't allow Array objects to be passed through
        args = [self._raw_arg(x) for x in args]
        kernel._pyopencl_kernel(self._pyopencl_command_queue,
                global_size, local_size, *args)

    def enqueue_marker(self):
        return Event(pyopencl.enqueue_marker(self._pyopencl_command_queue))

    def finish(self):
        self._pyopencl_command_queue.finish()
