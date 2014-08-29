import pycuda.driver
import pycuda.compiler
import pycuda.gpuarray

_nvcc_flags = pycuda.compiler.DEFAULT_NVCC_FLAGS + ['-lineinfo']

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

class Context(object):
    def __init__(self, pycuda_context):
        self._pycuda_context = pycuda_context

    def device_name(self):
        with self:
            return self._pycuda_context.get_device().name() + ' (CUDA)'

    def compile(self, source, extra_flags=None):
        with self:
            module = pycuda.compiler.SourceModule(source, options=_nvcc_flags + extra_flags)
            return Program(module)

    def allocate(self, shape, dtype):
        with self:
            return pycuda.gpuarray.GPUArray(shape, dtype)

    def create_command_queue(self):
        return CommandQueue(self)

    def __enter__(self):
        self._pycuda_context.push()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pycuda_context.pop()
        return False

class CommandQueue(object):
    def __init__(self, context, pycuda_stream=None):
        self.context = context
        if pycuda_stream is None:
            with context:
                pycuda_stream = pycuda.driver.Stream()
        self._pycuda_stream = pycuda_stream

    def enqueue_read_buffer(self, buffer, data, blocking=True):
        with self.context:
            if blocking:
                # TODO: PyCUDA doesn't take a stream argument here!
                buffer.get(data)
            else:
                buffer.get_async(data, stream=self._pycuda_stream)

    def enqueue_write_buffer(self, buffer, data, blocking=True):
        with self.context:
            if blocking:
                # TODO: PyCUDA doesn't take a stream argument here!
                buffer.set(data)
            else:
                buffer.set_async(data, stream=self._pycuda_stream)

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
            e = pycuda.driver.Event()
            e.record(self._pycuda_stream)
        return Event(e)
