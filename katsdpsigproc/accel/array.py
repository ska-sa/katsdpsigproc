import numpy as np

class DeviceArray(object):
    """A light-weight array-like wrapper around a device buffer, that
    handles padding better than `pycuda.gpuarray.GPUArray` (which
    has very poor support).

    It only supports C-order arrays where the inner-most dimension is
    contiguous.
    """

    def __init__(self, thread, shape, dtype, padded_shape=None):
        self.thread = thread
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.padded_shape = padded_shape
        size = reduce(lambda x, y: x * y, padded_shape) * self.dtype.itemsize
        self.buffer = thread.allocate(size)
        pass # TODO

    def set(self, ary, thread=None):
        if thread is None:
            thread = self.thread
        assert ary.shape == self.shape
        pass # TODO

    def get(self):
        pass # TODO
