import numpy as np
from .accel import Array, DeviceArray
import pycuda.autoinit
import pycuda.driver as cuda
from nose.tools import *

class TestArray(object):
    def setup(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = Array(shape=self.shape, dtype=np.int32, padded_shape=self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(Array)
        self.sliced = self.constructed[2:4, 3:5]

    def test_safe(self):
        assert Array.safe(self.constructed)
        assert not Array.safe(self.view)
        assert not Array.safe(self.sliced)
        assert not Array.safe(np.zeros(self.shape))

class TestDeviceArray(object):
    def setup(self):
        self.ctx = pycuda.autoinit.context
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = DeviceArray(
                ctx=self.ctx,
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)

    def test_strides(self):
        assert_equal(self.strides, self.array.strides)

    def test_empty_like(self):
        ary = self.array.empty_like()
        assert_equal(self.shape, ary.shape)
        assert_equal(self.strides, ary.strides)
        assert Array.safe(ary)

    def test_set_get(self):
        ary = np.random.randint(0, 100, self.shape).astype(np.int32)
        self.array.set(ary)
        # Read back results, check that it matches
        buf = np.zeros(self.padded_shape, dtype=np.int32)
        cuda.memcpy_dtoh(buf, self.array.buffer.gpudata)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = self.array.get()
        np.testing.assert_equal(ary, buf)
