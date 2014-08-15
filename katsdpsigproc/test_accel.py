import numpy as np
import unittest
from .accel import Array, DeviceArray
import pycuda.autoinit
import pycuda.driver as cuda

class TestArray(unittest.TestCase):
    def setUp(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = Array(shape=self.shape, dtype=np.int32, padded_shape=self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(Array)
        self.sliced = self.constructed[2:4, 3:5]

    def test_safe(self):
        self.assertTrue(Array.safe(self.constructed))
        self.assertFalse(Array.safe(self.view))
        self.assertFalse(Array.safe(self.sliced))
        self.assertFalse(Array.safe(np.zeros(self.shape)))

class TestDeviceArray(unittest.TestCase):
    def setUp(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = DeviceArray(shape=self.shape, dtype=np.int32, padded_shape=self.padded_shape)

    def test_strides(self):
        self.assertEqual(self.strides, self.array.strides)

    def test_empty_like(self):
        ary = self.array.empty_like()
        self.assertEqual(self.shape, ary.shape)
        self.assertEqual(self.strides, ary.strides)
        self.assertTrue(Array.safe(ary))

    def test_set_get(self):
        ary = np.random.randint(0, 100, self.shape).astype(np.int32)
        self.array.set(ary)
        # Read back results, check that it matches
        buf = np.zeros(self.padded_shape, dtype=np.int32)
        cuda.memcpy_dtoh(buf, self.array.buffer)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        assert np.all(ary == buf)
        # Check that it matches get
        buf = self.array.get()
        assert np.all(ary == buf)
