import numpy as np
import background
import pycuda.autoinit
from nose.tools import *

def setup():
    global _vis, _vis_big
    shape = (17, 13)
    _vis = np.array([[1.25, 1.5j, 1.0, 2.0, -1.75, 2.0]]).T.astype(np.complex64)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    _vis_big = (rs.randn(*shape) + rs.randn(*shape) * 1j).astype(np.complex64)

class TestBackgroundMedianFilterHost(object):
    def setup(self):
        self.background = background.BackgroundMedianFilterHost(3)

    def test(self):
        out = self.background(_vis)
        ref = np.array([[0.0, 0.25, -0.5, 0.25, -0.25, 0.25]]).T.astype(np.float32)
        np.testing.assert_equal(ref, out)

def test_device_classes():
    yield check_device_class, background.BackgroundMedianFilterDevice, 5, (128, 4)

def check_device_class(cls, width, device_args=(), device_kw={}):
    bg_host = cls.host_class(width)
    bg_device = background.BackgroundHostFromDevice(
            cls(pycuda.autoinit.context, width, *device_args, **device_kw))
    out_host = bg_host(_vis_big)
    out_device = bg_device(_vis_big)
    # Uses an abs tolerance because backgrounding subtracts nearby values
    np.testing.assert_allclose(out_host, out_device, atol=1e-6)
