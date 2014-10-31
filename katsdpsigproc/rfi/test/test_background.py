import numpy as np
from .. import host
from ...test.test_accel import device_test, test_context, test_command_queue, force_autotune
from .. import device

def setup():
    global _vis, _vis_big
    global _deviations, _spikes
    shape = (17, 13)
    _vis = np.array([[1.25, 1.5j, 1.0, 2.0, -1.75, 2.0]]).T.astype(np.complex64)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    _vis_big = (rs.standard_normal(shape) + rs.standard_normal(shape) * 1j).astype(np.complex64)

class TestBackgroundMedianFilterHost(object):
    def setup(self):
        self.background = host.BackgroundMedianFilterHost(3)

    def test(self):
        out = self.background(_vis)
        ref = np.array([[0.0, 0.25, -0.5, 0.25, -0.25, 0.25]]).T.astype(np.float32)
        np.testing.assert_equal(ref, out)

class BaseTestBackgroundDeviceClass(object):
    @device_test
    def test_result(self):
        width = 5
        bg_device_template = self.factory(width)
        bg_host = bg_device_template.host_class(width, self.amplitudes)
        bg_device = device.BackgroundHostFromDevice(bg_device_template, test_command_queue)
        if self.amplitudes:
            vis = np.abs(_vis_big)
        else:
            vis = _vis_big
        out_host = bg_host(vis)
        out_device = bg_device(vis)
        # Uses an abs tolerance because backgrounding subtracts nearby values
        np.testing.assert_allclose(out_host, out_device, atol=1e-6)

    @device_test
    @force_autotune
    def test_autotune(self):
        self.factory(5)

class TestBackgroundMedianFilterDevice(BaseTestBackgroundDeviceClass):
    amplitudes = False

    def factory(self, width):
        return device.BackgroundMedianFilterDeviceTemplate(
                test_context, width, self.amplitudes)

class TestBackgroundMedianFilterDeviceAmplitudes(TestBackgroundMedianFilterDevice):
    amplitudes = True
