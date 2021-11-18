"""Test RFI background estimation."""

from abc import ABC, abstractmethod

import numpy as np

from .. import host
from ...abc import AbstractContext, AbstractCommandQueue
from ...test.test_accel import device_test, force_autotune
from ...test import complex_normal
from .. import device


_vis = np.array([])
_vis_big = np.array([])
_flags = np.array([])
_flags_big = np.array([])


def setup():   # type: () -> None
    global _vis, _vis_big, _flags, _flags_big
    shape = (417, 313)
    _vis = np.array([[1.25, 1.5j, 1.0, 2.0, -1.75, 2.0]]).T.astype(np.complex64)
    _flags = np.array([0, 0, 1, 0, 0, 4]).T.astype(np.uint8)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    _vis_big = complex_normal(rs, size=shape).astype(np.complex64)
    _flags_big = (rs.random_sample(shape) < 0.1).astype(np.uint8)
    # Ensure that in some cases the entire window is flagged. Also test with non-0/1
    # flag values.
    _flags_big[100:110, 0:100] = 4


class TestBackgroundMedianFilterHost:
    def setup(self) -> None:
        self.background = host.BackgroundMedianFilterHost(3)

    def test(self) -> None:
        out = self.background(_vis)
        ref = np.array([[-0.125, 0.25, -0.5, 0.25, -0.25, 0.125]]).T.astype(np.float32)
        np.testing.assert_equal(ref, out)

    def test_flags(self) -> None:
        out = self.background(_vis, _flags)
        ref = np.array([[-0.125, 0.125, 0.0, 0.125, -0.125, 0.0]]).T.astype(np.float32)
        np.testing.assert_equal(ref, out)


class BaseTestBackgroundDeviceClass(ABC):
    amplitudes = None     # type: bool
    use_flags = None      # type: device.BackgroundFlags

    @device_test
    def test_result(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        width = 5
        bg_device_template = self.factory(context, width)
        bg_host = bg_device_template.host_class(width, self.amplitudes)
        bg_device = device.BackgroundHostFromDevice(bg_device_template, queue)
        if self.amplitudes:
            vis = np.abs(_vis_big)
        else:
            vis = _vis_big
        if self.use_flags:
            full = self.use_flags == device.BackgroundFlags.FULL
            flags = _flags_big if full else _flags_big[:, 0]
            out_host = bg_host(vis, flags)
            out_device = bg_device(vis, flags)
        else:
            out_host = bg_host(vis)
            out_device = bg_device(vis)
        # Uses an abs tolerance because backgrounding subtracts nearby values
        np.testing.assert_allclose(out_host, out_device, atol=1e-6)

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.factory(context, 5)

    @abstractmethod
    def factory(self, context: AbstractContext,
                width: int) -> device.AbstractBackgroundDeviceTemplate:
        pass       # pragma: nocover


class TestBackgroundMedianFilterDevice(BaseTestBackgroundDeviceClass):
    amplitudes = False
    use_flags = device.BackgroundFlags.NONE

    def factory(self, context: AbstractContext, width: int) -> \
            device.BackgroundMedianFilterDeviceTemplate:
        return device.BackgroundMedianFilterDeviceTemplate(
            context, width, self.amplitudes, self.use_flags)


class TestBackgroundMedianFilterDeviceAmplitudes(TestBackgroundMedianFilterDevice):
    amplitudes = True


class TestBackgroundMedianFilterDeviceFlags(TestBackgroundMedianFilterDevice):
    use_flags = device.BackgroundFlags.CHANNEL
