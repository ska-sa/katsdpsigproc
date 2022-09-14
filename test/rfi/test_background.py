"""Test RFI background estimation."""

import numpy as np
import pytest

from katsdpsigproc.rfi import host, device
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue
from .. import complex_normal


_vis = np.array([])
_vis_big = np.array([])
_flags = np.array([])
_flags_big = np.array([])


def setup() -> None:
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


class TestBackgroundDevice:
    @pytest.fixture(params=[True, False])
    def amplitudes(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[
        device.BackgroundFlags.NONE, device.BackgroundFlags.CHANNEL, device.BackgroundFlags.FULL
    ])
    def use_flags(self, request) -> device.BackgroundFlags:
        return request.param

    def test_result(self, context: AbstractContext, command_queue: AbstractCommandQueue,
                    amplitudes: bool, use_flags: device.BackgroundFlags) -> None:
        width = 5
        bg_device_template = device.BackgroundMedianFilterDeviceTemplate(
            context, width, amplitudes, use_flags)
        bg_host = bg_device_template.host_class(width, amplitudes)
        bg_device = device.BackgroundHostFromDevice(bg_device_template, command_queue)
        if amplitudes:
            vis = np.abs(_vis_big)
        else:
            vis = _vis_big
        if use_flags:
            full = use_flags == device.BackgroundFlags.FULL
            flags = _flags_big if full else _flags_big[:, 0]
            out_host = bg_host(vis, flags)
            out_device = bg_device(vis, flags)
        else:
            out_host = bg_host(vis)
            out_device = bg_device(vis)
        # Uses an abs tolerance because backgrounding subtracts nearby values
        np.testing.assert_allclose(out_host, out_device, atol=1e-6)

    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext,
                      amplitudes: bool, use_flags: device.BackgroundFlags) -> None:
        device.BackgroundMedianFilterDeviceTemplate(context, 5, amplitudes, use_flags)
