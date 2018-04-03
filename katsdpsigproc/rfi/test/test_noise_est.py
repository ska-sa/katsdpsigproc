"""Tests for RFI noise estimation algorithms"""

from __future__ import division, print_function, absolute_import

import numpy as np

from .. import host, device
from ...test.test_accel import device_test, force_autotune


_deviations = None
_deviations_big = None
_expected = None


def setup():
    global _deviations, _deviations_big, _expected
    _deviations = np.array(
        [
            [0.0, 3.0, 2.4],
            [1.5, -1.4, 4.6],
            [0.0, 1.1, 3.3],
            [5.0, 0.0, -3.1]
        ]).astype(np.float32)
    _expected = np.array([3.25, 1.4, 3.2]) * 1.4826
    shape = (117, 273)

    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    _deviations_big = rs.standard_normal(shape).astype(np.float32)


def test_NoiseEstMADHost():
    noise_est = host.NoiseEstMADHost()
    actual = noise_est(_deviations)
    np.testing.assert_allclose(_expected, actual)


class BaseTestNoiseEstDeviceClass(object):
    @device_test
    def test_result(self, context, queue):
        template = self.factory(context)
        ne_host = template.host_class()
        ne_device = device.NoiseEstHostFromDevice(template, queue)
        noise_host = ne_host(_deviations_big)
        noise_device = ne_device(_deviations_big)
        np.testing.assert_allclose(noise_host, noise_device)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        self.factory(context)


class TestNoiseEstMADDevice(BaseTestNoiseEstDeviceClass):
    def factory(self, context):
        return device.NoiseEstMADDeviceTemplate(context)


class TestNoiseEstMADTDevice(BaseTestNoiseEstDeviceClass):
    def factory(self, context):
        return device.NoiseEstMADTDeviceTemplate(context, 10240)
