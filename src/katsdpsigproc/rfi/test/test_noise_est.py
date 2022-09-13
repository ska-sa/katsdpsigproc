"""Tests for RFI noise estimation algorithms."""

from abc import ABC, abstractmethod

import numpy as np

from .. import host, device
from ...abc import AbstractContext, AbstractCommandQueue
from ...test.test_accel import device_test, force_autotune


_deviations: np.ndarray
_deviations_big: np.ndarray
_expected: np.ndarray


def setup():   # type: () -> None
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


def test_NoiseEstMADHost() -> None:
    global _deviations, _expected
    noise_est = host.NoiseEstMADHost()
    actual = noise_est(_deviations)
    np.testing.assert_allclose(_expected, actual)


class BaseTestNoiseEstDeviceClass(ABC):
    @device_test
    def test_result(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        global _deviations_big
        template = self.factory(context)
        ne_host = template.host_class()
        ne_device = device.NoiseEstHostFromDevice(template, queue)
        noise_host = ne_host(_deviations_big)
        noise_device = ne_device(_deviations_big)
        np.testing.assert_allclose(noise_host, noise_device)

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.factory(context)

    @abstractmethod
    def factory(self, context: AbstractContext) -> device.AbstractNoiseEstDeviceTemplate:
        pass       # pragma: nocover


class TestNoiseEstMADDevice(BaseTestNoiseEstDeviceClass):
    def factory(self, context: AbstractContext) -> device.NoiseEstMADDeviceTemplate:
        return device.NoiseEstMADDeviceTemplate(context)


class TestNoiseEstMADTDevice(BaseTestNoiseEstDeviceClass):
    def factory(self, context: AbstractContext) -> device.NoiseEstMADTDeviceTemplate:
        return device.NoiseEstMADTDeviceTemplate(context, 10240)
