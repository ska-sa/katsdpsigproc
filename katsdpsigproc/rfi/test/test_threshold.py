"""Tests for RFI thresholding algorithms."""

from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from .. import host
from ...abc import AbstractContext, AbstractCommandQueue
from ...test.test_accel import device_test, force_autotune
from .. import device


_deviations: np.ndarray
_spikes: np.ndarray


def setup():   # type: () -> None
    global _deviations, _spikes
    shape = (117, 273)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    # Pick 1/4 of samples to be RFI
    _spikes = rs.random_sample(shape) < 0.25
    _deviations = rs.standard_normal(shape).astype(np.float32) * 10.0
    _deviations[_spikes] += 200.0


def test_ThresholdSimpleHost() -> None:
    check_host_class(host.ThresholdSimpleHost, 11.0)


def test_ThresholdSumHost() -> None:
    check_host_class(host.ThresholdSumHost, 11.0)


def check_host_class(cls: Type[host.AbstractThresholdHost], n_sigma: float) -> None:
    global _deviations, _spikes
    threshold = cls(n_sigma)
    noise = np.repeat(10.0, _deviations.shape[1]).astype(np.float32)
    flags = threshold(_deviations, noise)
    np.testing.assert_equal(flags.astype(np.bool_), _spikes)


class BaseTestDeviceClass(ABC):
    @device_test
    def test_result(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        global _deviations
        n_sigma = 11.0
        template = self.factory(context)
        th_host = template.host_class(n_sigma)
        th_device = device.ThresholdHostFromDevice(template, queue, n_sigma=n_sigma)
        noise = np.linspace(0.0, 50.0, _deviations.shape[1]).astype(np.float32)
        flags_host = th_host(_deviations, noise)
        flags_device = th_device(_deviations, noise)
        np.testing.assert_equal(flags_host, flags_device)

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.factory(context)

    @abstractmethod
    def factory(self, context: AbstractContext) -> device.AbstractThresholdDeviceTemplate:
        pass        # pragma: nocover


class TestThresholdSimpleDevice(BaseTestDeviceClass):
    def factory(self, context: AbstractContext) -> device.ThresholdSimpleDeviceTemplate:
        return device.ThresholdSimpleDeviceTemplate(context, False)


class TestThresholdSimpleDeviceTransposed(BaseTestDeviceClass):
    def factory(self, context: AbstractContext) -> device.ThresholdSimpleDeviceTemplate:
        return device.ThresholdSimpleDeviceTemplate(context, True)


class TestThresholdSumDevice(BaseTestDeviceClass):
    def factory(self, context: AbstractContext) -> device.ThresholdSumDeviceTemplate:
        return device.ThresholdSumDeviceTemplate(context)
