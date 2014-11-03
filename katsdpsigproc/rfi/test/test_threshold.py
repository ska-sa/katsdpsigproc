"""Tests for RFI thresholding algorithms"""

import numpy as np
from .. import host
from ...test.test_accel import device_test, force_autotune
from .. import device

def setup():
    global _deviations, _spikes
    shape = (117, 273)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    # Pick 1/4 of samples to be RFI
    _spikes = rs.random_sample(shape) < 0.25
    _deviations = rs.standard_normal(shape).astype(np.float32) * 10.0
    _deviations[_spikes] += 200.0

def test_ThresholdSimpleHost():
    check_host_class(host.ThresholdSimpleHost, 11.0)

def test_ThresholdSumHost():
    check_host_class(host.ThresholdSumHost, 11.0)

def check_host_class(cls, n_sigma):
    threshold = cls(n_sigma)
    noise = np.repeat(10.0, _deviations.shape[1]).astype(np.float32)
    flags = threshold(_deviations, noise)
    np.testing.assert_equal(flags.astype(np.bool_), _spikes)

class BaseTestDeviceClass(object):
    @device_test
    def test_result(self, context, queue):
        n_sigma = 11.0
        template = self.factory(context, n_sigma)
        th_host = template.host_class(n_sigma)
        th_device = device.ThresholdHostFromDevice(template, queue)
        noise = np.linspace(0.0, 50.0, _deviations.shape[1]).astype(np.float32)
        flags_host = th_host(_deviations, noise)
        flags_device = th_device(_deviations, noise)
        np.testing.assert_equal(flags_host, flags_device)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        self.factory(context, 11.0)

class TestThresholdSimpleDevice(BaseTestDeviceClass):
    def factory(self, context, n_sigma):
        return device.ThresholdSimpleDeviceTemplate(context, n_sigma, False)

class TestThresholdSimpleDeviceTransposed(BaseTestDeviceClass):
    def factory(self, context, n_sigma):
        return device.ThresholdSimpleDeviceTemplate(context, n_sigma, True)

class TestThresholdSumDevice(BaseTestDeviceClass):
    def factory(self, context, n_sigma):
        return device.ThresholdSumDeviceTemplate(context, n_sigma)
