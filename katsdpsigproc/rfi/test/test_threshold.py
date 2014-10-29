"""Tests for RFI thresholding algorithms"""

import numpy as np
from .. import host
from ...test.test_accel import device_test, test_context, test_command_queue
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

@device_test
def test_ThresholdSimpleDevice():
    check_device_class(device.ThresholdSimpleDeviceTemplate, 11.0, False, 4, 3)

@device_test
def test_ThresholdSimpleDevice_transposed():
    check_device_class(device.ThresholdSimpleDeviceTemplate, 11.0, True, 4, 3)

@device_test
def test_ThresholdSumDevice():
    check_device_class(device.ThresholdSumDeviceTemplate, 11.0)

def check_device_class(cls, n_sigma, *device_args, **device_kw):
    th_host = cls.host_class(n_sigma)
    template = cls(test_context, n_sigma, *device_args, **device_kw)
    th_device = device.ThresholdHostFromDevice(template, test_command_queue)
    noise = np.linspace(0.0, 50.0, _deviations.shape[1]).astype(np.float32)
    flags_host = th_host(_deviations, noise)
    flags_device = th_device(_deviations, noise)
    np.testing.assert_equal(flags_host, flags_device)
