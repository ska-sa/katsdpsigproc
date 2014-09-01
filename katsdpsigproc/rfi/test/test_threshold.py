"""Tests for RFI thresholding algorithms"""

import numpy as np
from .. import host
from nose.tools import assert_equal
from ...test.test_accel import device_test, test_command_queue
from .. import device

def setup():
    global _deviations, _spikes
    shape = (117, 273)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    # Pick 1/4 of samples to be RFI
    _spikes = rs.random_sample(shape) < 0.25
    _deviations = rs.standard_normal(shape).astype(np.float32)
    _deviations[_spikes] += 50.0

def test_ThresholdMADHost():
    check_host_class('ThresholdMADHost', 11.0)

def check_host_class(cls_name, n_sigma):
    cls = getattr(host, cls_name)
    threshold = cls(n_sigma)
    flags = threshold(_deviations)
    np.testing.assert_equal(flags.astype(np.bool_), _spikes)

def test_ThresholdMADDevice():
    check_device_class('ThresholdMADDevice', 11.0, (4, 3))

def test_ThresholdMADTDevice():
    check_device_class('ThresholdMADTDevice', 11.0, (512,))

@device_test
def check_device_class(cls_name, n_sigma, device_args=(), device_kw={}):
    cls = getattr(device, cls_name)
    th_host = cls.host_class(n_sigma)
    th_device = device.ThresholdHostFromDevice(
            cls(test_command_queue, n_sigma, *device_args, **device_kw))
    flags_host = th_host(_deviations)
    flags_device = th_device(_deviations)
    np.testing.assert_equal(flags_host, flags_device)
