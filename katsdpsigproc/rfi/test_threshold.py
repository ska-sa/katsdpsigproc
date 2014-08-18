import numpy as np
import threshold
import pycuda.autoinit
from nose.tools import *

def setup():
    global _deviations, _spikes
    shape = (17, 13)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    # Pick 1/4 of samples to be RFI
    _spikes = rs.random_sample(shape) < 0.25
    _deviations = np.random.randn(*shape).astype(np.float32)
    _deviations[_spikes] += 50.0

def test_host_classes():
    yield check_host_class, threshold.ThresholdMADHost, 11.0

def check_host_class(cls, n_sigma):
    threshold = cls(n_sigma)
    flags = threshold(_deviations)
    np.testing.assert_equal(flags.astype(np.bool_), _spikes)

def test_device_classes():
    yield check_device_class, threshold.ThresholdMADDevice, 11.0, (4, 3)

def check_device_class(cls, n_sigma, device_args=(), device_kw={}):
    th_host = cls.host_class(n_sigma)
    th_device = threshold.ThresholdHostFromDevice(
            cls(pycuda.autoinit.context, n_sigma, *device_args, **device_kw))
    flags_host = th_host(_deviations)
    flags_device = th_device(_deviations)
    np.testing.assert_equal(flags_host, flags_device)
