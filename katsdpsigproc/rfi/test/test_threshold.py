"""Tests for RFI thresholding algorithms"""

import numpy as np
from .. import host
from nose.tools import assert_equal
from ...test.test_accel import device_test, test_context, test_command_queue
from .. import device
from ... import accel

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

def test_ThresholdSumHost():
    check_host_class('ThresholdSumHost', 11.0)

def check_host_class(cls_name, n_sigma):
    cls = getattr(host, cls_name)
    threshold = cls(n_sigma)
    flags = threshold(_deviations)
    np.testing.assert_equal(flags.astype(np.bool_), _spikes)

@device_test
def test_ThresholdMADDevice():
    check_device_class('ThresholdMADDevice', 11.0, (4, 3))

@device_test
def test_ThresholdMADTDevice():
    check_device_class('ThresholdMADTDevice', 11.0, (512,))

def check_device_class(cls_name, n_sigma, device_args=(), device_kw={}):
    cls = getattr(device, cls_name)
    th_host = cls.host_class(n_sigma)
    th_device = device.ThresholdHostFromDevice(
            cls(test_command_queue, n_sigma, *device_args, **device_kw))
    flags_host = th_host(_deviations)
    flags_device = th_device(_deviations)
    np.testing.assert_equal(flags_host, flags_device)

class TestMedianNonZero(object):
    @device_test
    def setup(self):
        self.size = 128  # Number of workitems
        program = accel.build(test_context,
                'rfi/test/test_median_non_zero.mako', {'size': self.size})
        self.kernel = program.get_kernel('test_median_non_zero')

    def check_array(self, data):
        data = np.asarray(data, dtype=np.float32)
        N = data.shape[0]
        expected = np.median(data[data > 0.0])
        data_d = accel.DeviceArray(test_context, shape=data.shape, dtype=np.float32)
        data_d.set(test_command_queue, data)
        out_d = accel.DeviceArray(test_context, shape=(2,), dtype=np.float32)
        test_command_queue.enqueue_kernel(
                self.kernel,
                [data_d.buffer, out_d.buffer, np.int32(N)],
                global_size=(self.size,),
                local_size=(self.size,))
        out = out_d.empty_like()
        out_d.get(test_command_queue, out)

        assert_equal(expected, out[0])
        assert_equal(expected, out[1])

    @device_test
    def test_single(self):
        self.check_array([5.3])

    @device_test
    def test_even(self):
        self.check_array([1.2, 2.3, 3.4, 4.5])

    @device_test
    def test_zeros(self):
        self.check_array([0.0, 0.0, 0.0, 1.2, 5.3, 2.4])

    @device_test
    def test_big_even(self):
        data = np.random.random_sample(10001) + 0.5
        data[123] = 0.0
        self.check_array(data)

    @device_test
    def test_big_odd(self):
        data = np.random.random_sample(10000) + 0.5
        data[456] = 0.0
        self.check_array(data)
