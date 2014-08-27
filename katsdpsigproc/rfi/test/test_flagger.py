"""Tests for RFI flagging wrappers. The backgrounding and thresholding have
separate tests - this module just tests that they can be glued together
properly."""

import numpy as np
from .. import host
from nose.tools import assert_equal
from ...test.test_accel import cuda_test, have_cuda
if have_cuda:
    import pycuda.autoinit
    from .. import device

def setup():
    global _vis, _spikes
    shape = (117, 131)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    _vis = rs.standard_normal(shape) + 1j * rs.standard_normal(shape)
    # Pick 1/16 of samples to be RFI
    # Give them random amplitude from a range, and random phase
    _spikes = rs.random_sample(shape) < 1.0 / 16.0
    _spikes = _spikes.astype(np.uint8)
    rfi_amp = rs.random_sample(shape) * 20.0 + 50.0
    rfi_phase = rs.random_sample(shape) * (2j * np.pi)
    rfi = rfi_amp * np.exp(rfi_phase)
    _vis += _spikes * rfi
    _vis = _vis.astype(np.complex64)

def test_flagger_host():
    background = host.BackgroundMedianFilterHost(13)
    threshold = host.ThresholdMADHost(11.0)
    flagger = host.FlaggerHost(background, threshold)
    flags = flagger(_vis)
    np.testing.assert_equal(_spikes, flags)

@cuda_test
def test_flagger_device():
    background = device.BackgroundMedianFilterDevice(pycuda.autoinit.context, 13)
    threshold = device.ThresholdMADDevice(pycuda.autoinit.context, 11.0, 8, 8)
    flagger_device = device.FlaggerDevice(background, threshold)
    flagger = device.FlaggerHostFromDevice(flagger_device)
    flags = flagger(_vis)
    np.testing.assert_equal(_spikes, flags)

@cuda_test
def test_flagger_device_transpose():
    """Test device flagger with a transposed thresholder"""
    background = device.BackgroundMedianFilterDevice(pycuda.autoinit.context, 13)
    threshold = device.ThresholdMADTDevice(pycuda.autoinit.context, 11.0, 1024)
    flagger_device = device.FlaggerDevice(background, threshold)
    flagger = device.FlaggerHostFromDevice(flagger_device)
    flags = flagger(_vis)
    np.testing.assert_equal(_spikes, flags)
