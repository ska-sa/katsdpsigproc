"""Tests for RFI flagging wrappers. The backgrounding and thresholding have
separate tests - this module just tests that they can be glued together
properly."""

import numpy as np
from .. import host
from ...test.test_accel import device_test, test_context, test_command_queue
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
    noise_est = host.NoiseEstMADHost()
    threshold = host.ThresholdSimpleHost(11.0)
    flagger = host.FlaggerHost(background, noise_est, threshold)
    flags = flagger(_vis)
    np.testing.assert_equal(_spikes, flags)

def check_flagger_device(transpose_noise_est, transpose_threshold):
    background = device.BackgroundMedianFilterDeviceTemplate(test_context, 13)
    if transpose_noise_est:
        noise_est = device.NoiseEstMADTDeviceTemplate(test_context, 1024)
    else:
        noise_est = device.NoiseEstMADDeviceTemplate(test_context, 8, 8)
    threshold = device.ThresholdSimpleDeviceTemplate(test_context,
            11.0, transpose_threshold, 8, 8)
    flagger_device = device.FlaggerDeviceTemplate(background, noise_est, threshold)
    flagger = device.FlaggerHostFromDevice(flagger_device, test_command_queue)
    flags = flagger(_vis)
    np.testing.assert_equal(_spikes, flags)

@device_test
def test_flagger_device():
    check_flagger_device(False, False)

@device_test
def test_flagger_device_transpose_noise_est():
    """Test device flagger with a transposed noise estimator"""
    check_flagger_device(True, False)

@device_test
def test_flagger_device_transpose_threshold():
    """Test device flagger with a transposed thresholder"""
    check_flagger_device(False, True)

@device_test
def test_flagger_device_transpose_both():
    """Test device flagger with a transposed noise estimator and thresholder"""
    check_flagger_device(True, True)
