################################################################################
# Copyright (c) 2014-2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Tests for RFI flagging wrappers.

The backgrounding and thresholding have separate tests - this module just tests
that they can be glued together properly.
"""

import numpy as np

from katsdpsigproc.rfi import host, device
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue
from .. import complex_normal


_vis: np.ndarray
_spikes: np.ndarray
_input_flags: np.ndarray


def setup():   # type: () -> None
    global _vis, _spikes, _input_flags
    shape = (117, 131)
    # Use a fixed seed to make the test repeatable
    rs = np.random.RandomState(seed=1)
    _vis = complex_normal(rs, size=shape)
    # Pick 1/16 of samples to be RFI
    # Give them random amplitude from a range, and random phase
    _spikes = rs.random_sample(shape) < 1.0 / 16.0
    _spikes = _spikes.astype(np.uint8)
    rfi_amp = rs.random_sample(shape) * 20.0 + 50.0
    rfi_phase = rs.random_sample(shape) * (2j * np.pi)
    rfi = rfi_amp * np.exp(rfi_phase)
    _vis += _spikes * rfi
    _vis = _vis.astype(np.complex64)
    _input_flags = (rs.random_sample(shape) < 1.0 / 16.0).astype(np.uint8) * 2


def test_flagger_host() -> None:
    global _vis, _spikes, _input_flags
    background = host.BackgroundMedianFilterHost(13)
    noise_est = host.NoiseEstMADHost()
    threshold = host.ThresholdSimpleHost(11.0)
    flagger = host.FlaggerHost(background, noise_est, threshold)
    flags = flagger(_vis)
    np.testing.assert_equal(_spikes, flags)
    # Test again, this time with channel flags
    flags = flagger(_vis, _input_flags[:, 0])
    input_flags = np.broadcast_to(_input_flags[:, 0:1], _vis.shape)
    expected = np.where(input_flags, 0, _spikes)
    np.testing.assert_equal(expected, flags)
    # And again with full input flags
    flags = flagger(_vis, _input_flags)
    expected = np.where(_input_flags, 0, _spikes)
    np.testing.assert_equal(expected, flags)


def check_flagger_device(use_flags: device.BackgroundFlags,
                         transpose_noise_est: bool, transpose_threshold: bool,
                         context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    global _vis, _spikes, _input_flags
    background = device.BackgroundMedianFilterDeviceTemplate(context, 13, use_flags=use_flags)
    if transpose_noise_est:
        noise_est = device.NoiseEstMADTDeviceTemplate(
            context, 1024)       # type: device.AbstractNoiseEstDeviceTemplate
    else:
        noise_est = device.NoiseEstMADDeviceTemplate(context, tuning={'wgsx': 8, 'wgsy': 8})
    threshold = device.ThresholdSimpleDeviceTemplate(
        context, transpose_threshold, tuning={'wgsx': 8, 'wgsy': 8})
    flagger_device = device.FlaggerDeviceTemplate(background, noise_est, threshold)
    flagger = device.FlaggerHostFromDevice(
        flagger_device, command_queue, threshold_args=dict(n_sigma=11.0))
    if use_flags == device.BackgroundFlags.CHANNEL:
        flags = flagger(_vis, _input_flags[:, 0])
        input_flags = np.broadcast_to(_input_flags[:, 0:1], _vis.shape)
        expected = np.where(input_flags, 0, _spikes)
        np.testing.assert_equal(expected, flags)
    elif use_flags == device.BackgroundFlags.FULL:
        flags = flagger(_vis, _input_flags)
        expected = np.where(_input_flags, 0, _spikes)
        np.testing.assert_equal(expected, flags)
    else:
        flags = flagger(_vis)
        np.testing.assert_equal(_spikes, flags)


def test_flagger_device(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    check_flagger_device(device.BackgroundFlags.NONE, False, False, context, command_queue)


def test_flagger_device_transpose_noise_est(context: AbstractContext,
                                            command_queue: AbstractCommandQueue) -> None:
    """Test device flagger with a transposed noise estimator."""
    check_flagger_device(device.BackgroundFlags.CHANNEL, True, False, context, command_queue)


def test_flagger_device_transpose_threshold(context: AbstractContext,
                                            command_queue: AbstractCommandQueue) -> None:
    """Test device flagger with a transposed thresholder."""
    check_flagger_device(device.BackgroundFlags.FULL, False, True, context, command_queue)


def test_flagger_device_transpose_both(context: AbstractContext,
                                       command_queue: AbstractCommandQueue) -> None:
    """Test device flagger with a transposed noise estimator and thresholder."""
    check_flagger_device(device.BackgroundFlags.NONE, True, True, context, command_queue)
