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

"""Tests for RFI noise estimation algorithms."""

from abc import ABC, abstractmethod

import numpy as np
import pytest

from katsdpsigproc.rfi import host, device
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue


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
    def test_result(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        global _deviations_big
        template = self.factory(context)
        ne_host = template.host_class()
        ne_device = device.NoiseEstHostFromDevice(template, command_queue)
        noise_host = ne_host(_deviations_big)
        noise_device = ne_device(_deviations_big)
        np.testing.assert_allclose(noise_host, noise_device)

    @pytest.mark.force_autotune
    def test_autotune(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
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
