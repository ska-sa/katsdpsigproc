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

"""Tests for implementations of the :class:`.AbstractDevice` interface."""

from katsdpsigproc.abc import AbstractContext


def test_properties(context: AbstractContext) -> None:
    """Test that attributes can be retrieved and have the appropriate type."""
    device = context.device
    assert isinstance(device.name, str)
    assert isinstance(device.platform_name, str)
    assert isinstance(device.driver_version, str)
    assert isinstance(device.is_cuda, bool)
    assert isinstance(device.is_gpu, bool)
    assert isinstance(device.is_accelerator, bool)
    assert isinstance(device.is_cpu, bool)
    assert isinstance(device.simd_group_size, int)
    assert isinstance(device.cache_size, int)
