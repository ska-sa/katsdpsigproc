################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Plugin for testing with pytest.

See the :ref:`testing-pytest` section of the documentation for details.
"""

import itertools
from typing import Generator

import pytest

from . import abc, accel, tune


@pytest.fixture
def patch_autotune(request, monkeypatch) -> None:
    autotuner = tune.stub_autotuner
    if request.node.get_closest_marker("force_autotune"):
        autotuner = tune.force_autotuner
    monkeypatch.setattr(tune, "autotuner_impl", autotuner)


@pytest.fixture
def context(device: abc.AbstractDevice,
            patch_autotune) -> Generator[abc.AbstractContext, None, None]:
    # Make the context current (for CUDA contexts). Ideally the test
    # should not depend on this, but PyCUDA leaks memory if objects
    # are deleted without the context current.
    with device.make_context() as context:
        yield context


@pytest.fixture
def command_queue(context: abc.AbstractContext) -> abc.AbstractCommandQueue:
    return context.create_command_queue()


def pytest_addoption(parser):
    group = parser.getgroup("katsdpsigproc")
    group.addoption(
        "--devices",
        choices=["first-per-api", "all", "none"],
        default="first-per-api",
        help="Select which devices to use for testing"
    )


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "force_autotune: unconditionally run autotuning")
    config.addinivalue_line("markers", "cuda_only: run test only on CUDA devices")
    config.addinivalue_line("markers", "opencl_only: run test only on OpenCL devices")
    config.addinivalue_line(
        "markers", "device_filter(filter): run test only on devices matching 'filter'")


def pytest_generate_tests(metafunc) -> None:
    if "device" in metafunc.fixturenames:
        option = metafunc.config.getoption("devices")
        if option == "none":
            metafunc.parametrize(
                "device",
                [
                    pytest.param(
                        None, marks=pytest.mark.skip(reason="--devices=none passed on command line")
                    )
                ]
            )
            return

        devices = accel.candidate_devices()
        # Apply filters. This is done after calling candidate_devices rather
        # than by passing a device filter, so that
        # environment variables like KATSDPSIGPROC_DEVICE are interpreted
        # relative to the full device list rather than a filtered list, which
        # could cause tests with different filters to select different
        # devices.
        for marker in metafunc.definition.iter_markers("cuda_only"):
            min_cc = marker.kwargs.get("min_compute_capability", (0, 0))
            devices = [
                device
                for device in devices
                if device.is_cuda and device.compute_capability >= min_cc]  # type: ignore
        if metafunc.definition.get_closest_marker("opencl_only") is not None:
            devices = [device for device in devices if not device.is_cuda]
        for marker in metafunc.definition.iter_markers("device_filter"):
            devices = [device for device in devices if marker.args[0](device)]

        # Apply --devices command-line setting
        if option == "first-per-api":
            # Sort so that itertools.groupby finds all devices from the same
            # API together.
            def classify_api(device: abc.AbstractDevice) -> bool:
                return device.is_cuda  # Currently only 2 APIs, so this is sufficient

            devices = sorted(devices, key=classify_api)
            # group is an iterable, so next(group) gives the first element
            devices = [next(group) for _, group in itertools.groupby(devices, key=classify_api)]
        # Nothing needed for --devices=all, and --devices=none handled earlier

        ids = [f"{d.name} ({d.platform_name})" for d in devices]
        if not devices:
            metafunc.parametrize(
                "device",
                [
                    pytest.param(
                        None, marks=pytest.mark.xfail(reason="No matching device found", run=False)
                    )
                ]
            )
        else:
            metafunc.parametrize("device", devices, ids=ids)
