################################################################################
# Copyright (c) 2019, National Research Foundation (SARAO)
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

"""Utilities for scheduling device operations with asyncio."""

import warnings

from ..resource import *  # noqa: F403

warnings.warn(
    "katsdpsigproc.asyncio.resource is deprecated. Use katsdpsigproc.resource instead",
    DeprecationWarning,
)
