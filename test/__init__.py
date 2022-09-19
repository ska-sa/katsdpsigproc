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

"""Unit tests for :mod:`katsdpsigproc`."""

from typing import Any, Tuple, Union

import numpy as np
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore


_RS = Union[np.random.RandomState, np.random.Generator]


def complex_normal(
    state: _RS,
    loc: ArrayLike = 0.0j,
    scale: ArrayLike = 1.0,
    size: Union[int, Tuple[int, ...], None] = None
):
    """Generate a circularly symmetric Gaussian in the Argand plane."""
    return (
        state.normal(np.real(loc), scale, size)         # type: ignore
        + 1j * state.normal(np.imag(loc), scale, size)  # type: ignore
    )
