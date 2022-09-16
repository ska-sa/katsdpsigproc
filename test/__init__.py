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
