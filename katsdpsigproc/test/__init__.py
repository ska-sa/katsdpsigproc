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


def complex_uniform(
    state: _RS,
    low: ArrayLike = 0.0,
    high: ArrayLike = 1.0,
    size: Union[int, Tuple[int, ...], None] = None
):
    """Generate a uniform distribution over a rectangle in the complex plane.

    If low or high is purely real (by dtype, not by value), it is taken to
    be the boundary on both the real and imaginary extent.
    """
    if not np.iscomplexobj(low):
        low = np.asarray(low) * (1 + 1j)
    if not np.iscomplexobj(high):
        high = np.asarray(high) * (1 + 1j)
    return state.uniform(np.real(low), np.real(high), size) \
        + 1j * state.uniform(np.imag(low), np.imag(high), size)
