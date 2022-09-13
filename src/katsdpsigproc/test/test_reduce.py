"""Tests for wg_reduce.mako and :mod:`katsdpsigproc.reduce`."""

from typing import List, Tuple

import numpy as np

from .test_accel import device_test, force_autotune
from ..accel import DeviceArray, build
from .. import reduce
from ..abc import AbstractContext, AbstractCommandQueue


class Fixture:
    def __init__(self, context: AbstractContext, rs: np.random.RandomState,
                 size: int, allow_shuffle: bool, broadcast: bool) -> None:
        rows = 256 // size
        self.data = rs.randint(0, 1000, size=(rows, size)).astype(np.int32)
        self.broadcast = broadcast
        self.program = build(context, 'test/test_reduce.mako', {
            'size': size,
            'allow_shuffle': allow_shuffle,
            'broadcast': broadcast,
            'rows': rows})
        self._description = f"Fixture(..., {size}, {allow_shuffle}, {broadcast})"

    def __str__(self) -> str:
        return self._description


_fixtures = []    # type: List[Fixture]


@device_test
def setup(context: AbstractContext, queue: AbstractContext):
    global _fixtures
    rs = np.random.RandomState(seed=1)  # Fixed seed to make test repeatable
    _fixtures = [Fixture(context, rs, size, allow_shuffle, broadcast)
                 for size in [1, 4, 12, 16, 32, 87, 97, 160, 256]
                 for allow_shuffle in [True, False]
                 for broadcast in [True, False]]


def check_reduce(context: AbstractContext, queue: AbstractCommandQueue,
                 kernel_name: str, op: np.ufunc) -> None:
    for fixture in _fixtures:
        data = fixture.data
        kernel = fixture.program.get_kernel(kernel_name)
        data_d = DeviceArray(context=context, shape=data.shape, dtype=data.dtype)
        data_d.set(queue, data)
        out_d = DeviceArray(context=context, shape=data.shape, dtype=data.dtype)
        dims = tuple(reversed(data.shape))
        queue.enqueue_kernel(
            kernel, [data_d.buffer, out_d.buffer], local_size=dims, global_size=dims)
        out = out_d.get(queue)
        expected = op.reduce(data, axis=1, keepdims=True)
        if fixture.broadcast:
            expected = np.repeat(expected, data.shape[1], axis=1)
        else:
            out = out[:, 0:1]
        np.testing.assert_array_equal(expected, out)


@device_test
def test_reduce_add(context: AbstractContext, queue: AbstractCommandQueue) -> None:
    check_reduce(context, queue, 'test_reduce_add', np.add)


@device_test
def test_reduce_max(context: AbstractContext, queue: AbstractCommandQueue) -> None:
    check_reduce(context, queue, 'test_reduce_max', np.maximum)


class TestHReduce:
    """Tests for :class:`katsdpsigproc.reduce.HReduce`."""

    def check(self, context: AbstractContext, queue: AbstractCommandQueue,
              rows: int, columns: int, column_range: Tuple[int, int]) -> None:
        template = reduce.HReduceTemplate(context, np.uint32, 'unsigned int', 'a + b', '0')
        fn = template.instantiate(queue, (rows, columns), column_range)
        fn.ensure_all_bound()
        device_src = fn.buffer('src')
        device_dest = fn.buffer('dest')
        src = device_src.empty_like()
        rs = np.random.RandomState(seed=1)  # To be reproducible
        src[:] = rs.randint(0, 100000, (rows, columns))
        device_src.set(queue, src)
        fn()
        dest = device_dest.get(queue)
        expected = np.sum(src[:, column_range[0]:column_range[1]], axis=1)
        np.testing.assert_equal(expected, dest)

    @device_test
    def test_normal(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Test the usual case."""
        self.check(context, queue, 129, 173, (67, 128))

    @device_test
    def test_small(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Test that the special case for columns < work group size works."""
        self.check(context, queue, 7, 8, (1, 7))

    @device_test
    @force_autotune
    def test_autotune(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Test that autotuner runs successfully."""
        reduce.HReduceTemplate(context, np.uint32, 'unsigned int', 'a + b', '0')
