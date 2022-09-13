"""Tests for :mod:`katsdpsigproc.accel` and test utilities for other modules."""

import sys
import functools
import inspect
import shutil
import os
import tempfile
from textwrap import dedent
from unittest import mock
from typing import Tuple, Optional, Callable, Awaitable, Type, TypeVar, Any

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore
from decorator import decorator
from mako.template import Template

from nose.tools import assert_equal, assert_raises
from nose.plugins.skip import SkipTest

from .. import accel, tune
from ..accel import HostArray, DeviceArray, SVMArray, LinenoLexer
from ..abc import AbstractContext, AbstractCommandQueue
if accel.have_cuda:
    import pycuda
if accel.have_opencl:
    from .. import opencl
    import pyopencl


_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_test_context = None           # type: Optional[AbstractContext]
_test_command_queue = None     # type: Optional[AbstractCommandQueue]
_test_initialized = False


def _prepare_device_test() -> Tuple[AbstractContext, AbstractCommandQueue]:
    global _test_initialized, _test_context, _test_command_queue
    if not _test_initialized:
        try:
            _test_context = accel.create_some_context(False)
            _test_command_queue = _test_context.create_command_queue()
            print("Testing on {} ({})".format(
                _test_context.device.name, _test_context.device.platform_name),
                file=sys.stderr)
        except RuntimeError:
            pass  # No devices available
        _test_initialized = True

    if not _test_context:
        raise SkipTest('CUDA/OpenCL not found')
    assert _test_command_queue is not None
    return _test_context, _test_command_queue


def _device_test_sync(test: Callable[..., _T]) -> Callable[..., _T]:
    @functools.wraps(test)
    def wrapper(*args, **kwargs) -> _T:
        context, command_queue = _prepare_device_test()
        with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner):
            args += (context, command_queue)
            # Make the context current (for CUDA contexts). Ideally the test
            # should not depend on this, but PyCUDA leaks memory if objects
            # are deleted without the context current.
            with context:
                return test(*args, **kwargs)
    return wrapper


def _device_test_async(test: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
    @functools.wraps(test)
    async def wrapper(*args, **kwargs) -> _T:
        context, command_queue = _prepare_device_test()
        with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner):
            args += (context, command_queue)
            with context:
                return await test(*args, **kwargs)
    return wrapper


def device_test(test: Callable[..., _T]) -> Callable[..., _T]:
    """Decorate an on-device test.

    It provides a context and command queue to the test, skipping it if a
    compute device is not available. It also disables autotuning, instead
    using the `test` value provided for the autotune function.

    If autotuning is desired, use :func:`force_autotune` inside (hence,
    afterwards on the decorator list) this one.
    """
    if inspect.iscoroutinefunction(test):
        return _device_test_async(test)    # type: ignore
    else:
        return _device_test_sync(test)


def cuda_test(test: _F) -> _F:
    """Skip a test if the device is not a CUDA device.

    Put this *after* :meth:`device_test`.
    """
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        global _test_context
        if not _test_context.device.is_cuda:
            raise SkipTest('Device is not a CUDA device')
        return test(*args, **kwargs)
    return wrapper     # type: ignore


@decorator
def force_autotune(test: Callable[..., _T], *args, **kw) -> _T:
    """Force autotuning for a test (decorator).

    It bypasses the autotuning cache so that the autotuning code always runs.
    """
    with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.force_autotuner):
        return test(*args, **kw)


# Prevent nose from treating it as a test
device_test.__test__ = False         # type: ignore
cuda_test.__test__ = False           # type: ignore


class TestLinenoLexer:
    def test_escape_filename(self) -> None:
        assert_equal(
            r'"abc\"def\\ghi"',
            LinenoLexer._escape_filename(r'abc"def\ghi'))

    def test_render(self) -> None:
        source = "line 1\nline 2\nline 3"
        out = Template(source, lexer_cls=LinenoLexer).render()
        assert_equal(dedent(
            """\
            #line 1 "<string>"
            line 1
            #line 2 "<string>"
            line 2
            #line 3 "<string>"
            line 3
            """), out)


class TestHostArray:
    cls = HostArray      # type: Type[HostArray]

    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 padded_shape: Tuple[int, ...]) -> HostArray:
        return self.cls(shape, dtype, padded_shape)

    def setup(self) -> None:
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = self.allocate(self.shape, np.int32, self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(self.cls)
        self.sliced = self.constructed[2:4, 3:5]

    def test_safe(self) -> None:
        assert self.cls.safe(self.constructed)
        assert not self.cls.safe(self.view)
        assert not self.cls.safe(self.sliced)
        assert not self.cls.safe(np.zeros(self.shape))


class TestDeviceArray:
    cls = DeviceArray      # type: Type[DeviceArray]

    @device_test
    def setup(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = self.cls(
            context=context,
            shape=self.shape,
            dtype=np.int32,
            padded_shape=self.padded_shape)

    @device_test
    def test_strides(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        assert_equal(self.strides, self.array.strides)

    @device_test
    def test_empty_like(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        ary = self.array.empty_like()
        assert_equal(self.shape, ary.shape)
        assert_equal(self.strides, ary.strides)
        assert HostArray.safe(ary)

    @device_test
    def test_set_get(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        ary = np.random.randint(0, 100, self.shape).astype(np.int32)
        self.array.set(queue, ary)
        # Read back results, check that it matches
        buf = np.zeros(self.padded_shape, dtype=np.int32)
        if isinstance(self.array, SVMArray):
            buf[:] = self.array.buffer
        elif context.device.is_cuda:
            with context:
                pycuda.driver.memcpy_dtoh(buf, self.array.buffer.gpudata)
        else:
            assert isinstance(queue, opencl.CommandQueue)
            pyopencl.enqueue_copy(
                queue._pyopencl_command_queue,
                buf, self.array.buffer.data)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = self.array.get(queue)
        np.testing.assert_equal(ary, buf)

    def _test_copy_region(
            self, context: AbstractContext, queue: AbstractCommandQueue,
            src_shape: Tuple[int, ...], src_padded_shape: Optional[Tuple[int, ...]],
            dest_shape: Tuple[int, ...], dest_padded_shape: Optional[Tuple[int, ...]],
            src_region: accel._Slice, dest_region: accel._Slice) -> None:
        # copy_region test
        dtype = np.int32
        src = self.cls(context, src_shape, dtype, src_padded_shape)
        dest = self.cls(context, dest_shape, dtype, dest_padded_shape)
        h_src = src.empty_like()
        h_src[...] = np.arange(h_src.size).reshape(src_shape)
        src.set(queue, h_src)
        dest.zero(queue)
        src.copy_region(queue, dest, src_region, dest_region)
        queue.finish()   # Needed for the SVMArray test, since the next line is a CPU copy
        h_dest = dest.get(queue)
        expected = np.zeros_like(h_dest)
        expected[dest_region] = h_src[src_region]
        np.testing.assert_array_equal(expected, h_dest)
        # set_region test
        dest.zero(queue)
        dest.set_region(queue, h_src, dest_region, src_region)
        queue.finish()
        h_dest = dest.get(queue)
        np.testing.assert_array_equal(expected, h_dest)
        # get_region test
        h_dest.fill(0)
        src.get_region(queue, h_dest, src_region, dest_region)
        queue.finish()
        np.testing.assert_array_equal(expected, h_dest)

    @device_test
    def test_copy_region_4d(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, queue,
            (7, 5, 12, 19), (8, 9, 14, 25),
            (7, 5, 12, 19), (10, 8, 12, 21),
            np.s_[2:4, :, 3:7:2, 10:], np.s_[1:3, 0:5, 4:8:2, 10:])

    @device_test
    def test_copy_region_0d(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self._test_copy_region(context, queue, (), (), (), (), (), ())

    @device_test
    def test_copy_region_1d(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, queue, (3, 5), (4, 6), (5, 7), (7, 10),
            np.s_[1, 0:3], np.s_[2, 1:4])

    @device_test
    def test_copy_region_2d(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, queue, (3, 5), (4, 6), (5, 7), (7, 10),
            np.s_[:1, 0:3], np.s_[-1:, 1:4])

    @device_test
    def test_copy_region_missing_axes(self, context: AbstractContext,
                                      queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, queue, (3, 5), (4, 6), (5, 5), (7, 10),
            np.s_[:1], np.s_[-1:])

    @device_test
    def test_copy_region_newaxis(self, context: AbstractContext,
                                 queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, queue, (10,), None, (10, 10), None,
            np.s_[np.newaxis, 4:6], np.s_[:1, 7:9])

    @device_test
    def test_copy_region_negative(self, context: AbstractContext,
                                  queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, queue, (10, 12), None, (7, 8), None,
            np.s_[-4, -3:-1], np.s_[-7, -8:-4:2])

    @device_test
    def test_copy_region_errors(self, context: AbstractContext,
                                queue: AbstractCommandQueue) -> None:
        # Too many axes
        with assert_raises(IndexError):
            self._test_copy_region(
                context, queue, (10,), None, (10,), None,
                np.s_[3, 4], np.s_[5, 6])
        # Out-of-range single index
        with assert_raises(IndexError):
            self._test_copy_region(
                context, queue, (10,), None, (10,), None,
                np.s_[5], np.s_[10])
        # Out-of-range slice
        with assert_raises(IndexError):
            self._test_copy_region(
                context, queue, (10,), None, (10,), None,
                np.s_[10:12], np.s_[8:10])
        # Empty slice
        with assert_raises(IndexError):
            self._test_copy_region(
                context, queue, (10,), None, (10,), None,
                np.s_[2:2], np.s_[3:3])
        # Negative stride
        with assert_raises(IndexError):
            self._test_copy_region(
                context, queue, (10,), None, (10,), None,
                np.s_[3:0:-1], np.s_[4:1:-1])

    @device_test
    def test_zero(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        ary = np.random.randint(0x12345678, 0x23456789, self.shape).astype(np.int32)
        self.array.set(queue, ary)
        before = self.array.get(queue)
        self.array.zero(queue)
        after = self.array.get(queue)
        np.testing.assert_equal(ary, before)
        assert_equal(0, np.max(after))

    def _allocate_raw(self, context: AbstractContext, n_bytes: int) -> Any:
        return context.allocate_raw(n_bytes)

    @device_test
    def test_raw(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        raw = self._allocate_raw(context, 2048)
        ary = self.cls(
            context=context,
            shape=self.shape,
            dtype=np.int32,
            padded_shape=self.padded_shape,
            raw=raw)
        actual_raw = None
        try:
            # CUDA
            if isinstance(ary, SVMArray):
                actual_raw = ary.base.base  # type: ignore
                raw = raw._wrapped
            else:
                actual_raw = ary.buffer.gpudata
        except AttributeError:
            # OpenCL
            actual_raw = ary.buffer.data
        assert actual_raw is raw


class TestPinnedAMD(TestDeviceArray):
    """Run DeviceArray tests forcing `_PinnedAMD` class for pinned memory."""

    @device_test
    def setup(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        context._force_pinned_amd = True        # type: ignore
        super().setup()

    @device_test
    def teardown(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        context._force_pinned_amd = False       # type: ignore


class TestSVMArrayHost(TestHostArray):
    """Test SVMArray using the HostArray tests."""

    cls = SVMArray

    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 padded_shape: Tuple[int, ...]) -> SVMArray:
        return self.cls(self.context, shape, dtype, padded_shape)

    @device_test
    @cuda_test
    def setup(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        self.context = context
        super().setup()


class TestSVMArray(TestDeviceArray):
    """Tests SVMArray using the DeviceArray tests, plus some new ones."""

    cls = SVMArray

    def _allocate_raw(self, context: AbstractContext, n_bytes: int) -> object:
        return context.allocate_svm_raw(n_bytes)

    @device_test
    @cuda_test
    def setup(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        super().setup()

    @device_test
    @cuda_test
    def test_coherence(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        """Check that runtime correct transfers data between the host and device views."""
        source = """\
            <%include file="/port.mako"/>
            KERNEL void triple(unsigned int *data)
            {
                data[get_global_id(0)] *= 3;
            }"""
        program = accel.build(context, 'not a file', source=source)
        kernel = program.get_kernel("triple")
        ary = SVMArray(context, (123,), np.uint32, (128,))
        ary[:] = np.arange(123, dtype=np.uint32)
        queue.enqueue_kernel(kernel, [ary.buffer], (128,), (64,))
        queue.finish()
        np.testing.assert_equal(np.arange(0, 369, step=3, dtype=np.uint32), ary)


class TestSVMAllocator:
    @device_test
    @cuda_test
    def test_allocate(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        allocator = accel.SVMAllocator(context)
        ary = allocator.allocate((12, 34), np.int32, (13, 35))
        assert isinstance(ary, accel.SVMArray)
        assert_equal((12, 34), ary.shape)
        assert_equal(np.int32, ary.dtype)
        assert_equal((13, 35), ary.padded_shape)
        ary.fill(1)  # Just to make sure it doesn't crash

    @device_test
    @cuda_test
    def test_allocate_raw(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        allocator = accel.SVMAllocator(context)
        raw = allocator.allocate_raw(13 * 35 * 4)
        ary = allocator.allocate((12, 34), np.int32, (13, 35), raw=raw)
        assert isinstance(ary, accel.SVMArray)
        assert_equal((12, 34), ary.shape)
        assert_equal(np.int32, ary.dtype)
        assert_equal((13, 35), ary.padded_shape)
        ary.fill(1)  # Just to make sure it doesn't crash


class TestDimension:
    """Tests for :class:`katsdpsigproc.accel.Dimension`."""

    def test_is_power2(self) -> None:
        assert accel.Dimension._is_power2(1)
        assert accel.Dimension._is_power2(2)
        assert accel.Dimension._is_power2(32)
        assert not accel.Dimension._is_power2(-1)
        assert not accel.Dimension._is_power2(0)
        assert not accel.Dimension._is_power2(3)
        assert not accel.Dimension._is_power2(5)

    def test_min_padded_round(self) -> None:
        """Constructor computes min padded size correctly."""
        dim = accel.Dimension(17, min_padded_round=4)
        assert_equal(20, dim.min_padded_size)
        dim = accel.Dimension(20, min_padded_round=5)
        assert_equal(20, dim.min_padded_size)

    def test_add_align_dtype(self) -> None:
        dim = accel.Dimension(20, alignment=8)
        assert_equal(8, dim.alignment)
        dim.add_align_dtype(np.complex64)
        assert_equal(accel.Dimension.ALIGN_BYTES // 8, dim.alignment_hint)
        dim.add_align_dtype(np.uint8)
        assert_equal(accel.Dimension.ALIGN_BYTES, dim.alignment_hint)
        dim.add_align_dtype(np.float32)
        assert_equal(accel.Dimension.ALIGN_BYTES, dim.alignment_hint)

    def test_valid(self) -> None:
        """Test `valid` method on non-exact dimension."""
        dim = accel.Dimension(17, min_padded_round=8, alignment=4)
        assert dim.valid(24)
        assert not dim.valid(20)
        assert dim.valid(28)
        assert not dim.valid(30)

    def test_valid_exact(self) -> None:
        """Test `valid` method on exact dimension."""
        dim = accel.Dimension(20, alignment=4, exact=True)
        assert dim.valid(20)
        assert not dim.valid(24)
        dim = accel.Dimension(20, min_padded_size=23, exact=True)
        assert dim.valid(23)
        assert not dim.valid(24)
        assert not dim.valid(20)

    @classmethod
    def assert_dimensions_equal(cls, dim1: accel.Dimension, dim2: accel.Dimension) -> None:
        assert_equal(dim1.size, dim2.size)
        assert_equal(dim1.min_padded_size, dim2.min_padded_size)
        assert_equal(dim1.alignment, dim2.alignment)
        assert_equal(dim1.alignment_hint, dim2.alignment_hint)
        assert_equal(dim1.exact, dim2.exact)

    def test_link(self) -> None:
        """Linking several dimensions together gives them all the right properties."""
        dim1 = accel.Dimension(22, min_padded_size=28, alignment=4)
        dim2 = accel.Dimension(22, min_padded_size=24, alignment=8, align_dtype=np.int32)
        dim3 = accel.Dimension(22, min_padded_size=22, align_dtype=np.uint16)
        dim1.link(dim2)
        dim1.link(dim3)
        self.assert_dimensions_equal(dim1, dim2)
        self.assert_dimensions_equal(dim2, dim3)
        assert_equal(22, dim1.size)
        assert_equal(28, dim1.min_padded_size)
        assert_equal(8, dim1.alignment)
        assert_equal(accel.Dimension.ALIGN_BYTES // 2, dim1.alignment_hint)
        assert_equal(False, dim1.exact)

    def test_link_bad_size(self) -> None:
        """Linking dimensions with different sizes fails."""
        dim1 = accel.Dimension(22, min_padded_size=28, alignment=4)
        dim2 = accel.Dimension(23, min_padded_size=24, alignment=8, align_dtype=np.int32)
        with assert_raises(ValueError):
            dim1.link(dim2)
        assert dim1._root() is not dim2._root()

    def test_link_bad_exact(self) -> None:
        """Linking dimensions into an unsatisfiable requirement fails."""
        dim1 = accel.Dimension(22, exact=True)
        dim2 = accel.Dimension(22, min_padded_size=28)
        dim3 = accel.Dimension(22, alignment=4)
        with assert_raises(ValueError):
            dim1.link(dim2)
        with assert_raises(ValueError):
            dim1.link(dim3)
        # Check that linking didn't happen anyway
        assert dim1._root() is not dim2._root()
        assert dim1._root() is not dim3._root()

    def test_required_padded_size(self) -> None:
        """The padded size is computed correctly."""
        dim = accel.Dimension(30, 7, alignment=4)
        assert_equal(36, dim.required_padded_size())

    def test_required_padded_size_dtype(self) -> None:
        """The padded size is computed correctly when an alignment hint is given."""
        dim = accel.Dimension(1100, 200, align_dtype=np.float32)
        assert_equal(1216, dim.required_padded_size())

    def test_required_padded_size_exact(self) -> None:
        """The padded size is computed correctly for exact dimensions."""
        dim = accel.Dimension(1100, align_dtype=np.float32, exact=True)
        assert_equal(1100, dim.required_padded_size())

    def test_required_padded_size_small(self) -> None:
        """The alignment hint is ignored for small sizes."""
        dim = accel.Dimension(18, alignment=8, align_dtype=np.uint8)
        assert_equal(24, dim.required_padded_size())


class TestIOSlot:
    """Tests for :class:`katsdpsigproc.accel.IOSlot`."""

    @mock.patch('katsdpsigproc.accel.DeviceArray', spec=True)
    def test_allocate(self, DeviceArray: mock.Mock) -> None:
        """IOSlot.allocate must correctly allocate a buffer."""
        dims = (
            accel.Dimension(50, min_padded_size=60, alignment=8),
            accel.Dimension(30, min_padded_size=50, alignment=4)
        )
        shape = (50, 30)
        padded_shape = (64, 52)
        dtype = np.dtype(np.uint8)
        # Create the device array that will be created. We need to populate it
        # with some attributes to allow validation to pass
        ary = mock.Mock()
        ary.dtype = dtype
        ary.shape = shape
        ary.padded_shape = padded_shape
        # Set the mocked DeviceArray class to return this array
        DeviceArray.return_value = ary
        # Run the system under test
        slot = accel.IOSlot(dims, dtype)
        ret = slot.allocate(accel.DeviceAllocator(mock.sentinel.context))
        # Validation
        assert_equal(ary, ret)
        assert_equal(ary, slot.buffer)
        DeviceArray.assert_called_once_with(
            mock.sentinel.context, shape, dtype, padded_shape, None)
        # Check that the inner dimension had a type hint set
        assert dims[1].alignment_hint == accel.Dimension.ALIGN_BYTES

    @mock.patch('katsdpsigproc.accel.DeviceArray', spec=True)
    def test_allocate_raw(self, DeviceArray: mock.Mock) -> None:
        """Test IOSlot.allocate with a raw parameter."""
        shape = (50, 30)
        dtype = np.dtype(np.float32)
        raw = mock.sentinel.raw
        # Create the device array that will be created. We need to populate it
        # with some attributes to allow validation to pass
        ary = mock.Mock()
        ary.dtype = dtype
        ary.shape = shape
        ary.padded_shape = shape
        ary.raw = raw
        # Set the mocked DeviceArray class to return this array
        DeviceArray.return_value = ary
        # Run the system under test
        slot = accel.IOSlot(shape, dtype)
        slot.allocate(accel.DeviceAllocator(mock.sentinel.context), raw)
        # Validation
        assert_equal(ary, slot.buffer)
        DeviceArray.assert_called_once_with(
            mock.sentinel.context, shape, dtype, shape, raw)

    def test_validate_shape(self) -> None:
        """IOSlot.validate must check that the shape matches."""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = (5, 3)
        slot = accel.IOSlot((5, 3), np.float32)
        slot.bind(ary)  # Should pass

        with assert_raises(ValueError):
            # Wrong shape
            ary.shape = (5, 4)
            slot.bind(ary)
        with assert_raises(ValueError):
            # Wrong dimensions
            ary.shape = (5,)
            slot.bind(ary)

    def test_validate_dtype(self) -> None:
        """IOSlot.validate must check that the dtype matches."""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = ary.shape
        slot = accel.IOSlot((5, 3), np.float32)
        slot.bind(ary)  # Should pass

        ary.dtype = np.dtype(np.float32)  # Equivalent dtype
        slot.bind(ary)  # Should pass

        with assert_raises(TypeError):
            ary.dtype = np.int32
            slot.bind(ary)

    def test_validate_padded_shape(self) -> None:
        """IOSlot.validate must check that the padded shape is valid."""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = (8, 6)
        slot = accel.IOSlot((accel.Dimension(5, 8), accel.Dimension(3, 6)), np.float32)
        slot.bind(ary)  # Should pass

        ary.padded_shape = (10, 11)
        with assert_raises(ValueError):
            # Bigger than needed, but not an exact match - must fail
            slot.bind(ary)

    def test_required_bytes(self) -> None:
        slot = accel.IOSlot(
            (accel.Dimension(27, alignment=4), accel.Dimension(33, alignment=32)),
            np.float32)
        assert_equal(4 * 28 * 64, slot.required_bytes())

    def test_bind_none(self) -> None:
        """IOSlot.bind must accept `None`."""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = ary.shape
        slot = accel.IOSlot((5, 3), np.float32)
        slot.bind(ary)
        assert slot.buffer is ary
        slot.bind(None)
        assert slot.buffer is None


class TestCompoundIOSlot:
    """Tests for :class:`katsdpsigproc.accel.CompoundIOSlot`."""

    def setup(self) -> None:
        self.dims1 = (
            accel.Dimension(13, min_padded_size=17, alignment=1),
            accel.Dimension(7, min_padded_size=8, alignment=8),
            accel.Dimension(22, min_padded_size=25, alignment=4)
        )
        self.dims2 = (
            accel.Dimension(13, min_padded_size=14, alignment=4),
            accel.Dimension(7, min_padded_size=10, alignment=4),
            accel.Dimension(22, min_padded_size=22, alignment=1)
        )
        self.slot1 = accel.IOSlot(self.dims1, np.float32)
        self.slot2 = accel.IOSlot(self.dims2, np.float32)

    def test_check_empty(self) -> None:
        """:class:`.CompoundIOSlot` constructor must reject empty list."""
        with assert_raises(ValueError):
            accel.CompoundIOSlot([])

    def test_validate_shape(self) -> None:
        """:class:`.CompoundIOSlot` must check that children have consistent shapes."""
        slot1 = accel.IOSlot((5, 3), np.float32)
        slot2 = accel.IOSlot((5, 4), np.float32)
        with assert_raises(ValueError):
            accel.CompoundIOSlot([slot1, slot2])

    def test_validate_dtype(self) -> None:
        """:class:`.CompoundIOSlot` must check that children have consistent data types."""
        slot1 = accel.IOSlot((5, 3), np.float32)
        slot2 = accel.IOSlot((5, 3), np.int32)
        with assert_raises(TypeError):
            accel.CompoundIOSlot([slot1, slot2])

    def test_attributes(self) -> None:
        """:class:`.CompoundIOSlot` must correctly combine attributes."""
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        assert_equal((13, 7, 22), slot.shape)
        assert_equal(np.float32, slot.dtype)
        assert_equal(17, self.dims1[0].min_padded_size)
        assert_equal(10, self.dims1[1].min_padded_size)
        assert_equal(25, self.dims1[2].min_padded_size)
        assert_equal(4, self.dims1[0].alignment)
        assert_equal(8, self.dims1[1].alignment)
        assert_equal(4, self.dims1[2].alignment)
        for x, y in zip(self.dims1, self.dims2):
            TestDimension.assert_dimensions_equal(x, y)

    def test_bind(self) -> None:
        """CompoundIOSlot.bind must bind children."""
        ary = mock.sentinel.ary
        ary.shape = (13, 7, 22)
        ary.dtype = np.float32
        ary.padded_shape = (20, 16, 28)
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        slot.bind(ary)
        assert_equal(ary, slot.buffer)
        assert_equal(ary, self.slot1.buffer)
        assert_equal(ary, self.slot2.buffer)

    def test_bind_fail(self) -> None:
        """CompoundIOSlot.bind must not have side effects if the array is \
        not valid for all children."""
        ary = mock.sentinel.ary
        ary.shape = (13, 7, 22)
        ary.dtype = np.float32
        ary.padded_shape = (17, 8, 28)
        # Check that the array has the properties we want for the test
        self.slot1.validate(ary)
        with assert_raises(ValueError):
            self.slot2.validate(ary)
        # The actual test
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        with assert_raises(ValueError):
            slot.bind(ary)
        assert slot.buffer is None
        assert self.slot1.buffer is None
        assert self.slot2.buffer is None


class TestAliasIOSlot:
    """Tests for :class:`katsdpsigproc.accel.AliasIOSlot`."""

    def setup(self) -> None:
        self.slot1 = accel.IOSlot((3, 7), np.float32)
        self.slot2 = accel.IOSlot((5, 3), np.complex64)

    def test_required_bytes(self) -> None:
        slot = accel.AliasIOSlot([self.slot1, self.slot2])
        assert_equal(120, slot.required_bytes())

    def test_allocate(self) -> None:
        # Set up mocks
        raw = mock.sentinel.raw
        context = mock.NonCallableMock()
        context.allocate_raw = mock.Mock(return_value=raw)
        # Run the test
        slot = accel.AliasIOSlot([self.slot1, self.slot2])
        ret = slot.allocate(accel.DeviceAllocator(context))
        # Validation
        context.allocate_raw.assert_called_once_with(120)
        assert ret is raw
        assert slot.raw is raw
        assert self.slot1.buffer is not None
        assert self.slot2.buffer is not None


class TestVisualizeOperation:
    """Tests for :class:`katsdpsigproc.accel.visualize_operation`.

    This is just a basic smoke test to ensure that it runs without crashing.
    A real test requires a human to sanity-check the visual output.
    """

    class Inner(accel.Operation):
        def _run(self) -> None:
            pass

    def setup(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown(self) -> None:
        shutil.rmtree(self.tmpdir)

    @device_test
    def test(self, context: AbstractContext, queue: AbstractCommandQueue) -> None:
        inner = TestVisualizeOperation.Inner(queue)
        inner.slots['foo'] = accel.IOSlot((3, 4), np.float32)
        inner.slots['bar'] = accel.IOSlot((4, 3), np.float32)

        mid = accel.OperationSequence(
            queue,
            [('inner', inner)],
            {'inner_foo': ['inner:foo']})

        outer = accel.OperationSequence(
            queue,
            [('mid', mid)],
            {'mid_foo': ['mid:inner_foo']},
            {'scratch': ['mid:inner:bar']})

        accel.visualize_operation(outer, os.path.join(self.tmpdir, 'test_visualize.gv'))
