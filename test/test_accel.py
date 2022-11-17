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

"""Tests for :mod:`katsdpsigproc.accel`."""

import shutil
import os
import tempfile
from textwrap import dedent
from unittest import mock
from typing import Tuple, Optional, Callable, Type, Any

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore
import pytest
from mako.template import Template

from katsdpsigproc import accel
from katsdpsigproc.accel import HostArray, DeviceArray, SVMArray, LinenoLexer
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue
if accel.have_cuda:
    import pycuda
if accel.have_opencl:
    from katsdpsigproc import opencl
    import pyopencl


class TestLinenoLexer:
    def test_escape_filename(self) -> None:
        assert LinenoLexer._escape_filename(r'abc"def\ghi') == r'"abc\"def\\ghi"'

    def test_render(self) -> None:
        source = "line 1\nline 2\nline 3"
        out = Template(source, lexer_cls=LinenoLexer).render()
        assert out == dedent(
            """\
            #line 1 "<string>"
            line 1
            #line 2 "<string>"
            line 2
            #line 3 "<string>"
            line 3
            """)


class TestHostArray:
    cls = HostArray      # type: Type[HostArray]

    @pytest.fixture
    def factory(self) -> Callable[[Tuple[int, ...], DTypeLike, Tuple[int, ...]], HostArray]:
        def allocate(shape: Tuple[int, ...], dtype: DTypeLike,
                     padded_shape: Tuple[int, ...]) -> HostArray:
            return self.cls(shape, dtype, padded_shape)
        return allocate

    def test_safe(
            self,
            factory: Callable[[Tuple[int, ...], DTypeLike, Tuple[int, ...]], HostArray]) -> None:
        shape = (17, 13)
        padded_shape = (32, 16)
        constructed = factory(shape, np.int32, padded_shape)
        view = np.zeros(padded_shape)[2:4, 3:5].view(self.cls)
        sliced = constructed[2:4, 3:5]
        assert self.cls.safe(constructed)
        assert not self.cls.safe(view)
        assert not self.cls.safe(sliced)
        assert not self.cls.safe(np.zeros(shape))


class TestDeviceArray:
    cls = DeviceArray      # type: Type[DeviceArray]

    @pytest.fixture
    def shape(self) -> Tuple[int, ...]:
        return (17, 13)

    @pytest.fixture
    def padded_shape(self) -> Tuple[int, ...]:
        return (32, 16)

    @pytest.fixture
    def strides(self) -> Tuple[int, ...]:
        return (64, 4)

    @pytest.fixture
    def array(self, context: AbstractContext,
              shape: Tuple[int, ...], padded_shape: Tuple[int, ...]) -> DeviceArray:
        return self.cls(
            context=context,
            shape=shape,
            dtype=np.int32,
            padded_shape=padded_shape)

    def test_properties(self, array: DeviceArray, shape: Tuple[int, ...],
                        padded_shape: Tuple[int, ...], strides: Tuple[int, ...]) -> None:
        assert array.shape == shape
        assert array.padded_shape == padded_shape
        assert array.strides == strides
        assert array.dtype == np.int32

    def test_empty_like(self, array: DeviceArray) -> None:
        ary = array.empty_like()
        assert ary.shape == array.shape
        assert ary.strides == array.strides
        assert HostArray.safe(ary)

    def test_set_get(self, array: DeviceArray,
                     context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        ary = np.random.randint(0, 100, array.shape).astype(np.int32)
        array.set(command_queue, ary)
        # Read back results, check that it matches
        buf = np.zeros(array.padded_shape, dtype=np.int32)
        if isinstance(array, SVMArray):
            buf[:] = array.buffer
        elif context.device.is_cuda:
            with context:
                pycuda.driver.memcpy_dtoh(buf, array.buffer.gpudata)
        else:
            assert isinstance(command_queue, opencl.CommandQueue)
            pyopencl.enqueue_copy(
                command_queue._pyopencl_command_queue,
                buf, array.buffer.data)
        buf = buf[0:array.shape[0], 0:array.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = array.get(command_queue)
        np.testing.assert_equal(ary, buf)

    def _test_copy_region(
            self, context: AbstractContext, command_queue: AbstractCommandQueue,
            src_shape: Tuple[int, ...], src_padded_shape: Optional[Tuple[int, ...]],
            dest_shape: Tuple[int, ...], dest_padded_shape: Optional[Tuple[int, ...]],
            src_region: accel._Slice, dest_region: accel._Slice) -> None:
        # copy_region test
        dtype = np.int32
        src = self.cls(context, src_shape, dtype, src_padded_shape)
        dest = self.cls(context, dest_shape, dtype, dest_padded_shape)
        h_src = src.empty_like()
        h_src[...] = np.arange(h_src.size).reshape(src_shape)
        src.set(command_queue, h_src)
        dest.zero(command_queue)
        src.copy_region(command_queue, dest, src_region, dest_region)
        command_queue.finish()   # Needed for the SVMArray test, since the next line is a CPU copy
        h_dest = dest.get(command_queue)
        expected = np.zeros_like(h_dest)
        expected[dest_region] = h_src[src_region]
        np.testing.assert_array_equal(expected, h_dest)
        # set_region test
        dest.zero(command_queue)
        dest.set_region(command_queue, h_src, dest_region, src_region)
        command_queue.finish()
        h_dest = dest.get(command_queue)
        np.testing.assert_array_equal(expected, h_dest)
        # get_region test
        h_dest.fill(0)
        src.get_region(command_queue, h_dest, src_region, dest_region)
        command_queue.finish()
        np.testing.assert_array_equal(expected, h_dest)

    def test_copy_region_4d(self, context: AbstractContext,
                            command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, command_queue,
            (7, 5, 12, 19), (8, 9, 14, 25),
            (7, 5, 12, 19), (10, 8, 12, 21),
            np.s_[2:4, :, 3:7:2, 10:], np.s_[1:3, 0:5, 4:8:2, 10:])

    def test_copy_region_0d(self, context: AbstractContext,
                            command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(context, command_queue, (), (), (), (), (), ())

    def test_copy_region_1d(self, context: AbstractContext,
                            command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, command_queue, (3, 5), (4, 6), (5, 7), (7, 10),
            np.s_[1, 0:3], np.s_[2, 1:4])

    def test_copy_region_2d(self, context: AbstractContext,
                            command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, command_queue, (3, 5), (4, 6), (5, 7), (7, 10),
            np.s_[:1, 0:3], np.s_[-1:, 1:4])

    def test_copy_region_missing_axes(self, context: AbstractContext,
                                      command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, command_queue, (3, 5), (4, 6), (5, 5), (7, 10),
            np.s_[:1], np.s_[-1:])

    def test_copy_region_newaxis(self, context: AbstractContext,
                                 command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, command_queue, (10,), None, (10, 10), None,
            np.s_[np.newaxis, 4:6], np.s_[:1, 7:9])

    def test_copy_region_negative(self, context: AbstractContext,
                                  command_queue: AbstractCommandQueue) -> None:
        self._test_copy_region(
            context, command_queue, (10, 12), None, (7, 8), None,
            np.s_[-4, -3:-1], np.s_[-7, -8:-4:2])

    def test_copy_region_errors(self, context: AbstractContext,
                                command_queue: AbstractCommandQueue) -> None:
        # Too many axes
        with pytest.raises(IndexError):
            self._test_copy_region(
                context, command_queue, (10,), None, (10,), None,
                np.s_[3, 4], np.s_[5, 6])
        # Out-of-range single index
        with pytest.raises(IndexError):
            self._test_copy_region(
                context, command_queue, (10,), None, (10,), None,
                np.s_[5], np.s_[10])
        # Out-of-range slice
        with pytest.raises(IndexError):
            self._test_copy_region(
                context, command_queue, (10,), None, (10,), None,
                np.s_[10:12], np.s_[8:10])
        # Empty slice
        with pytest.raises(IndexError):
            self._test_copy_region(
                context, command_queue, (10,), None, (10,), None,
                np.s_[2:2], np.s_[3:3])
        # Negative stride
        with pytest.raises(IndexError):
            self._test_copy_region(
                context, command_queue, (10,), None, (10,), None,
                np.s_[3:0:-1], np.s_[4:1:-1])

    def test_zero(self, array: DeviceArray, context: AbstractContext,
                  command_queue: AbstractCommandQueue) -> None:
        ary = np.random.randint(0x12345678, 0x23456789, array.shape).astype(np.int32)
        array.set(command_queue, ary)
        before = array.get(command_queue)
        array.zero(command_queue)
        after = array.get(command_queue)
        np.testing.assert_equal(ary, before)
        assert np.max(after) == 0

    def _allocate_raw(self, context: AbstractContext, n_bytes: int) -> Any:
        return context.allocate_raw(n_bytes)

    def test_raw(self, shape: Tuple[int, ...], padded_shape: Tuple[int, ...],
                 context: AbstractContext) -> None:
        raw = self._allocate_raw(context, 2048)
        ary = self.cls(
            context=context,
            shape=shape,
            dtype=np.int32,
            padded_shape=padded_shape,
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

    @pytest.fixture(autouse=True)
    def setup(self, context: AbstractContext):
        context._force_pinned_amd = True        # type: ignore
        yield
        context._force_pinned_amd = False       # type: ignore


@pytest.mark.cuda_only
class TestSVMArrayHost(TestHostArray):
    """Test SVMArray using the HostArray tests."""

    cls = SVMArray

    # Function is given a different name to avoid mypy complaining about the
    # signature being different.
    @pytest.fixture(name='factory')
    def factory_override(self, context: AbstractContext) -> \
            Callable[[Tuple[int, ...], DTypeLike, Tuple[int, ...]], HostArray]:
        def allocate(shape: Tuple[int, ...], dtype: DTypeLike,
                     padded_shape: Tuple[int, ...]) -> HostArray:
            return self.cls(context, shape, dtype, padded_shape)
        return allocate


@pytest.mark.cuda_only
class TestSVMArray(TestDeviceArray):
    """Tests SVMArray using the DeviceArray tests, plus some new ones."""

    cls = SVMArray

    def _allocate_raw(self, context: AbstractContext, n_bytes: int) -> object:
        return context.allocate_svm_raw(n_bytes)

    def test_coherence(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
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
        command_queue.enqueue_kernel(kernel, [ary.buffer], (128,), (64,))
        command_queue.finish()
        np.testing.assert_equal(np.arange(0, 369, step=3, dtype=np.uint32), ary)


@pytest.mark.cuda_only
class TestSVMAllocator:
    def test_allocate(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        allocator = accel.SVMAllocator(context)
        ary = allocator.allocate((12, 34), np.int32, (13, 35))
        assert isinstance(ary, accel.SVMArray)
        assert ary.shape == (12, 34)
        assert ary.dtype == np.int32
        assert ary.padded_shape == (13, 35)
        ary.fill(1)  # Just to make sure it doesn't crash

    def test_allocate_raw(self, context: AbstractContext,
                          command_queue: AbstractCommandQueue) -> None:
        allocator = accel.SVMAllocator(context)
        raw = allocator.allocate_raw(13 * 35 * 4)
        ary = allocator.allocate((12, 34), np.int32, (13, 35), raw=raw)
        assert isinstance(ary, accel.SVMArray)
        assert ary.shape == (12, 34)
        assert ary.dtype == np.int32
        assert ary.padded_shape == (13, 35)
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
        assert dim.min_padded_size == 20
        dim = accel.Dimension(20, min_padded_round=5)
        assert dim.min_padded_size == 20

    def test_add_align_dtype(self) -> None:
        dim = accel.Dimension(20, alignment=8)
        assert dim.alignment == 8
        dim.add_align_dtype(np.complex64)
        assert dim.alignment_hint == accel.Dimension.ALIGN_BYTES // 8
        dim.add_align_dtype(np.uint8)
        assert dim.alignment_hint == accel.Dimension.ALIGN_BYTES
        dim.add_align_dtype(np.float32)
        assert dim.alignment_hint == accel.Dimension.ALIGN_BYTES

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
        assert dim1.size == dim2.size
        assert dim1.min_padded_size == dim2.min_padded_size
        assert dim1.alignment == dim2.alignment
        assert dim1.alignment_hint == dim2.alignment_hint
        assert dim1.exact == dim2.exact

    def test_link(self) -> None:
        """Linking several dimensions together gives them all the right properties."""
        dim1 = accel.Dimension(22, min_padded_size=28, alignment=4)
        dim2 = accel.Dimension(22, min_padded_size=24, alignment=8, align_dtype=np.int32)
        dim3 = accel.Dimension(22, min_padded_size=22, align_dtype=np.uint16)
        dim1.link(dim2)
        dim1.link(dim3)
        self.assert_dimensions_equal(dim1, dim2)
        self.assert_dimensions_equal(dim2, dim3)
        assert dim1.size == 22
        assert dim1.min_padded_size == 28
        assert dim1.alignment == 8
        assert dim1.alignment_hint == accel.Dimension.ALIGN_BYTES // 2
        assert dim1.exact is False

    def test_link_bad_size(self) -> None:
        """Linking dimensions with different sizes fails."""
        dim1 = accel.Dimension(22, min_padded_size=28, alignment=4)
        dim2 = accel.Dimension(23, min_padded_size=24, alignment=8, align_dtype=np.int32)
        with pytest.raises(ValueError):
            dim1.link(dim2)
        assert dim1._root() is not dim2._root()

    def test_link_bad_exact(self) -> None:
        """Linking dimensions into an unsatisfiable requirement fails."""
        dim1 = accel.Dimension(22, exact=True)
        dim2 = accel.Dimension(22, min_padded_size=28)
        dim3 = accel.Dimension(22, alignment=4)
        with pytest.raises(ValueError):
            dim1.link(dim2)
        with pytest.raises(ValueError):
            dim1.link(dim3)
        # Check that linking didn't happen anyway
        assert dim1._root() is not dim2._root()
        assert dim1._root() is not dim3._root()

    def test_required_padded_size(self) -> None:
        """The padded size is computed correctly."""
        dim = accel.Dimension(30, 7, alignment=4)
        assert dim.required_padded_size() == 36

    def test_required_padded_size_dtype(self) -> None:
        """The padded size is computed correctly when an alignment hint is given."""
        dim = accel.Dimension(1100, 200, align_dtype=np.float32)
        assert dim.required_padded_size() == 1216

    def test_required_padded_size_exact(self) -> None:
        """The padded size is computed correctly for exact dimensions."""
        dim = accel.Dimension(1100, align_dtype=np.float32, exact=True)
        assert dim.required_padded_size() == 1100

    def test_required_padded_size_small(self) -> None:
        """The alignment hint is ignored for small sizes."""
        dim = accel.Dimension(18, alignment=8, align_dtype=np.uint8)
        assert dim.required_padded_size() == 24


class TestIOSlot:
    """Tests for :class:`katsdpsigproc.accel.IOSlot`."""

    @pytest.mark.parametrize("bind", [True, False])
    @mock.patch('katsdpsigproc.accel.DeviceArray', spec=True)
    def test_allocate(self, DeviceArray: mock.Mock, bind: bool) -> None:
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
        ret = slot.allocate(accel.DeviceAllocator(mock.sentinel.context), bind=bind)
        # Validation
        assert ret == ary
        if bind:
            assert slot.buffer == ary
        else:
            assert slot.buffer is None
        DeviceArray.assert_called_once_with(
            mock.sentinel.context, shape, dtype, padded_shape, None)
        # Check that the inner dimension had a type hint set
        assert dims[1].alignment_hint == accel.Dimension.ALIGN_BYTES

    @mock.patch('katsdpsigproc.accel.HostArray', spec=True)
    def test_allocate_host(self, HostArray: mock.Mock) -> None:
        dims = (
            accel.Dimension(50, min_padded_size=60, alignment=8),
            accel.Dimension(30, min_padded_size=50, alignment=4)
        )
        shape = (50, 30)
        padded_shape = (64, 52)
        dtype = np.int16
        slot = accel.IOSlot(dims, dtype)
        HostArray.return_value = mock.sentinel.host_array
        ret = slot.allocate_host(mock.sentinel.context)
        assert ret is mock.sentinel.host_array
        HostArray.assert_called_with(shape, dtype, padded_shape, context=mock.sentinel.context)

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
        assert slot.buffer == ary
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

        with pytest.raises(ValueError):
            # Wrong shape
            ary.shape = (5, 4)
            slot.bind(ary)
        with pytest.raises(ValueError):
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

        with pytest.raises(TypeError):
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
        with pytest.raises(ValueError):
            # Bigger than needed, but not an exact match - must fail
            slot.bind(ary)

    def test_required_bytes(self) -> None:
        slot = accel.IOSlot(
            (accel.Dimension(27, alignment=4), accel.Dimension(33, alignment=32)),
            np.float32)
        assert slot.required_bytes() == 4 * 28 * 64

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

    def setup_method(self) -> None:
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
        with pytest.raises(ValueError):
            accel.CompoundIOSlot([])

    def test_validate_shape(self) -> None:
        """:class:`.CompoundIOSlot` must check that children have consistent shapes."""
        slot1 = accel.IOSlot((5, 3), np.float32)
        slot2 = accel.IOSlot((5, 4), np.float32)
        with pytest.raises(ValueError):
            accel.CompoundIOSlot([slot1, slot2])

    def test_validate_dtype(self) -> None:
        """:class:`.CompoundIOSlot` must check that children have consistent data types."""
        slot1 = accel.IOSlot((5, 3), np.float32)
        slot2 = accel.IOSlot((5, 3), np.int32)
        with pytest.raises(TypeError):
            accel.CompoundIOSlot([slot1, slot2])

    def test_attributes(self) -> None:
        """:class:`.CompoundIOSlot` must correctly combine attributes."""
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        assert slot.shape == (13, 7, 22)
        assert slot.dtype == np.float32
        assert self.dims1[0].min_padded_size == 17
        assert self.dims1[1].min_padded_size == 10
        assert self.dims1[2].min_padded_size == 25
        assert self.dims1[0].alignment == 4
        assert self.dims1[1].alignment == 8
        assert self.dims1[2].alignment == 4
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
        assert slot.buffer == ary
        assert self.slot1.buffer == ary
        assert self.slot2.buffer == ary

    def test_bind_fail(self) -> None:
        """CompoundIOSlot.bind must not have side effects if the array is \
        not valid for all children."""
        ary = mock.sentinel.ary
        ary.shape = (13, 7, 22)
        ary.dtype = np.float32
        ary.padded_shape = (17, 8, 28)
        # Check that the array has the properties we want for the test
        self.slot1.validate(ary)
        with pytest.raises(ValueError):
            self.slot2.validate(ary)
        # The actual test
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        with pytest.raises(ValueError):
            slot.bind(ary)
        assert slot.buffer is None
        assert self.slot1.buffer is None
        assert self.slot2.buffer is None


class TestAliasIOSlot:
    """Tests for :class:`katsdpsigproc.accel.AliasIOSlot`."""

    def setup_method(self) -> None:
        self.slot1 = accel.IOSlot((3, 7), np.float32)
        self.slot2 = accel.IOSlot((5, 3), np.complex64)

    def test_required_bytes(self) -> None:
        slot = accel.AliasIOSlot([self.slot1, self.slot2])
        assert slot.required_bytes() == 120

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

    @mock.patch('katsdpsigproc.accel.HostArray', spec=True)
    def test_allocate_host(self, HostArray: mock.Mock) -> None:
        HostArray.return_value = mock.sentinel.host_array
        slot = accel.AliasIOSlot([self.slot1, self.slot2])
        ret = slot.allocate_host(mock.sentinel.context)
        assert ret is mock.sentinel.host_array
        HostArray.assert_called_with((120,), np.uint8, context=mock.sentinel.context)


class TestVisualizeOperation:
    """Tests for :class:`katsdpsigproc.accel.visualize_operation`.

    This is just a basic smoke test to ensure that it runs without crashing.
    A real test requires a human to sanity-check the visual output.
    """

    class Inner(accel.Operation):
        def _run(self) -> None:
            pass

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test(self, context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
        inner = TestVisualizeOperation.Inner(command_queue)
        inner.slots['foo'] = accel.IOSlot((3, 4), np.float32)
        inner.slots['bar'] = accel.IOSlot((4, 3), np.float32)

        mid = accel.OperationSequence(
            command_queue,
            [('inner', inner)],
            {'inner_foo': ['inner:foo']})

        outer = accel.OperationSequence(
            command_queue,
            [('mid', mid)],
            {'mid_foo': ['mid:inner_foo']},
            {'scratch': ['mid:inner:bar']})

        accel.visualize_operation(outer, os.path.join(self.tmpdir, 'test_visualize.gv'))
