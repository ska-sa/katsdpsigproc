#!/usr/bin/env python
import sys
import numpy as np
from decorator import decorator
from mako.template import Template
from nose.tools import assert_equal
from .test_tune import assert_raises
from nose.plugins.skip import SkipTest
import mock
from .. import accel
from ..accel import HostArray, DeviceArray, LinenoLexer
if accel.have_cuda:
    import pycuda
if accel.have_opencl:
    import pyopencl

test_context = None
test_command_queue = None
try:
    test_context = accel.create_some_context(False)
    test_command_queue = test_context.create_command_queue()
    print >>sys.stderr, "Testing on {0} ({1})".format(
            test_context.device.name, test_context.device.platform_name)
except RuntimeError:
    pass  # No devices available

@decorator
def device_test(test, *args, **kw):
    """Decorator that causes a test to be skipped if a compute device is not available"""
    if not test_context:
        raise SkipTest('CUDA/OpenCL not found')
    return test(*args, **kw)

# Prevent nose from treating it as a test
device_test.__test__ = False

class TestLinenoLexer(object):
    def test_escape_filename(self):
        assert_equal(
                r'"abc\"def\\ghi"',
                LinenoLexer._escape_filename(r'abc"def\ghi'))

    def test_render(self):
        source = "line 1\nline 2\nline 3"
        out = Template(source, lexer_cls=LinenoLexer).render()
        assert_equal("#line 1 \nline 1\n#line 2 \nline 2\n#line 3 \nline 3\n", out)

class TestArray(object):
    def setup(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = HostArray(
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(HostArray)
        self.sliced = self.constructed[2:4, 3:5]

    def test_safe(self):
        assert HostArray.safe(self.constructed)
        assert not HostArray.safe(self.view)
        assert not HostArray.safe(self.sliced)
        assert not HostArray.safe(np.zeros(self.shape))

class TestDeviceArray(object):
    @device_test
    def setup(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = DeviceArray(
                context=test_context,
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)

    @device_test
    def test_strides(self):
        assert_equal(self.strides, self.array.strides)

    @device_test
    def test_empty_like(self):
        ary = self.array.empty_like()
        assert_equal(self.shape, ary.shape)
        assert_equal(self.strides, ary.strides)
        assert HostArray.safe(ary)

    @device_test
    def test_set_get(self):
        ary = np.random.randint(0, 100, self.shape).astype(np.int32)
        self.array.set(test_command_queue, ary)
        # Read back results, check that it matches
        buf = np.zeros(self.padded_shape, dtype=np.int32)
        if hasattr(self.array.buffer, 'gpudata'):
            # It's CUDA
            with test_context:
                pycuda.driver.memcpy_dtoh(buf, self.array.buffer.gpudata)
        else:
            pyopencl.enqueue_copy(
                    test_command_queue._pyopencl_command_queue,
                    buf, self.array.buffer.data)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = self.array.get(test_command_queue)
        np.testing.assert_equal(ary, buf)

    @device_test
    def test_raw(self):
        raw = test_context.allocate_raw(2048)
        ary = DeviceArray(
                context=test_context,
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape,
                raw=raw)
        assert ary.buffer is raw

class TestIOSlot(object):
    """Tests for :class:`katsdpsigproc.accel.IOSlot`"""
    @mock.patch('katsdpsigproc.accel.DeviceArray', spec=True)
    def test_allocate(self, DeviceArray):
        """IOSlot.allocate must correctly apply alignment an allocate a buffer"""
        shape = (50, 30)
        min_padded_shape = (60, 50)
        alignment = (8, 4)
        padded_shape = (64, 52)
        dtype = np.dtype(np.float32)
        # Create the device array that will be created. We need to populate it
        # with some attributes to allow validation to pass
        ary = mock.Mock()
        ary.dtype = dtype
        ary.shape = shape
        ary.padded_shape = padded_shape
        # Set the mocked DeviceArray class to return this array
        DeviceArray.return_value = ary
        # Run the system under test
        slot = accel.IOSlot(shape, dtype, min_padded_shape=min_padded_shape, alignment=alignment)
        ret = slot.allocate(mock.sentinel.context)
        # Validation
        assert_equal(ary, ret)
        assert_equal(ary, slot.buffer)
        DeviceArray.assert_called_once_with(
                mock.sentinel.context, shape, dtype, padded_shape, raw=None)

    @mock.patch('katsdpsigproc.accel.DeviceArray', spec=True)
    def test_allocate_raw(self, DeviceArray):
        """Test IOSlot.allocate with a raw parameter"""
        shape = (50, 30)
        min_padded_shape = (60, 50)
        alignment = (8, 4)
        padded_shape = (64, 52)
        dtype = np.dtype(np.float32)
        raw = mock.sentinel.raw
        # Create the device array that will be created. We need to populate it
        # with some attributes to allow validation to pass
        ary = mock.Mock()
        ary.dtype = dtype
        ary.shape = shape
        ary.padded_shape = padded_shape
        ary.raw = raw
        # Set the mocked DeviceArray class to return this array
        DeviceArray.return_value = ary
        # Run the system under test
        slot = accel.IOSlot(shape, dtype, min_padded_shape=min_padded_shape, alignment=alignment)
        slot.allocate(mock.sentinel.context, raw)
        # Validation
        assert_equal(ary, slot.buffer)
        DeviceArray.assert_called_once_with(
                mock.sentinel.context, shape, dtype, padded_shape, raw=raw)

    def test_validate_shape(self):
        """IOSlot.validate must check that the shape matches"""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = (10, 10)
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

    def test_validate_dtype(self):
        """IOSlot.validate must check that the dtype matches"""
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

    def test_validate_padded_shape(self):
        """IOSlot.validate must check that the padded shape is large enough"""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = (8, 6)
        slot = accel.IOSlot((5, 3), np.float32, min_padded_shape=(8, 6))
        slot.bind(ary)  # Should pass

        ary.padded_shape = (10, 11)
        slot.bind(ary)  # Bigger than needed - should pass

        with assert_raises(ValueError):
            ary.padded_shape = (7, 7)
            slot.bind(ary)

    def test_validate_alignment(self):
        """IOSlot.validate must check that the alignment is correct"""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (27, 33)
        ary.padded_shape = (32, 40)
        slot = accel.IOSlot((27, 33), np.float32, alignment=(4, 8))
        slot.bind(ary)  # Should pass

        with assert_raises(ValueError):
            ary.padded_shape = (32, 44)
            slot.bind(ary)

    def test_required_bytes(self):
        slot = accel.IOSlot((27, 33), np.float32, alignment=(4, 8))
        assert_equal(4 * 28 * 40, slot.required_bytes())

    def test_bind_none(self):
        """IOSlot.bind must accept `None`"""
        ary = mock.sentinel.ary
        ary.dtype = np.float32
        ary.shape = (5, 3)
        ary.padded_shape = ary.shape
        slot = accel.IOSlot((5, 3), np.float32)
        slot.bind(ary)
        assert slot.buffer is ary
        slot.bind(None)
        assert slot.buffer is None

    def test_min_padded_round(self):
        """IOSlot constructor must compute min padded shape correctly"""
        slot = accel.IOSlot((17, 23, 5), np.float32, (3, 5, 5))
        assert_equal((18, 25, 5), slot.min_padded_shape)

class TestCompoundIOSlot(object):
    """Tests for :class:`katsdpsigproc.accel.CompoundIOSlot`"""

    def setup(self):
        self.slot1 = accel.IOSlot((13, 7, 22), np.float32, min_padded_shape=(17, 8, 25), alignment=(1, 8, 4))
        self.slot2 = accel.IOSlot((13, 7, 22), np.float32, min_padded_shape=(14, 10, 22), alignment=(4, 4, 1))

    def test_check_empty(self):
        """CompoundIOSlot constructor must reject empty list"""
        with assert_raises(ValueError):
            accel.CompoundIOSlot([])

    def test_validate_shape(self):
        """CompoundIOSlot must check that children have consistent shapes"""
        slot1 = accel.IOSlot((5, 3), np.float32)
        slot2 = accel.IOSlot((5, 4), np.float32)
        with assert_raises(ValueError):
            accel.CompoundIOSlot([slot1, slot2])

    def test_validate_dtype(self):
        """CompoundIOSlot must check that children have consistent data types"""
        slot1 = accel.IOSlot((5, 3), np.float32)
        slot2 = accel.IOSlot((5, 3), np.int32)
        with assert_raises(TypeError):
            accel.CompoundIOSlot([slot1, slot2])

    def test_attributes(self):
        """CompoundIOSlot must correctly combine attributes"""
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        assert_equal((13, 7, 22), slot.shape)
        assert_equal(np.float32, slot.dtype)
        assert_equal((17, 10, 25), slot.min_padded_shape)
        assert_equal((4, 8, 4), slot.alignment)

    def test_bind(self):
        """CompoundIOSlot.bind must bind children"""
        ary = mock.sentinel.ary
        ary.shape = (13, 7, 22)
        ary.dtype = np.float32
        ary.padded_shape = (20, 16, 28)
        slot = accel.CompoundIOSlot([self.slot1, self.slot2])
        slot.bind(ary)
        assert_equal(ary, slot.buffer)
        assert_equal(ary, self.slot1.buffer)
        assert_equal(ary, self.slot2.buffer)

    def test_bind_fail(self):
        """CompoundIOSlot.bind must not have side effects if the array is not
        valid for all children.
        """
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

class TestAliasIOSlot(object):
    """Tests for :class:`katsdpsigproc.accel.AliasIOSlot`"""

    def setup(self):
        self.slot1 = accel.IOSlot((3, 7), np.float32)
        self.slot2 = accel.IOSlot((5, 3), np.complex64)

    def test_required_bytes(self):
        slot = accel.AliasIOSlot([self.slot1, self.slot2])
        assert_equal(120, slot.required_bytes())

    def test_allocate(self):
        # Set up mocks
        raw = mock.sentinel.raw
        context = mock.NonCallableMock()
        context.allocate_raw = mock.Mock(return_value=raw)
        # Run the test
        slot = accel.AliasIOSlot([self.slot1, self.slot2])
        ret = slot.allocate(context)
        # Validation
        context.allocate_raw.assert_called_once_with(120)
        assert ret is raw
        assert slot.raw is raw
        assert self.slot1.buffer is not None
        assert self.slot2.buffer is not None
