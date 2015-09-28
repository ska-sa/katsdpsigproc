#!/usr/bin/env python
import sys
import numpy as np
from decorator import decorator
import functools
from mako.template import Template
from nose.tools import assert_equal
from .test_tune import assert_raises
from nose.plugins.skip import SkipTest
import mock
from .. import accel, tune
from ..accel import HostArray, DeviceArray, SVMArray, LinenoLexer
if accel.have_cuda:
    import pycuda
if accel.have_opencl:
    import pyopencl

_test_context = None
_test_command_queue = None
_test_initialized = False

def device_test(test):
    """Decorator that causes a test to be skipped if a compute device is not
    available, and which disables autotuning. If autotuning is desired, use
    :func:`force_autotune` inside (hence, afterwards on the decorator list)
    this one."""
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        global _test_initialized, _test_context, _test_command_queue
        if not _test_initialized:
            try:
                _test_context = accel.create_some_context(False)
                _test_command_queue = _test_context.create_command_queue()
                print >>sys.stderr, "Testing on {0} ({1})".format(
                        _test_context.device.name, _test_context.device.platform_name)
            except RuntimeError:
                pass  # No devices available
            _test_initialized = True

        if not _test_context:
            raise SkipTest('CUDA/OpenCL not found')
        with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner):
            args += (_test_context, _test_command_queue)
            return test(*args, **kwargs)
    return wrapper

def cuda_test(test):
    """Decorator that causes a test to be skipped if the device is not a CUDA
    device. Put this *after* :meth:`device_test`."""
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        global _test_context
        if not _test_context.device.is_cuda:
            raise SkipTest('Device is not a CUDA device')
        return test(*args, **kwargs)
    return wrapper

@decorator
def force_autotune(test, *args, **kw):
    """Decorator that disables autotuning for a test. Instead, the test
    value specified for each class is returned."""
    with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.force_autotuner):
        return test(*args, **kw)

# Prevent nose from treating it as a test
device_test.__test__ = False
cuda_test.__test__ = False

class TestLinenoLexer(object):
    def test_escape_filename(self):
        assert_equal(
                r'"abc\"def\\ghi"',
                LinenoLexer._escape_filename(r'abc"def\ghi'))

    def test_render(self):
        source = "line 1\nline 2\nline 3"
        out = Template(source, lexer_cls=LinenoLexer).render()
        assert_equal("""#line 1 "<string>"\nline 1\n#line 2 "<string>"\nline 2\n#line 3 "<string>"\nline 3\n""", out)

class TestHostArray(object):
    cls = HostArray

    def allocate(self, shape, dtype, padded_shape):
        return self.cls(shape, dtype, padded_shape)

    def setup(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = self.allocate(self.shape, np.int32, self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(self.cls)
        self.sliced = self.constructed[2:4, 3:5]

    def test_safe(self):
        assert self.cls.safe(self.constructed)
        assert not self.cls.safe(self.view)
        assert not self.cls.safe(self.sliced)
        assert not self.cls.safe(np.zeros(self.shape))

class TestDeviceArray(object):
    cls = DeviceArray

    @device_test
    def setup(self, context, queue):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = self.cls(
                context=context,
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)

    @device_test
    def test_strides(self, context, queue):
        assert_equal(self.strides, self.array.strides)

    @device_test
    def test_empty_like(self, context, queue):
        ary = self.array.empty_like()
        assert_equal(self.shape, ary.shape)
        assert_equal(self.strides, ary.strides)
        assert HostArray.safe(ary)

    @device_test
    def test_set_get(self, context, queue):
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
            pyopencl.enqueue_copy(
                    queue._pyopencl_command_queue,
                    buf, self.array.buffer.data)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = self.array.get(queue)
        np.testing.assert_equal(ary, buf)

    def _allocate_raw(self, context, n_bytes):
        return context.allocate_raw(n_bytes)

    @device_test
    def test_raw(self, context, queue):
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
                actual_raw = ary.base.base
                raw = raw._wrapped
            else:
                actual_raw = ary.buffer.gpudata
        except AttributeError:
            # OpenCL
            actual_raw = ary.buffer.data
        assert actual_raw is raw

class TestSVMArrayHost(TestHostArray):
    """Tests SVMArray using the HostArray tests"""
    cls = SVMArray

    def allocate(self, shape, dtype, padded_shape):
        return self.cls(self.context, shape, dtype, padded_shape)

    @device_test
    @cuda_test
    def setup(self, context, queue):
        self.context = context
        super(TestSVMArrayHost, self).setup()

class TestSVMArray(TestDeviceArray):
    """Tests SVMArray using the DeviceArray tests, plus some new ones"""
    cls = SVMArray

    def _allocate_raw(self, context, n_bytes):
        return context.allocate_svm_raw(n_bytes)

    @device_test
    @cuda_test
    def setup(self, context, queue):
        super(TestSVMArray, self).setup()

    @device_test
    @cuda_test
    def test_coherence(self, context, queue):
        """Check that runtime correct transfers data between the host and
        device views."""
        source = """\
            <%include file="/port.mako"/>
            KERNEL void triple(unsigned int *data)
            {
                data[get_global_id(0)] *= 3;
            }"""
        program = accel.build(context, None, source=source)
        kernel = program.get_kernel("triple")
        ary = SVMArray(context, (123,), np.uint32, (128,))
        ary[:] = np.arange(123, dtype=np.uint32)
        queue.enqueue_kernel(kernel, [ary.buffer], (128,), (64,))
        queue.finish()
        np.testing.assert_equal(np.arange(369, step=3, dtype=np.uint32), ary)

class TestSVMAllocator(object):
    @device_test
    @cuda_test
    def test_allocate(self, context, queue):
        allocator = accel.SVMAllocator(context)
        ary = allocator.allocate((12, 34), np.int32, (13, 35))
        assert isinstance(ary, accel.SVMArray)
        assert_equal((12, 34), ary.shape)
        assert_equal(np.int32, ary.dtype)
        assert_equal((13, 35), ary.padded_shape)
        ary.fill(1)  # Just to make sure it doesn't crash

    @device_test
    @cuda_test
    def test_allocate_raw(self, context, queue):
        allocator = accel.SVMAllocator(context)
        raw = allocator.allocate_raw(13 * 35 * 4)
        ary = allocator.allocate((12, 34), np.int32, (13, 35), raw=raw)
        assert isinstance(ary, accel.SVMArray)
        assert_equal((12, 34), ary.shape)
        assert_equal(np.int32, ary.dtype)
        assert_equal((13, 35), ary.padded_shape)
        ary.fill(1)  # Just to make sure it doesn't crash


class TestDimension(object):
    """Tests for :class:`katsdpsigproc.accel.Dimension`"""
    def test_is_power2(self):
        assert accel.Dimension._is_power2(1)
        assert accel.Dimension._is_power2(2)
        assert accel.Dimension._is_power2(32)
        assert not accel.Dimension._is_power2(-1)
        assert not accel.Dimension._is_power2(0)
        assert not accel.Dimension._is_power2(3)
        assert not accel.Dimension._is_power2(5)

    def test_min_padded_round(self):
        """Constructor computes min padded size correctly"""
        dim = accel.Dimension(17, min_padded_round=4)
        assert_equal(20, dim.min_padded_size)
        dim = accel.Dimension(20, min_padded_round=5)
        assert_equal(20, dim.min_padded_size)

    def test_add_align_dtype(self):
        dim = accel.Dimension(20, alignment=8)
        assert_equal(8, dim.alignment)
        dim.add_align_dtype(np.complex64)
        assert_equal(accel.Dimension.ALIGN_BYTES // 8, dim.alignment_hint)
        dim.add_align_dtype(np.uint8)
        assert_equal(accel.Dimension.ALIGN_BYTES, dim.alignment_hint)
        dim.add_align_dtype(np.float32)
        assert_equal(accel.Dimension.ALIGN_BYTES, dim.alignment_hint)

    def test_valid(self):
        """Test `valid` method on non-exact dimension"""
        dim = accel.Dimension(17, min_padded_round=8, alignment=4)
        assert dim.valid(24)
        assert not dim.valid(20)
        assert dim.valid(28)
        assert not dim.valid(30)

    def test_valid_exact(self):
        """Test `valid` method on exact dimension"""
        dim = accel.Dimension(20, alignment=4, exact=True)
        assert dim.valid(20)
        assert not dim.valid(24)

    @classmethod
    def assert_dimensions_equal(cls, dim1, dim2):
        assert_equal(dim1.size, dim2.size)
        assert_equal(dim1.min_padded_size, dim2.min_padded_size)
        assert_equal(dim1.alignment, dim2.alignment)
        assert_equal(dim1.alignment_hint, dim2.alignment_hint)
        assert_equal(dim1.exact, dim2.exact)

    def test_link(self):
        """Linking several dimensions together gives them all the right
        properties."""
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

    def test_link_bad_size(self):
        """Linking dimensions with different sizes fails"""
        dim1 = accel.Dimension(22, min_padded_size=28, alignment=4)
        dim2 = accel.Dimension(23, min_padded_size=24, alignment=8, align_dtype=np.int32)
        with assert_raises(ValueError):
            dim1.link(dim2)
        assert dim1._root() is not dim2._root()

    def test_link_bad_exact(self):
        """Linking dimensions into an unsatisfiable requirement fails"""
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

    def test_required_padded_size(self):
        """The padded size is computed correctly"""
        dim = accel.Dimension(30, 7, alignment=4)
        assert_equal(36, dim.required_padded_size())

    def test_required_padded_size_dtype(self):
        """The padded size is computed correctly when an alignment hint is given"""
        dim = accel.Dimension(1100, 200, align_dtype=np.float32)
        assert_equal(1216, dim.required_padded_size())

    def test_required_padded_size_exact(self):
        """The padded size is computed correctly for exact dimensions"""
        dim = accel.Dimension(1100, align_dtype=np.float32, exact=True)
        assert_equal(1100, dim.required_padded_size())

    def test_required_padded_size_small(self):
        """The alignment hint is ignored for small sizes"""
        dim = accel.Dimension(18, alignment=8, align_dtype=np.uint8)
        assert_equal(24, dim.required_padded_size())

class TestIOSlot(object):
    """Tests for :class:`katsdpsigproc.accel.IOSlot`"""
    @mock.patch('katsdpsigproc.accel.DeviceArray', spec=True)
    def test_allocate(self, DeviceArray):
        """IOSlot.allocate must correctly allocate a buffer"""
        dims = [
            accel.Dimension(50, min_padded_size=60, alignment=8),
            accel.Dimension(30, min_padded_size=50, alignment=4)
        ]
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
    def test_allocate_raw(self, DeviceArray):
        """Test IOSlot.allocate with a raw parameter"""
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

    def test_validate_shape(self):
        """IOSlot.validate must check that the shape matches"""
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
        """IOSlot.validate must check that the padded shape is valid"""
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

    def test_required_bytes(self):
        slot = accel.IOSlot(
                (accel.Dimension(27, alignment=4), accel.Dimension(33, alignment=32)),
                np.float32)
        assert_equal(4 * 28 * 64, slot.required_bytes())

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

class TestCompoundIOSlot(object):
    """Tests for :class:`katsdpsigproc.accel.CompoundIOSlot`"""

    def setup(self):
        self.dims1 = [
            accel.Dimension(13, min_padded_size=17, alignment=1),
            accel.Dimension(7,  min_padded_size=8,  alignment=8),
            accel.Dimension(22, min_padded_size=25, alignment=4)
        ]
        self.dims2 = [
            accel.Dimension(13, min_padded_size=14, alignment=4),
            accel.Dimension(7,  min_padded_size=10, alignment=4),
            accel.Dimension(22, min_padded_size=22, alignment=1)
        ]
        self.slot1 = accel.IOSlot(self.dims1, np.float32)
        self.slot2 = accel.IOSlot(self.dims2, np.float32)

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
        assert_equal(17, self.dims1[0].min_padded_size)
        assert_equal(10, self.dims1[1].min_padded_size)
        assert_equal(25, self.dims1[2].min_padded_size)
        assert_equal(4, self.dims1[0].alignment)
        assert_equal(8, self.dims1[1].alignment)
        assert_equal(4, self.dims1[2].alignment)
        for x, y in zip(self.dims1, self.dims2):
            TestDimension.assert_dimensions_equal(x, y)

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
        ret = slot.allocate(accel.DeviceAllocator(context))
        # Validation
        context.allocate_raw.assert_called_once_with(120)
        assert ret is raw
        assert slot.raw is raw
        assert self.slot1.buffer is not None
        assert self.slot2.buffer is not None
