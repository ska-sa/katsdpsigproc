import functools
import sys
import numpy as np
from mako.template import Template
from nose.tools import assert_equal
from nose.plugins.skip import SkipTest
try:
    from .. import accel
    have_accel = True
except ImportError:
    have_accel = False
if accel.have_cuda:
    import pycuda
if accel.have_opencl:
    import pyopencl

test_context = None
test_command_queue = None
if have_accel:
    from ..accel import Array, DeviceArray, LinenoLexer, Transpose
    test_context = accel.create_some_context(False)
    test_command_queue = test_context.create_command_queue()
    print >>sys.stderr, "Testing on {0} ({1})".format(
            test_context.device.name, test_context.device.platform_name)

def device_test(test):
    """Decorator that causes a test to be skipped if a compute device is not available"""
    def wrapper(*args, **kw):
        if not test_context:
            raise SkipTest('CUDA/OpenCL not found')
        return test(*args, **kw)
    return functools.update_wrapper(wrapper, test)

# Prevent nose from treating it as a test
device_test.__test__ = False

class TestLinenoLexer(object):
    @device_test
    def test_escape_filename(self):
        assert_equal(
                r'"abc\"def\\ghi"',
                LinenoLexer._escape_filename(r'abc"def\ghi'))

    @device_test
    def test_render(self):
        source = "line 1\nline 2\nline 3"
        out = Template(source, lexer_cls=LinenoLexer).render()
        assert_equal("#line 1 \nline 1\n#line 2 \nline 2\n#line 3 \nline 3\n", out)

class TestArray(object):
    @device_test
    def setup(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = Array(
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(Array)
        self.sliced = self.constructed[2:4, 3:5]

    @device_test
    def test_safe(self):
        assert Array.safe(self.constructed)
        assert not Array.safe(self.view)
        assert not Array.safe(self.sliced)
        assert not Array.safe(np.zeros(self.shape))

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
        assert Array.safe(ary)

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

class TestTranspose(object):
    def test_transpose(self):
        yield self.check_transpose, 4, 5
        yield self.check_transpose, 53, 7
        yield self.check_transpose, 53, 81
        yield self.check_transpose, 32, 64

    @device_test
    def setup(self):
        self.transpose = Transpose(test_command_queue, 'float')

    @device_test
    def check_transpose(self, R, C):
        ary = np.random.randn(R, C).astype(np.float32)
        src = DeviceArray(test_context, (R, C), dtype=np.float32, padded_shape=(R + 1, C + 4))
        dest = DeviceArray(test_context, (C, R), dtype=np.float32, padded_shape=(C + 2, R + 3))
        src.set_async(test_command_queue, ary)
        self.transpose(dest, src)
        out = dest.get(test_command_queue)
        np.testing.assert_equal(ary.T, out)
