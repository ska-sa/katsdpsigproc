import functools
import numpy as np
from mako.template import Template
from nose.tools import assert_equal
from nose.plugins.skip import SkipTest
try:
    import pycuda.autoinit
    have_cuda = True
except ImportError:
    have_cuda = False

test_context = None
test_command_queue = None
if have_cuda:
    from ..accel import Array, DeviceArray, LinenoLexer, Transpose
    from .. import cuda
    test_context = cuda.Context(pycuda.autoinit.context)
    test_command_queue = cuda.CommandQueue(test_context, None)

def cuda_test(test):
    """Decorator that causes a test to be skipped if CUDA is not available"""
    def wrapper(*args, **kw):
        if not have_cuda:
            raise SkipTest('CUDA not found')
        return test(*args, **kw)
    return functools.update_wrapper(wrapper, test)

# Prevent nose from treating it as a test
cuda_test.__test__ = False

class TestLinenoLexer(object):
    @cuda_test
    def test_escape_filename(self):
        assert_equal(
                r'"abc\"def\\ghi"',
                LinenoLexer._escape_filename(r'abc"def\ghi'))

    @cuda_test
    def test_render(self):
        source = "line 1\nline 2\nline 3"
        out = Template(source, lexer_cls=LinenoLexer).render()
        assert_equal("#line 1 \nline 1\n#line 2 \nline 2\n#line 3 \nline 3\n", out)

class TestArray(object):
    @cuda_test
    def setup(self):
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.constructed = Array(
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)
        self.view = np.zeros(self.padded_shape)[2:4, 3:5].view(Array)
        self.sliced = self.constructed[2:4, 3:5]

    @cuda_test
    def test_safe(self):
        assert Array.safe(self.constructed)
        assert not Array.safe(self.view)
        assert not Array.safe(self.sliced)
        assert not Array.safe(np.zeros(self.shape))

class TestDeviceArray(object):
    @cuda_test
    def setup(self):
        self.context = cuda.Context(pycuda.autoinit.context)
        self.command_queue = cuda.CommandQueue(self.context, None)
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = DeviceArray(
                context=self.context,
                shape=self.shape,
                dtype=np.int32,
                padded_shape=self.padded_shape)

    @cuda_test
    def test_strides(self):
        assert_equal(self.strides, self.array.strides)

    @cuda_test
    def test_empty_like(self):
        ary = self.array.empty_like()
        assert_equal(self.shape, ary.shape)
        assert_equal(self.strides, ary.strides)
        assert Array.safe(ary)

    @cuda_test
    def test_set_get(self):
        ary = np.random.randint(0, 100, self.shape).astype(np.int32)
        self.array.set(self.command_queue, ary)
        # Read back results, check that it matches
        buf = np.zeros(self.padded_shape, dtype=np.int32)
        with self.context:
            pycuda.driver.memcpy_dtoh(buf, self.array.buffer.gpudata)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = self.array.get(self.command_queue)
        np.testing.assert_equal(ary, buf)

class TestTranspose(object):
    def test_transpose(self):
        yield self.check_transpose, 4, 5
        yield self.check_transpose, 53, 7
        yield self.check_transpose, 53, 81
        yield self.check_transpose, 32, 64

    @cuda_test
    def setup(self):
        self.context = cuda.Context(pycuda.autoinit.context)
        self.command_queue = cuda.CommandQueue(self.context, None)
        self.transpose = Transpose(self.command_queue, 'float')

    @cuda_test
    def check_transpose(self, R, C):
        ary = np.random.randn(R, C).astype(np.float32)
        src = DeviceArray(self.context, (R, C), dtype=np.float32, padded_shape=(R + 1, C + 4))
        dest = DeviceArray(self.context, (C, R), dtype=np.float32, padded_shape=(C + 2, R + 3))
        src.set_async(self.command_queue, ary)
        self.transpose(dest, src)
        out = dest.get(self.command_queue)
        np.testing.assert_equal(ary.T, out)
