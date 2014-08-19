import functools
import numpy as np
from mako.template import Template
from nose.tools import assert_equal
from nose.plugins.skip import SkipTest
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    have_cuda = True
except ImportError:
    have_cuda = False

if have_cuda:
    from ..accel import Array, DeviceArray, LinenoLexer

def cuda_test(test):
    """Decorator that causes a test to be skipped if CUDA is not available"""
    def wrapper(*args, **kw):
        global have_cuda
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
        self.ctx = pycuda.autoinit.context
        self.shape = (17, 13)
        self.padded_shape = (32, 16)
        self.strides = (64, 4)
        self.array = DeviceArray(
                ctx=self.ctx,
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
        self.array.set(ary)
        # Read back results, check that it matches
        buf = np.zeros(self.padded_shape, dtype=np.int32)
        cuda.memcpy_dtoh(buf, self.array.buffer.gpudata)
        buf = buf[0:self.shape[0], 0:self.shape[1]]
        np.testing.assert_equal(ary, buf)
        # Check that it matches get
        buf = self.array.get()
        np.testing.assert_equal(ary, buf)
