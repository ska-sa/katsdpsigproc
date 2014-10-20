#!/usr/bin/env python
import sys
import numpy as np
from decorator import decorator
from mako.template import Template
from nose.tools import assert_equal
from nose.plugins.skip import SkipTest
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
