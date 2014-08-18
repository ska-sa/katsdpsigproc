import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import mako.lexer
import re
from contextlib import contextmanager

"""Utilities for interfacing with accelerator hardware. Currently only CUDA
is supported, but it is intended to later support OpenCL too. It currently
only supports a single CUDA context/device.
"""

class LinenoLexer(mako.lexer.Lexer):
    """A wrapper that inserts #line directives into the source code. It
    is used by passing `lexer_cls` to the mako template constructor.
    """
    def __init__(self, *args, **kw):
        super(LinenoLexer, self).__init__(*args, **kw)
        self.preprocessor.insert(0, self._lineno_preproc)

    @classmethod
    def _escape_filename(cls, filename):
        """Escapes a string for the C preprocessor"""
        return '"' + re.sub(r'([\\"])', r'\\\1', filename) + '"'

    def _lineno_preproc(self, source):
        if self.filename is not None:
            escaped_filename = self._escape_filename(self.filename)
        else:
            escaped_filename = ''
        lines = source.split('\n')
        # If the last line is \n-terminated, it will cause an empty
        # final string in lines
        if len(lines) > 0 and lines[-1] == '':
            lines.pop()
        out = []
        for i, line in enumerate(lines):
            out.append('#line {0} {1}\n'.format(i + 1, escaped_filename))
            out.append(line + '\n')
        return ''.join(out)

@contextmanager
def push_context(ctx):
    ctx.push()
    yield
    ctx.pop()

class Array(np.ndarray):
    """A restricted array class that can be used to initialise a
    :class:`DeviceArray`. It uses C ordering and allows padding, which
    is always in units of the dtype.

    See the numpy documentation on subclassing for an explanation of
    why it is written the way it is.
    """
    def __new__(cls, shape, dtype, padded_shape=None):
        if padded_shape is None:
            padded_shape = shape
        assert len(padded_shape) == len(shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        owner = cuda.pagelocked_empty(padded_shape, dtype).view(Array)
        index = tuple([slice(0, x) for x in shape])
        obj = owner[index]
        obj._accel_safe = True
        return obj

    @classmethod
    def safe(cls, obj):
        try:
            return obj._accel_safe
        except AttributeError:
            return False

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # View casting or created from a template: we cannot vouch for it.
        # We can't unconditionally set the attribute, because the base class
        # doesn't allow new attributes.
        if isinstance(self, Array):
            self._accel_safe = False

class DeviceArray(object):
    """A light-weight array-like wrapper around a device buffer, that
    handles padding better than `pycuda.gpuarray.GPUArray` (which
    has very poor support).

    It only supports C-order arrays where the inner-most dimension is
    contiguous. Transfers are designed to use an :class:`Array` of the
    same shape and padding, but fall back to using a copy when
    necessary.
    """

    def __init__(self, ctx, shape, dtype, padded_shape=None):
        if padded_shape is None:
            padded_shape = shape
        assert len(shape) == len(padded_shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.padded_shape = padded_shape
        self.ctx = ctx
        with push_context(ctx):
            self.buffer = gpuarray.GPUArray(padded_shape, dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def strides(self):
        s = [self.dtype.itemsize]
        for i in range(len(self.padded_shape) - 1, 0, -1):
            s.append(s[-1] * self.padded_shape[i])
        return tuple(reversed(s))

    def _copyable(self, ary):
        return (Array.safe(ary) and
                ary.dtype == self.dtype and
                ary.shape == self.shape and
                ary.padded_shape == self.padded_shape)

    def _contiguous(self, ary):
        """Returns a contiguous view of a copyable array, for passing to
        PyCUDA functions (which require a contiguous view).
        """
        if ary.base is None:
            return ary
        else:
            return ary.base

    def empty_like(self):
        """Return an array-like object that can be efficiently copied."""
        return Array(self.shape, self.dtype, self.padded_shape)

    def asarray_like(self, ary):
        """Return an array with the same content as `ary`, but the same memory
        layout as self.
        """
        assert ary.shape == self.shape
        if self._copyable(ary):
            return ary
        tmp = self.empty_like()
        np.copyto(tmp, ary, casting = 'no')
        return tmp

    def set(self, ary):
        ary = self.asarray_like(ary)
        with push_context(self.ctx):
            self.buffer.set(self._contiguous(ary))

    def get(self, ary=None):
        if ary is None:
            ary = self.empty_like()
        with push_context(self.ctx):
            self.buffer.get(self._contiguous(ary))
        return ary

    def set_async(self, ary, stream=None):
        ary = self.asarray_like(ary)
        with push_context(self.ctx):
            self.buffer.set_async(self._contiguous(ary), stream=stream)

    def get_async(self, ary, stream=None):
        if ary is None:
            ary = self.empty_like()
        with push_context(self.ctx):
            self.buffer.get_async(self._contiguous(ary), stream=stream)
        return ary
