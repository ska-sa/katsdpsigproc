import numpy as np
import pycuda.driver as cuda
import mako.lexer
import re

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
        self.preprocessor.insert(0, self.lineno_preproc)

    @classmethod
    def _escape_filename(cls, filename):
        """Escapes a string for the C preprocessor"""
        return '"' + re.sub(r'([\\"])', r'\\\1', filename) + '"'

    def lineno_preproc(self, source):
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
        owner = np.ndarray.__new__(cls, padded_shape, dtype)
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
    contiguous. Transfers are most efficient when the numpy array
    involved in the transfer have the same memory layout as the
    target (otherwise a copy is made).
    """

    def __init__(self, shape, dtype, padded_shape=None):
        if padded_shape is None:
            padded_shape = shape
        assert len(shape) == len(padded_shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.padded_shape = padded_shape
        size = reduce(lambda x, y: x * y, padded_shape) * self.dtype.itemsize
        self.buffer = cuda.mem_alloc(size)

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
                ary.strides == self.strides)

    def _contiguous(self, ary):
        """Returns a contiguous view of a copyable array, for passing to
        PyCUDA functions (which require a contiguous view).
        """
        if ary.base is None:
            return ary
        else:
            # The slice is so that we do not include any extra padding beyond
            # the last row.
            return ary.base[0:self.shape[0]]

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

    def set(self, ary, thread=None):
        ary = self.asarray_like(ary)
        cuda.memcpy_htod(self.buffer, self._contiguous(ary))

    def get(self):
        ary = self.empty_like()
        cuda.memcpy_dtoh(self._contiguous(ary), self.buffer)
        return ary
