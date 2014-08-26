"""Utilities for interfacing with accelerator hardware. Currently only CUDA
is supported, but it is intended to later support OpenCL too. It currently
only supports a single CUDA context/device.
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import mako.lexer
from mako.lookup import TemplateLookup
import pkg_resources
import re
from contextlib import contextmanager

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
        """Insert #line directives between every line in `source` and return
        the result.
        """
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
    """Context manager that pushes a CUDA context on entry and pops it
    on exit.
    """
    ctx.push()
    yield
    ctx.pop()

_lookup = TemplateLookup(
        pkg_resources.resource_filename(__name__, ''), lexer_cls=LinenoLexer)
_nvcc_flags = DEFAULT_NVCC_FLAGS + ['-lineinfo']

def build(name, render_kws = None, extra_flags = None):
    """Build a source module from a mako template.

    Parameters
    ----------
    name : str
        Source file name, relative to the katsdpsigproc module
    render_kws : dict (optional)
        Context arguments to pass to mako
    extra_flags : list (optional)
        Flags to pass to the compiler

    Returns
    -------
    `pycuda.compiler.SourceModule`
        Compiled module
    """
    if render_kws is None:
        render_kws = {}
    if extra_flags is None:
        extra_flags = []
    source = _lookup.get_template(name).render(**render_kws)
    return SourceModule(source, options=_nvcc_flags + extra_flags, no_extern_c=True)

class Array(np.ndarray):
    """A restricted array class that can be used to initialise a
    :class:`DeviceArray`. It uses C ordering and allows padding, which
    is always in units of the dtype. It is allocated from page-locked
    CUDA memory.

    Because of limitations in numpy and PyCUDA (which do not support
    non-contiguous memory very well), it works by taking a slice from
    the origin of contiguous storage, and using that contiguous storage
    in host-device copies. While one can create views, those views
    cannot be used in these copied because there is no way to know
    whether the view is anchored at the origin.

    See the numpy documentation on subclassing for an explanation of
    why it is written the way it is.
    """
    def __new__(cls, shape, dtype, padded_shape=None):
        """Constructor.

        Parameters
        ----------
        shape : tuple
            Shape for the usable data
        dtype : numpy dtype
            Data type
        padded_shape : tuple or `None`
            Shape for memory allocation (defaults to `shape`)
        """
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
        """Determines whether self can be copied to/from the GPU
        directly.
        """
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
        """Constructor.

        Parameters
        ----------
        ctx : `pycuda.driver.Context`
            CUDA context in which to allocate the memory
        shape : tuple
            Shape for the usable data
        dtype : numpy dtype
            Data type
        padded_shape : tuple or `None`
            Shape for memory allocation (defaults to `shape`)
        """
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
        """Number of dimensions"""
        return len(self.shape)

    @property
    def strides(self):
        """Strides, as in numpy"""
        ans = [self.dtype.itemsize]
        for i in range(len(self.padded_shape) - 1, 0, -1):
            ans.append(ans[-1] * self.padded_shape[i])
        return tuple(reversed(ans))

    def _copyable(self, ary):
        """Whether `ary` can be copied to/from this array directly"""
        return (Array.safe(ary) and
                ary.dtype == self.dtype and
                ary.shape == self.shape and
                ary.padded_shape == self.padded_shape)

    @classmethod
    def _contiguous(cls, ary):
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
        np.copyto(tmp, ary, casting='no')
        return tmp

    def set(self, ary):
        """Synchronous copy from `ary` to self"""
        ary = self.asarray_like(ary)
        with push_context(self.ctx):
            self.buffer.set(self._contiguous(ary))

    def get(self, ary=None):
        """Synchronous copy from self to `ary`. If `ary` is None,
        or if it is not suitable as a target, the copy is to a newly
        allocated :class:`Array`. The actual target is returned.
        """
        if ary is None or not self._copyable(ary):
            ary = self.empty_like()
        with push_context(self.ctx):
            self.buffer.get(self._contiguous(ary))
        return ary

    def set_async(self, ary, stream=None):
        """Asynchronous copy from `ary` to self"""
        ary = self.asarray_like(ary)
        with push_context(self.ctx):
            self.buffer.set_async(self._contiguous(ary), stream=stream)

    def get_async(self, ary, stream=None):
        """Asynchronous copy from self to `ary` (see `get`)."""
        if ary is None or not self._copyable(ary):
            ary = self.empty_like()
        with push_context(self.ctx):
            self.buffer.get_async(self._contiguous(ary), stream=stream)
        return ary

class Transpose(object):
    def __init__(self, ctx, ctype):
        self.ctx = ctx
        self.block = 32
        module = build("transpose.cu", {'block': self.block, 'ctype': ctype})
        self.kernel = module.get_function("transpose")

    def __call__(self, dest, src, stream=None):
        assert src.dtype == dest.dtype
        assert src.shape[0] == dest.shape[1]
        assert src.shape[1] == dest.shape[0]
        in_row_blocks = (src.shape[0] + self.block - 1) // self.block
        in_col_blocks = (src.shape[1] + self.block - 1) // self.block
        self.kernel(
                dest.buffer, src.buffer,
                np.int32(src.shape[0]), np.int32(src.shape[1]),
                np.int32(dest.padded_shape[1]),
                np.int32(src.padded_shape[1]),
                block=(self.block, self.block, 1),
                grid=(in_col_blocks, in_row_blocks),
                stream=stream)
