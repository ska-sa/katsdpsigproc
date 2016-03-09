"""Utilities for interfacing with accelerator hardware. Both OpenCL and CUDA
are supported, as well as multiple devices.

The modules :mod:`cuda` and :mod:`opencl` provide the abstraction layer,
but most code will not import these directly (and it might not be possible
to import them). Instead, use :func:`create_some_context` to set up a context
on whatever device is available.

Attributes
----------
have_cuda : boolean
    True if PyCUDA could be imported (does not guarantee any CUDA devices)
have_opencl : boolean
    True if PyOpenCL could be imported (does not guarantee any OpenCL devices)
"""

from __future__ import division
import numpy as np
import mako.lexer
from mako.template import Template
from mako.lookup import TemplateLookup
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import pkg_resources
import re
import os
import sys
import itertools
from . import tune

try:
    import pycuda.driver
    from . import cuda
    have_cuda = True
except ImportError:
    have_cuda = False

try:
    import pyopencl
    from . import opencl
    have_opencl = True
except ImportError:
    have_opencl = False

def divup(x, y):
    """Divide x by y and round the result upwards"""
    return (x + y - 1) // y

def roundup(x, y):
    """Rounds x up to the next multiple of y"""
    return divup(x, y) * y

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
            escaped_filename = self._escape_filename('<string>')
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

def _make_lookup(extra_dirs):
    dirs = extra_dirs + [pkg_resources.resource_filename(__name__, '')]
    return TemplateLookup(dirs, lexer_cls=LinenoLexer, strict_undefined=True)

# Cached for convenience
_lookup = _make_lookup([])

def build(context, name, render_kws=None, extra_dirs=None, extra_flags=None, source=None):
    """Build a source module from a mako template.

    Parameters
    ----------
    context : :class:`cuda.Context` or `opencl.Context`
        Context for which to compile the code
    name : str
        Source file name, relative to the katsdpsigproc module
    render_kws : dict, optional
        Keyword arguments to pass to mako
    extra_dirs : list of `str`, optional
        Extra directories to search for source code
    extra_flags : list, optional
        Flags to pass to the compiler
    source : str, optional
        If specified, provides the source, overriding `name`

    Returns
    -------
    :class:`cuda.Program` or :class:`opencl.Program`
        Compiled module
    """
    if render_kws is None:
        render_kws = {}
    if extra_flags is None:
        extra_flags = []
    if extra_dirs is None:
        lookup = _lookup
    else:
        lookup = _make_lookup(extra_dirs)
    if source is not None:
        template = Template(source, lookup=lookup, lexer_cls=LinenoLexer)
    else:
        template = lookup.get_template(name)
    rendered_source = template.render(
            simd_group_size=context.device.simd_group_size,
            **render_kws)
    return context.compile(rendered_source, extra_flags)

def all_devices():
    """Return a list of all discovered devices"""
    devices = []
    if have_cuda:
        pycuda.driver.init()
        devices.extend(cuda.Device.get_devices())
    if have_opencl:
        devices.extend(opencl.Device.get_devices())
    return devices

def create_some_context(interactive=True, device_filter=None):
    """Create a single-device context, selecting a device automatically. This
    is similar to `pyopencl.create_some_context`. A number of environment
    variables can be set to limit the choice to a single device:

     - KATSDPSIGPROC_DEVICE: device number from amongst all devices
     - CUDA_DEVICE: CUDA device number (compatible with PyCUDA)
     - PYOPENCL_CTX: OpenCL platform and optionally device number (compatible with PyOpenCL)

    The first of these that is encountered takes effect. If it does not exist,
    an exception is thrown.

    Parameters
    ----------
    interactive : boolean
        If true, and `sys.stdin.isatty()` is true, and there are multiple
        choices, it will prompt the user. Otherwise, it will choose the first
        available device, favouring CUDA over OpenCL, then GPU over
        accelerators over other OpenCL devices.
    device_filter : callable, optional
        If specified, each device in turn is passed to it, and it must return
        True to keep the device as a candidate or False to reject it.

    Raises
    ------
    RuntimeError
        If no device could be found or the user made an invalid selection
    """

    def key(device):
        if device.is_cuda:
            return 100
        elif device.is_gpu:
            return 50
        elif device.is_accelerator:
            return 40
        else:
            return 30

    def parse_id(envar):
        if envar in os.environ:
            try:
                num = int(os.environ[envar])
                if num >= 0:
                    return num
            except ValueError:
                pass
        return None

    def parse_id_list(envar, max_parts):
        """Split an environment variable of the form a[:b[:c...]] into
        its components. If the variable is not validly formatted, returns
        `None`.
        """
        if envar in os.environ:
            try:
                fields = os.environ[envar].split(':', max_parts)
                out = []
                for value in fields:
                    num = int(value)
                    if num < 0:
                        raise ValueError('')
                    out.append(num)
                if not out:
                    raise ValueError('')
                return out
            except ValueError:
                pass
        return None

    cuda_id = None
    opencl_id = None
    device_id = None

    device_id = parse_id('KATSDPSIGPROC_DEVICE')
    if device_id is None:
        cuda_id = parse_id('CUDA_DEVICE')
        if cuda_id is None:
            # Fields are platform number and device number
            opencl_id = parse_id_list('PYOPENCL_CTX', 2)

    cuda_devices = []
    opencl_devices = []
    if have_cuda:
        pycuda.driver.init()
        cuda_devices = cuda.Device.get_devices()
    if have_opencl:
        opencl_devices = opencl.Device.get_devices_by_platform()
    if device_filter is not None:
        cuda_devices = filter(device_filter, cuda_devices)
        opencl_devices = [filter(device_filter, platform) for platform in opencl_devices]

    try:
        if cuda_id is not None:
            devices = [cuda_devices[cuda_id]]
        elif opencl_id is not None:
            if len(opencl_id) == 1:
                devices = opencl_devices[opencl_id[0]]
            else:
                devices = [opencl_devices[opencl_id[0]][opencl_id[1]]]
        else:
            devices = cuda_devices + list(itertools.chain(*opencl_devices))
            if device_id is not None:
                devices = [devices[device_id]]
    except IndexError:
        raise RuntimeError('Out-of-range device selected')

    if not devices:
        raise RuntimeError('No compute devices found')

    if interactive and len(devices) > 1 and sys.stdin.isatty():
        print "Select device:"
        for i, device in enumerate(devices):
            print "    [{0}]: {1} ({2})".format(i, device.name, device.platform_name)
        print
        choice = raw_input('Enter selection: ')
        try:
            choice = int(choice)
            if choice < 0:
                raise IndexError   # Otherwise Python's negative indexing kicks in
            device = devices[choice]
        except (ValueError, IndexError):
            raise RuntimeError('Invalid device number')
    else:
        devices.sort(key=key, reverse=True)
        device = devices[0]

    return device.make_context()

class HostArray(np.ndarray):
    """A restricted array class that can be used to initialise a
    :class:`DeviceArray`. It uses C ordering and allows padding, which
    is always in units of the dtype. It optionally uses pinned memory
    to allow fast transfer to and from device memory.

    Because of limitations in numpy and PyCUDA (which do not support
    non-contiguous memory very well), it works by taking a slice from
    the origin of contiguous storage, and using that contiguous storage
    in host-device copies. While one can create views, those views
    cannot be used in fast copies because there is no way to know
    whether the view is anchored at the origin.

    See the numpy documentation on subclassing for an explanation of
    why it is written the way it is.

    Parameters
    ----------
    shape : tuple
        Shape for the array
    dtype : numpy dtype
        Data type for the array
    padded_shape : tuple, optional
        Total size of memory allocation (defaults to `shape`)
    context : :class:`cuda.Context` or :class:`opencl.Context`, optional
        If specified, the memory will be allocated in a way that allows
        efficient copies to and from this context.
    """
    def __new__(cls, shape, dtype, padded_shape=None, context=None):
        if padded_shape is None:
            padded_shape = shape
        assert len(padded_shape) == len(shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        if context is not None:
            owner = context.allocate_pinned(padded_shape, dtype)
        else:
            owner = np.empty(padded_shape, dtype)
        if shape:
            index = tuple([slice(0, x) for x in shape])
            obj = owner[index].view(HostArray)
        else:
            obj = owner.view(HostArray)
        obj._owner = owner
        obj.padded_shape = padded_shape
        return obj

    @classmethod
    def safe(cls, obj):
        """Determines whether self can be copied to/from a device
        directly.
        """
        try:
            return obj._owner is not None
        except AttributeError:
            return False

    @classmethod
    def padded_view(cls, obj):
        """Retrieves the view of the full memory without padding. Returns
        `None` if `cls.safe(obj)` is `False`."""
        try:
            return obj._owner
        except AttributeError:
            return None

    def __array_finalize__(self, obj):
        if obj is not None:
            # View casting or created from a template: we cannot vouch for it.
            self._owner = None
            if isinstance(obj, HostArray):
                self.padded_shape = obj.padded_shape
            else:
                self.padded_shape = None

class DeviceArray(object):
    """A light-weight array-like wrapper around a device buffer, that
    handles padding better than PyCUDA (which
    has very poor support).

    It only supports C-order arrays where the inner-most dimension is
    contiguous. Transfers are designed to use an :class:`HostArray` of the
    same shape and padding, but fall back to using a copy when
    necessary.

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context in which to allocate the memory
    shape : tuple
        Shape for the usable data
    dtype : numpy dtype
        Data type
    padded_shape : tuple, optional
        Shape for memory allocation (defaults to `shape`)
    raw : low-level allocation from `allocate_raw` method of an allocator
        If specified, provides the backing memory
    """

    def __init__(self, context, shape, dtype, padded_shape=None, raw=None):
        if padded_shape is None:
            padded_shape = shape
        assert len(shape) == len(padded_shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.padded_shape = padded_shape
        self.context = context
        self.buffer = context.allocate(padded_shape, dtype, raw)

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
        return (HostArray.safe(ary) and
                ary.dtype == self.dtype and
                ary.shape == self.shape and
                ary.padded_shape == self.padded_shape)

    @classmethod
    def _contiguous(cls, ary):
        """Returns a contiguous view of a copyable array, for passing to
        PyCUDA or PyOpenCL functions (which require a contiguous view).
        """
        return HostArray.padded_view(ary)

    def empty_like(self):
        """Return an array-like object that can be efficiently copied."""
        return HostArray(self.shape, self.dtype, self.padded_shape, context=self.context)

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

    def set(self, command_queue, ary):
        """Synchronous copy from `ary` to self"""
        ary = self.asarray_like(ary)
        command_queue.enqueue_write_buffer(self.buffer, self._contiguous(ary))

    def get(self, command_queue, ary=None):
        """Synchronous copy from self to `ary`. If `ary` is None,
        or if it is not suitable as a target, the copy is to a newly
        allocated :class:`HostArray`. The actual target is returned.
        """
        if ary is None or not self._copyable(ary):
            ary = self.empty_like()
        command_queue.enqueue_read_buffer(self.buffer, self._contiguous(ary))
        return ary

    def set_async(self, command_queue, ary):
        """Asynchronous copy from `ary` to self"""
        ary = self.asarray_like(ary)
        command_queue.enqueue_write_buffer(
                self.buffer, self._contiguous(ary), blocking=False)

    def get_async(self, command_queue, ary=None):
        """Asynchronous copy from self to `ary` (see `get`)."""
        if ary is None or not self._copyable(ary):
            ary = self.empty_like()
        command_queue.enqueue_read_buffer(
                self.buffer, self._contiguous(ary), blocking=False)
        return ary

    def _canonical_slice(self, region):
        """Transforms a slice selection into a form that is easier to consume
        internally.

        Parameters
        ----------
        region : int, slice, `np.newaxis` or tuple of same
            Index expression generated by `np.s_` or `np.index_exp`

        Returns
        -------
        origin : int
            Byte offset for the first selected element
        shape : tuple of int
            Shape of the region to copy
        strides : tuple of int
            Byte stride between elements along each axis of `shape`. Each stride
            is a multiple of the following ones, except that a dimension of
            length 1 may have a stride of 0.
        """
        origin = 0
        shape = []
        strides = []
        axis = 0
        self_strides = self.strides
        if not isinstance(region, tuple):
            region = (region,)
        for index in region:
            if index is np.newaxis:
                shape.append(1)
                strides.append(0)
            elif isinstance(index, slice):
                if axis >= len(self.shape):
                    raise IndexError('Too many axes in index expression')
                start, stop, stride = index.indices(self.shape[axis])
                if stride <= 0:
                    raise IndexError('Only positive strides are supported')
                origin += start * self_strides[axis]
                shape.append((stop - start) // stride)
                if shape[-1] <= 0:
                    raise IndexError('Empty slice selection')
                strides.append(stride * self_strides[axis])
                axis += 1
            elif isinstance(index, (int, long)):
                if axis >= len(self.shape):
                    raise IndexError('Too many axes in index expression')
                if index < 0:
                    index += self.shape[axis]
                if index < 0 or index >= self.shape[axis]:
                    raise IndexError('Index out of range')
                origin += index * self_strides[axis]
                axis += 1
            else:
                raise TypeError('Invalid type in slice: {}'.format(type(index)))
        while axis < len(self.shape):
            shape.append(self.shape[axis])
            strides.append(self_strides[axis])
            axis += 1
        return origin, shape, strides

    def _copy_region(self, command_queue, dest, src_origin, dest_origin, shape, src_strides, dest_strides):
        """Wrapper around the command-queue's copy that handles high dimensions
        by looping.
        """
        if len(shape) > 3:
            for i in range(shape[-1]):
                self._copy_region(
                        command_queue, dest,
                        src_origin + src_strides[-1] * i,
                        dest_origin + dest_strides[-1] * i,
                        shape[:-1], src_strides[:-1], dest_strides[:-1])
        else:
            command_queue.enqueue_copy_buffer_rect(
                    self.buffer, dest.buffer, src_origin, dest_origin,
                    shape, src_strides, dest_strides)

    def copy_region(self, command_queue, dest, src_region, dest_region):
        """Perform a device-to-device copy of a subregion of `self` to `dest`.
        If the source and destination memory overlap, the result is undefined.

        The source and destination region are specified using a limited subset
        of numpy indexing syntax. The following are supported:

        - slices with positive strides
        - integers
        - :code:`np.newaxis`
        - If fewer indices than axes are specified, all elements on the
          remaining axes are used.

        Ellipses are not yet supported, but it would be straightforward to add
        support.

        Parameters
        ----------
        command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
            Command queue for the asynchronous operation.
        dest : :class:`DeviceArray`
            Target of the copy
        src_region,dest_region : tuple
            Index expressions constructed by `np.s_` or `np.index_exp`.

        Raises
        ------
        TypeError
            if the source and destination do not have the same dtype
        ValueError
            if the source and destination regions select different shapes
        IndexError
            if the source or destination regions are unsupported or out-of-range
        """
        if self.dtype != dest.dtype:
            raise TypeError('dtypes do not match ({} and {})'.format(self.dtype, dest.dtype))
        src_origin, src_shape, src_strides = self._canonical_slice(src_region)
        dest_origin, dest_shape, dest_strides = dest._canonical_slice(dest_region)
        if src_shape != dest_shape:
            raise ValueError('Source and destination shapes do not match')
        # Search for axes that can be collapsed together
        new_shape = [self.dtype.itemsize]
        new_src_strides = [1]
        new_dest_strides = [1]
        for axis in range(len(src_shape) - 1, -1, -1):
            if src_shape[axis] == 1:
                continue    # Can just ignore this axis
            if (src_strides[axis] == new_shape[-1] * new_src_strides[-1] and
                dest_strides[axis] == new_shape[-1] * new_dest_strides[-1]):
                new_shape[-1] *= src_shape[axis]
            else:
                new_shape.append(src_shape[axis])
                new_src_strides.append(src_strides[axis])
                new_dest_strides.append(dest_strides[axis])
        self._copy_region(
                command_queue, dest, src_origin, dest_origin, new_shape,
                new_src_strides, new_dest_strides)

    def zero(self, command_queue):
        """Memset with zeros (asynchronously)"""
        command_queue.enqueue_zero_buffer(self.buffer)


class SVMArray(HostArray, DeviceArray):
    """An array that uses shared virtual memory (aka managed memory) to be
    accessible from both the host and the device. It should not be used as
    a source or target for copies to the device, since it already resides on
    the device.

    Due to limitations in PyOpenCL, this is currently only available for CUDA,
    and CUDA's restrictions on managed memory apply. Only the base array (not
    views) can be passed to kernels.

    Parameters
    ----------
    context : :class:`cuda.Context` or :class:`opencl.Context`
        Context in which to allocate the memory
    shape : tuple
        Shape for the array
    dtype : numpy dtype
        Data type for the array
    padded_shape : tuple, optional
        Total size of memory allocation (defaults to `shape`)
    raw : PyCUDA DeviceAllocation or PyOpenCL Buffer, optional
        If specified, provides the backing memory
    """
    def __new__(cls, context, shape, dtype, padded_shape=None, raw=None):
        if padded_shape is None:
            padded_shape = shape
        assert len(padded_shape) == len(shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        owner = context.allocate_svm(padded_shape, dtype, raw=raw)
        if shape:
            index = tuple([slice(0, x) for x in shape])
            obj = owner[index].view(SVMArray)
        else:
            obj = owner.view(SVMArray)
        obj._owner = owner
        obj.padded_shape = padded_shape
        obj.context = context
        return obj

    def __init__(self, *args, **kwargs):
        # Suppress DeviceArray's constructor, since we did everything in __new__
        pass

    @property
    def buffer(self):
        return self._owner

    def _copyable(self, ary):
        """Whether `ary` can be copied to/from this array directly"""
        return (ary.dtype == self.dtype and
                ary.shape == self.shape)

    def set(self, command_queue, ary):
        """Synchronous copy from `ary` to self. For SVMArray, this
        is a CPU copy."""
        self[...] = ary

    def get(self, command_queue, ary=None):
        """Synchronous copy from self to `ary`. If `ary` is None,
        or if it is not suitable as a target, the copy is to a newly
        allocated :class:`HostArray`. The actual target is returned.
        For SVMArray, this is a CPU copy.
        """
        if ary is None or not self._copyable(ary):
            return self.copy()
        else:
            ary[:] = self
            return ary

    def set_async(self, command_queue, ary):
        """Asynchronous copy from `ary` to self. This is implemented
        synchronously for SVMArray, but exists for compatibility."""
        self.set(command_queue, ary)

    def get_async(self, command_queue, ary=None):
        """Asynchronous copy from self to `ary` (see `get`). This
        is implemented synchronously for SVMArray, but exists for
        compatibility."""
        return self.get(command_queue, ary)


class DeviceAllocator(object):
    """Allocates DeviceArray objects from a context"""
    def __init__(self, context):
        self.context = context

    def allocate(self, shape, dtype, padded_shape=None, raw=None):
        return DeviceArray(self.context, shape, dtype, padded_shape, raw)

    def allocate_raw(self, n_bytes):
        return self.context.allocate_raw(n_bytes)


class SVMAllocator(object):
    """Allocates SVMArray objects from a context"""
    def __init__(self, context):
        self.context = context

    def allocate(self, shape, dtype, padded_shape=None, raw=None):
        return SVMArray(self.context, shape, dtype, padded_shape, raw)

    def allocate_raw(self, n_bytes):
        return self.context.allocate_svm_raw(n_bytes)


class Dimension(object):
    """A single dimension of an :class:`IOSlot`, representing padding and
    alignment requirements. Instances can be linked together, with all
    linked instances (whether directly or indirectly linked) exposing the
    intersection of their requirements. Internally this is represented
    using a union-find tree with path compression.

    Note that `min_padded_round` and `alignment` provide similar but
    different requirements. The former is used to determine a minimum amount
    of padding, but any larger amount is acceptable, even if not a multiple of
    `min_padded_round`. The latter is stricter, always requiring a multiple of
    this value. The latter is not expected to be frequently used except when
    type-punning.

    Dimensions can also be frozen, which is done by :class:`IOSlot` once a
    buffer is bound. This prevents the requirements from changing later (via
    linking or otherwise) and hence invalidating the bound buffer, or
    allowing two buffers with a shared dimension but different strides to
    be used.

    Parameters
    ----------
    size : int
        Size of the actual data, before padding
    min_padded_round : int, optional
        The `min_padded_size` will be computed by rounding `size` up to a
        multiple of this.
    min_padded_size : int, optional
        Minimum size of the padded allocation, overriding `min_padded_round`.
    alignment : int, optional
        Padded size is required to be a multiple of this.
        This must be a power of 2.
    align_dtype : numpy dtype, optional
        If specified, it is a hint that this data type is the fastest-varying
        axis of a multidimensional array. The padded size may be chosen to be
        such that the stride is a multiple of `ALIGN_BYTES`, to ensure
        efficient access by GPUs. The hint will be ignored if `exact` is true.
    exact : bool, optional
        If true, padding is forbidden
    """
    ALIGN_BYTES = 128

    @classmethod
    def _is_power2(cls, value):
        return value > 0 and (value & (value - 1)) == 0

    def __init__(self, size,
                 min_padded_round=None, min_padded_size=None,
                 alignment=1, align_dtype=None, exact=False):
        if min_padded_size is None:
            if min_padded_round is not None:
                min_padded_size = roundup(size, min_padded_round)
            else:
                min_padded_size = size

        if not self._is_power2(alignment):
            raise ValueError('alignment is not a power of 2')
        if min_padded_size < size:
            raise ValueError('padded size is less than size')

        self._parent = None
        self._size = size
        self._min_padded_size = min_padded_size
        self._alignment = alignment
        self._alignment_hint = alignment
        self._exact = exact
        self._frozen = False
        if align_dtype is not None:
            self.add_align_dtype(align_dtype)

    def _root(self):
        if self._parent is None:
            return self
        else:
            self._parent = self._parent._root()
            return self._parent

    @property
    def size(self):
        return self._root()._size

    @property
    def min_padded_size(self):
        return self._root()._min_padded_size

    @property
    def alignment(self):
        return self._root()._alignment

    @property
    def alignment_hint(self):
        return self._root()._alignment_hint

    @property
    def exact(self):
        return self._root()._exact

    @property
    def frozen(self):
        return self._root()._frozen

    def required_padded_size(self):
        """Padded size required to satisfy this dimension"""
        root = self._root()
        size = roundup(root._min_padded_size, root._alignment)
        # If the size is less than the alignment hint, it could
        # waste a huge amount of memory to pad it
        if not root._exact and size >= root._alignment_hint:
            size = roundup(size, root._alignment_hint)
        return size

    def valid(self, padded_size):
        """Whether `size` is a valid padded size"""
        root = self._root()
        if root._exact:
            return padded_size == root.required_padded_size()
        else:
            return (padded_size >= root._min_padded_size and
                    padded_size % root._alignment == 0)

    def add_align_dtype(self, dtype):
        """Add an alignment hint that this will be used with an array whose
        fastest-varying dimension is of type `dtype`. If the size is not
        a power of 2, it is ignored.
        """
        if self.frozen:
            raise ValueError('cannot modify a frozen requirement')
        itemsize = np.dtype(dtype).itemsize
        if self._is_power2(itemsize):
            root = self._root()
            root._alignment_hint = max(root._alignment_hint, self.ALIGN_BYTES // itemsize)

    def link(self, other):
        """Make both `self` and `other` reference a single shared
        requirement.

        Raises
        ------
        ValueError
            If the resulting requirement is exact and unsatisfiable
        """
        root1 = self._root()
        root2 = other._root()
        if root1._frozen or root2._frozen:
            raise ValueError('cannot link frozen requirements')
        if root1 is root2:
            return
        if root1._size != root2._size:
            raise ValueError('sizes are incompatible')
        if root1._exact:
            actual1 = root1.required_padded_size()
            if not root2.valid(actual1):
                raise ValueError('linked requirement is unsatisfiable')
        if root2._exact:
            actual2 = root2.required_padded_size()
            if not root2.valid(actual2):
                raise ValueError('linked requirement is unsatisfiable')
        root1._min_padded_size = max(root1._min_padded_size, root2._min_padded_size)
        root1._alignment = max(root1._alignment, root2._alignment)
        root1._alignment_hint = max(root1._alignment_hint, root2._alignment_hint)
        root1._exact = root1._exact or root2._exact
        root2._parent = root1
        # Clear properties on the child, so that they can't accidentally be used
        # _size can stay as it is immutable
        del root2._min_padded_size
        del root2._alignment
        del root2._alignment_hint
        del root2._exact

    def freeze(self):
        """Prevent further modifications"""
        self._root()._frozen = True

class IOSlotBase(object):
    """An input/output slot of an operation. A slot can be bound to storage,
    or can allocate storage itself. This base class is untyped and unshaped,
    so in most cases one will use :class:`IOSlot` instead.

    Slots are arranged in a tree, and only the root can be manipulated
    directly. The entire tree shares the same storage.
    A slot cannot be reattached to a new parent.
    """
    def __init__(self):
        self.is_root = True

    def check_root(self):
        """Check whether this slot is a root slot, and raise an exception if not."""
        if not self.is_root:
            raise ValueError('not a root slot')

    def required_bytes(self):
        """Number of bytes of device storage required"""
        raise NotImplementedError('abstract base class')

    def is_bound(self):
        """Whether storage is currently attached to this slot"""
        raise NotImplementedError('abstract base class')

    def attachable(self):
        """Whether this slot can be attached as a child to another"""
        return self.is_root and not self.is_bound()

    def _allocate(self, context, raw=None):
        """Variant of :meth:`allocate` that does not check whether this is a
        root slot."""
        raise NotImplementedError('abstract base class')

    def allocate(self, allocator, raw=None):
        """Allocate and immediately bind a buffer satisfying the requirements.

        .. warning:: When `raw` is provided, there is no check that the storage is large enough.

        Parameters
        ----------
        allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`
            Memory allocator from which to obtain the memory
        raw : PyCUDA DeviceAllocation or PyOpenCL Buffer, optional
            Backing storage for the allocation

        Raises
        ------
        ValueError
            If this is not a root slot
        """
        self.check_root()
        return self._allocate(allocator, raw)

class IOSlot(IOSlotBase):
    """An input/output slot with type and shape information. It contains a
    reference to a buffer (initially `None`) and the shape, type, padding and
    alignment requirements for that buffer. It can allocate the buffer on
    request. Each dimension is represesented by a :class:`Dimension`.

    Slots are arranged in a tree, and only the root can be manipulated
    directly. A slot cannot be reattached to a new parent.

    Parameters
    ----------
    dimensions : tuple of int or :class:`Dimension`
        The dimensions of the slots. Integers are converted to
        :class:`Dimension` with no further requirements.
    dtype : numpy dtype
        Type of the data elements
    """

    @classmethod
    def _make_dimension(cls, dimension):
        if isinstance(dimension, Dimension):
            return dimension
        else:
            return Dimension(dimension)

    def __init__(self, dimensions, dtype):
        super(IOSlot, self).__init__()
        self.dimensions = tuple([self._make_dimension(x) for x in dimensions])
        self.shape = tuple([x.size for x in self.dimensions])
        if len(self.dimensions) > 1:
            self.dimensions[-1].add_align_dtype(dtype)
        self.dtype = np.dtype(dtype)
        self.buffer = None

    def is_bound(self):
        return self.buffer is not None

    def validate(self, buffer):
        """Check that `buffer` is suitable for binding.

        Parameters
        ----------
        buffer : :class:`DeviceArray`
            Buffer to validate (must not be `None`)

        Raises
        ------
        TypeError
            If the data type does not match
        ValueError
            If the dimensions or shape do not match
        ValueError
            If a padded size does not match :meth:`Dimension.required_padded_size`
        """
        if buffer.dtype != self.dtype:
            raise TypeError('dtype does not match')
        if len(buffer.shape) != len(self.shape):
            raise ValueError('number of dimensions does not match')
        for size, padded_size, dimension in zip(buffer.shape, buffer.padded_shape, self.dimensions):
            if size != dimension.size:
                raise ValueError('size does not match')
            if padded_size != dimension.required_padded_size():
                raise ValueError('padded size does not match')

    def _bind(self, buffer):
        """Variant of :meth:`bind` that does not check whether this is a root
        slot.
        """
        if buffer is not None:
            self.validate(buffer)
        self.buffer = buffer
        for dimension in self.dimensions:
            dimension.freeze()

    def bind(self, buffer):
        """Set the internal buffer reference. Always use this function rather
        that writing it directly.

        If the buffer is not `None`, it is validated (see :meth:`validate`).

        Parameters
        ----------
        buffer : :class:`DeviceArray` or `None`
            Buffer to store

        Raises
        ------
        ValueError
            If this is not a root slot
        """
        self.check_root()
        self._bind(buffer)

    def required_padded_shape(self):
        """Padded shape required to satisfy only this slot"""
        return tuple([x.required_padded_size() for x in self.dimensions])

    def required_bytes(self):
        return np.product(self.required_padded_shape()) * self.dtype.itemsize

    def _allocate(self, allocator, raw=None):
        buffer = allocator.allocate(self.shape, self.dtype, self.required_padded_shape(), raw=raw)
        self._bind(buffer)
        return buffer

class CompoundIOSlot(IOSlot):
    """IO slot that owns multiple child slots, and presents the combined
    requirement. The children must all have the same type and shape. This
    is used for connecting a single buffer to multiple operations.

    Parameters
    ----------
    children : sequence of :class:`IOSlot`
        Child slots

    Raises
    ------
    ValueError
        If `children` is empty, or if elements have inconsistent shapes
    ValueError
        If any child is not attachable
    TypeError
        If `children` contains inconsistent data types
    """

    def __init__(self, children):
        self.children = list(children)
        if not len(self.children):
            raise ValueError('empty child list')
        shape = children[0].shape
        dtype = children[0].dtype
        dimensions = children[0].dimensions
        # Validate consistency and compute combined requirements
        for child in children:
            if not child.attachable():
                raise ValueError('child is not attachable')
            if child.shape != shape:
                raise ValueError('inconsistent shapes')
            if child.dtype != dtype:
                raise TypeError('inconsistent dtypes')
            for dimension in child.dimensions:
                if dimension.frozen:
                    raise ValueError('child has frozen dimensions')
        for child in children:
            for x, y in zip(dimensions, child.dimensions):
                x.link(y)
        super(CompoundIOSlot, self).__init__(dimensions, dtype)
        for child in children:
            child.is_root = False

    def _bind(self, buffer):
        super(CompoundIOSlot, self)._bind(buffer)
        for child in self.children:
            child._bind(buffer)

class AliasIOSlot(IOSlotBase):
    """Slot that aggregates multiple child slots (which need not have the same
    type or shape), and allocates a single low-level buffer to back all of them.

    This is typically used when logically distinct slots can share memory
    because they do not contain live data at the same time.

    Parameters
    ----------
    children : list of :class:`IOSlotBase`

    Raises
    ------
    ValueError
        If `children` is empty or contains non-attachable elements
    """
    def __init__(self, children):
        super(AliasIOSlot, self).__init__()
        self.children = children
        self.raw = None
        if not len(self.children):
            raise ValueError('empty child list')
        for child in self.children:
            if not child.attachable():
                raise ValueError('child is not attachable')
        for child in self.children:
            child.is_root = False

    def is_bound(self):
        return self.raw is not None

    def required_bytes(self):
        return max([child.required_bytes() for child in self.children])

    def _allocate(self, allocator, raw=None):
        if raw is None:
            raw = allocator.allocate_raw(self.required_bytes())
        for child in self.children:
            child._allocate(allocator, raw)
        self.raw = raw
        return raw

class Operation(object):
    """An instance of a device operation. Typically one first creates a
    template (which contains the program code, and is expensive to create) and
    then instantiates it for use with specific buffers.

    An instance of this class contains *slots*, which are named instances of
    :class:`IOSlot`. The user binds specific buffers to these slots to specify
    the memory used in the operation.

    This class is only useful when subclassed. The subclass will populate
    the slots. Subclasses also provide a `_run` function that handles the
    implementation of `__call__`.

    Operations are arranged in a tree, with internal nodes subclassing
    :class:`OperationSequence`. Internal nodes provide slots that proxy for
    the slots of their children; the children's slots should not be
    manipulated directly. When a parent ceases to exist, its children should
    no longer be used i.e., they cannot be reattached to a new parent.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots

    Attributes
    ----------
    slots : dictionary
        Maps names to slot instances
    hidden_slots : dictionary
        Extra slots which are not root slots, and hence cannot have buffers
        bound to them, but whose buffers can still be referenced. This is
        generally used by :class:`OperationSequence` when several slots are
        aliased to the same memory.
    is_root : boolean
        True if this operation is not part of an :class:`OperationSequence`.
    """
    def __init__(self, command_queue, allocator=None):
        if allocator is None:
            allocator = DeviceAllocator(command_queue.context)
        elif allocator.context is not None:
            if allocator.context is not command_queue.context:
                raise ValueError('command_queue and allocator have different contexts')
        self.slots = {}
        self.hidden_slots = {}
        self.command_queue = command_queue
        self.is_root = True
        self.allocator = allocator

    def bind(self, **kwargs):
        """Bind buffers to slots by keyword. Each keyword argument name
        specifies a slot name.
        """
        for name, buffer in kwargs.items():
            self.slots[name].bind(buffer)

    def ensure_all_bound(self):
        """Make sure that all slots have a buffer bound, allocating if necessary"""
        for slot in self.slots.itervalues():
            if not slot.is_bound():
                slot.allocate(self.allocator)

    def buffer(self, name):
        """Retrieve the buffer bound to a slot. It will consult
        both :attr:`slots` and :attr:`hidden_slots`.

        Parameters
        ----------
        name : str
            Name of the slot to access

        Returns
        -------
        :class:`DeviceArray`
            Buffer bound to slot `name`, or `None` if the slot exists but is unbound

        Raises
        ------
        KeyError
            If no slot with this name exists
        """
        try:
            slot = self.slots[name]
        except KeyError:
            try:
                slot = self.hidden_slots[name]
            except KeyError:
                raise KeyError('no slot named ' + name)
        return slot.buffer

    def required_bytes(self):
        """Number of bytes of device storage required"""
        return sum([x.required_bytes() for x in self.slots.itervalues()])

    def parameters(self):
        """Returns dictionary of configuration options for this operation"""
        return {}

    def _run(self):
        raise NotImplementedError('abstract base class')

    def __call__(self, **kwargs):
        """Run the operation. Any slots that are not already bound will
        allocate a new buffer. Keyword arguments are passed to :meth:`bind`.
        """
        self.bind(**kwargs)
        self.ensure_all_bound()
        return self._run()

class OperationSequence(Operation):
    """Convenience class for setting up an operation that is built up of
    smaller named operations, with mappings of slots to share data.
    Initially, each slot named *slot* in a child named *op* is remapped to a
    parent slot named *op*:*slot*. After this, each set provided in
    `compounds` is removed from the slots and combined into a single compound
    slot. Finally, each set in `aliases` is removed and combined into an
    alias slot.

    For both `compounds` and `aliases`, if a child does not exist, it is
    skipped, and if none of the children exist, no action is taken.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    operations : sequence of 2-tuples
        Name, operation pairs to add. Calling the operation executes them in order
    compounds : mapping of `str` to sequence of `str`, optional
        Names for compound slots, mapped to the original slot names that are replaced
    aliases : mapping of `str` to sequence of `str`, optional
        Names for alias slots, mapped to the original slot names that are replaced
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, command_queue, operations, compounds=None, aliases=None, allocator=None):
        super(OperationSequence, self).__init__(command_queue, allocator)
        self.operations = OrderedDict(operations)
        for (name, operation) in operations:
            if operation.command_queue is not command_queue:
                raise ValueError('child has a different command queue to the parent')
            if not operation.is_root:
                raise ValueError('child already has another parent')
            for (slot_name, slot) in operation.slots.items():
                self.slots[name + ':' + slot_name] = slot
        if compounds is not None:
            for (name, child_names) in compounds.iteritems():
                children = self._extract_slots(child_names, False)
                if children:
                    self.slots[name] = CompoundIOSlot(children)
        if aliases is not None:
            for (name, child_names) in aliases.iteritems():
                children = self._extract_slots(child_names, True)
                if children:
                    self.slots[name] = AliasIOSlot(children)
        for operation in self.operations.itervalues():
            operation.is_root = False

    def _extract_slots(self, names, add_to_hidden):
        """Remove and return the slots with the given names"""
        ans = []
        for name in names:
            try:
                ans.append(self.slots[name])
                del self.slots[name]
                if add_to_hidden:
                    assert name not in self.hidden_slots
                    self.hidden_slots[name] = ans[-1]
            except KeyError:
                # Ignore missing items
                pass
        return ans

    def _run(self):
        for operation in self.operations.values():
            operation()
