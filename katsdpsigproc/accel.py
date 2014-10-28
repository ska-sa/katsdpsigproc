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
from mako.lookup import TemplateLookup
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import pkg_resources
import re
import os
import sys
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

def _make_lookup(extra_dirs):
    dirs = extra_dirs + [pkg_resources.resource_filename(__name__, '')]
    return TemplateLookup(dirs, lexer_cls=LinenoLexer, strict_undefined=True)

# Cached for convenience
_lookup = _make_lookup([])

def build(context, name, render_kws=None, extra_dirs=None, extra_flags=None):
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
    source = lookup.get_template(name).render(
            simd_group_size=context.device.simd_group_size,
            **render_kws)
    return context.compile(source, extra_flags)

def create_some_context(interactive=True):
    """Create a single-device context, selecting a device automatically. This
    is similar to `pyopencl.create_some_context`. A number of environment
    variables can be set to limit the choice to a single device:

     - KATSDPSIGPROC_DEVICE: device number from amongst all devices
     - CUDA_DEVICE: CUDA device number (compatible with PyCUDA)
     - PYOPENCL_CTX: OpenCL device number (compatible with PyOpenCL)

    The first of these that is encountered takes effect. If it does not exist,
    an exception is thrown.

    Parameters
    ----------
    interactive : boolean
        If true, and `sys.stdin.isatty()` is true, and there are multiple
        choices, it will prompt the user. Otherwise, it will choose the first
        available device, favouring CUDA over OpenCL, then GPU over
        accelerators over other OpenCL devices.

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

    cuda_id = None
    opencl_id = None
    device_id = None

    device_id = parse_id('KATSDPSIGPROC_DEVICE')
    if device_id is None:
        cuda_id = parse_id('CUDA_DEVICE')
        if cuda_id is None:
            opencl_id = parse_id('PYOPENCL_CTX')

    cuda_devices = []
    opencl_devices = []
    if have_cuda:
        pycuda.driver.init()
        cuda_devices = cuda.Device.get_devices()
    if have_opencl:
        opencl_devices = opencl.Device.get_devices()

    if cuda_id is not None:
        devices = [cuda_devices[cuda_id]]
    elif opencl_id is not None:
        devices = [opencl_devices[opencl_id]]
    else:
        devices = cuda_devices + opencl_devices
        if device_id is not None:
            devices = [devices[device_id]]

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
            owner = context.allocate_pinned(padded_shape, dtype).view(HostArray)
        else:
            owner = np.empty(padded_shape, dtype).view(HostArray)
        index = tuple([slice(0, x) for x in shape])
        obj = owner[index]
        obj._accel_safe = True
        obj.padded_shape = padded_shape
        return obj

    @classmethod
    def safe(cls, obj):
        """Determines whether self can be copied to/from a device
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
        if isinstance(self, HostArray):
            self._accel_safe = False

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
    raw : PyCUDA DeviceAllocation or PyOpenCL Buffer, optional
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
        if raw is None:
            self.buffer = context.allocate(padded_shape, dtype, raw)
        else:
            self.buffer = raw

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
        if ary.base is None:
            return ary
        else:
            return ary.base

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

    def allocate(self, context, raw=None):
        """Allocate and immediately bind a buffer satisfying the requirements.

        .. warning:: When `raw` is provided, there is no check that the storage is large enough

        Parameters
        ----------
        context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
            Context in which to allocate the memory
        raw : PyCUDA DeviceAllocation or PyOpenCL Buffer, optional
            Backing storage for the :class:`DeviceArray`

        Raises
        ------
        ValueError
            If this is not a root slot
        """
        self.check_root()
        return self._allocate(context, raw)

class IOSlot(IOSlotBase):
    """An input/output slot with type and shape information. It contains a
    reference to a buffer (initially `None`) and the shape, type, padding and
    alignment requirements for that buffer. It can allocate the buffer on
    request.

    Note that `min_padded_round` and `alignment` provide similar but
    different requirements. The former is used to determine a minimum amount
    of padding, but any larger amount is acceptable, even if not a multiple of
    `min_padded_round`. The latter is stricter, always requiring a multiple of
    this value. The latter is not expected to be frequently used except when
    type-punning.

    Slots are arranged in a tree, and only the root can be manipulated
    directly. A slot cannot be reattached to a new parent.

    Parameters
    ----------
    shape : tuple of int
        The exact size of the data itself
    dtype : numpy dtype
        Type of the data elements
    min_padded_round : tuple of int, optional
        The `min_padded_shape` will be computed by rounding `shape` up to a
        multiple of this.
    min_padded_shape : tuple of int, optional
        Minimum size of the padded allocation, overriding `min_padded_round`.
    alignment : tuple of int, optional
        Padded size is required to be a multiple of this in each dimension.
        These values must be powers of 2.
    """

    def __init__(self, shape, dtype, min_padded_round=None, min_padded_shape=None, alignment=None):
        super(IOSlot, self).__init__()
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        if min_padded_shape is None:
            if min_padded_round is None:
                self.min_padded_shape = tuple(shape)
            else:
                assert len(min_padded_round) == len(shape)
                self.min_padded_shape = tuple([roundup(x, y) for x, y in zip(shape, min_padded_round)])
        else:
            assert len(min_padded_shape) == len(shape)
            assert np.all(np.greater_equal(min_padded_shape, shape))
            self.min_padded_shape = tuple(min_padded_shape)
        if alignment is None:
            self.alignment = tuple(1 for x in shape)
        else:
            assert len(alignment) == len(shape)
            self.alignment = tuple(alignment)
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
            If the dimensions or shape do not match, or the padding or alignment are invalid
        """
        if buffer.dtype != self.dtype:
            raise TypeError('dtype does not match')
        if len(buffer.shape) != len(self.shape):
            raise ValueError('number of dimensions does not match')
        if buffer.shape != self.shape:
            raise ValueError('shape does not match')
        if np.any(np.less(buffer.padded_shape, self.min_padded_shape)):
            raise ValueError('padded shape is too small')
        for i in range(len(self.alignment)):
            if buffer.padded_shape[i] % self.alignment[i] != 0:
                raise ValueError('padded shape is not correctly aligned')

    def _bind(self, buffer):
        """Variant of :meth:`bind` that does not check whether this is a root
        slot.
        """
        if buffer is not None:
            self.validate(buffer)
        self.buffer = buffer

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
        padded_shape = list(self.min_padded_shape)
        for i, a in enumerate(self.alignment):
            padded_shape[i] = roundup(padded_shape[i], a)
        return tuple(padded_shape)

    def required_bytes(self):
        return np.product(self.required_padded_shape()) * self.dtype.itemsize

    def _allocate(self, context, raw=None):
        buffer = DeviceArray(context, self.shape, self.dtype, self.required_padded_shape(), raw=raw)
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
        min_padded_shape = children[0].min_padded_shape
        alignment = children[0].alignment
        # Validate consistency and compute combined requirements
        for child in children:
            if not child.attachable():
                raise ValueError('child is not attachable')
            if child.shape != shape:
                raise ValueError('inconsistent shapes')
            if child.dtype != dtype:
                raise TypeError('inconsistent dtypes')
            min_padded_shape = np.maximum(min_padded_shape, child.min_padded_shape)
            alignment = np.maximum(alignment, child.alignment)
        super(CompoundIOSlot, self).__init__(shape, dtype, None, min_padded_shape, alignment)
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

    def _allocate(self, context, raw=None):
        if raw is None:
            raw = context.allocate_raw(self.required_bytes())
        for child in self.children:
            child._allocate(context, raw)
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
    the slots. It also conventially provides a `__call__` method which takes
    an arbitrary set of keyword arguments, which are passed to :meth:`bind`.

    Operations are arranged in a tree, with internal nodes subclassing
    :class:`OperationSequence`. Internal nodes provide slots that proxy for
    the slots of their children; the children's slots should not be
    manipulated directly. When a parent ceases to exist, its children should
    no longer be used i.e., they cannot be reattached to a new parent.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    """
    def __init__(self, command_queue):
        self.slots = {}
        self.command_queue = command_queue
        self.is_root = True

    def bind(self, **kwargs):
        """Bind buffers to slots by keyword. Each keyword argument name
        specifies a slot name.
        """
        for name, buffer in kwargs.items():
            self.slots[name].bind(buffer)

    def ensure_all_bound(self):
        """Make sure that all slots have a buffer bound, allocating if necessary"""
        for name, slot in self.slots.items():
            if not slot.is_bound():
                slot.allocate(self.command_queue.context)

    def required_bytes(self):
        """Number of bytes of device storage required"""
        return sum([x.required_bytes() for x in self.slots.itervalues()])

    def parameters(self):
        """Returns dictionary of configuration options for this operation"""
        return {}

class OperationSequence(Operation):
    """Convenience class for setting up an operation that is built up of
    from smaller named operations, with mappings of slots to share data.
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
    aliases : mapping of `str` to `str`, optional
        Names for alias slots, mapped to the original slot names that are replaced
    """
    def __init__(self, command_queue, operations, compounds=None, aliases=None):
        super(OperationSequence, self).__init__(command_queue)
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
                children = self._extract_slots(child_names)
                if children:
                    self.slots[name] = CompoundIOSlot(children)
        if aliases is not None:
            for (name, child_names) in aliases.iteritems():
                children = self._extract_slots(child_names)
                if children:
                    self.slots[name] = AliasIOSlot(children)
        for operation in self.operations.itervalues():
            operation.is_root = False

    def _extract_slots(self, names):
        """Remove and return the slots with the given names"""
        ans = []
        for name in names:
            try:
                ans.append(self.slots[name])
                del self.slots[name]
            except KeyError:
                # Ignore missing items
                pass
        return ans

    def __call__(self, **kwargs):
        self.bind(**kwargs)
        self.ensure_all_bound()
        for operation in self.operations.values():
            operation()
