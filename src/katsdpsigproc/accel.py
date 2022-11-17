################################################################################
# Copyright (c) 2014-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Utilities for interfacing with accelerator hardware.

Both OpenCL and CUDA are supported, as well as multiple devices.

The modules :mod:`.cuda` and :mod:`.opencl` provide the abstraction layer,
but most code will not import these directly (and it might not be possible
to import them). Instead, use :func:`create_some_context` to set up a context
on whatever device is available.

Attributes
----------
have_cuda : bool
    True if PyCUDA could be imported (does not guarantee any CUDA devices)
have_opencl : bool
    True if PyOpenCL could be imported (does not guarantee any OpenCL devices)
"""

import re
import os
import sys
import io
from abc import ABC, abstractmethod
import itertools
from collections import OrderedDict
from typing import (List, Tuple, Dict, Set, Mapping, MutableMapping, Sequence,
                    Callable, Iterable, Optional, Union, TypeVar, Generic, Any,
                    cast, overload, TYPE_CHECKING)

import numpy as np
try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any     # type: ignore
import mako.lexer
from mako.template import Template
from mako.lookup import TemplateLookup
import pkg_resources

from .abc import (AbstractContext, AbstractDevice, AbstractCommandQueue,  # noqa: F401
                  AbstractTuningCommandQueue, AbstractProgram)

try:
    import pycuda.driver
    from . import cuda
    have_cuda = True
except ImportError:
    have_cuda = False
try:
    from . import opencl
    have_opencl = True
except ImportError:
    have_opencl = False

if TYPE_CHECKING:
    import graphviz


_Slice = Union[int, slice, None, Tuple[Union[int, slice, None], ...]]
_B = TypeVar('_B')     # buffer type
_RB = TypeVar('_RB')   # raw buffer type
_RS = TypeVar('_RS')   # raw buffer for SVM
_D = TypeVar('_D', bound='AbstractDevice')
_P = TypeVar('_P', bound='AbstractProgram')
_Q = TypeVar('_Q', bound='AbstractCommandQueue')
_TQ = TypeVar('_TQ', bound='AbstractTuningCommandQueue')


def divup(x: int, y: int) -> int:
    """Divide x by y and round the result upwards."""
    return (x + y - 1) // y


def roundup(x: int, y: int) -> int:
    """Round x up to the next multiple of y."""
    return divup(x, y) * y


class LinenoLexer(mako.lexer.Lexer):
    """Insert #line directives into the source code.

    It is used by passing `lexer_cls` to the mako template constructor.
    """

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.preprocessor.insert(0, self._lineno_preproc)

    @classmethod
    def _escape_filename(cls, filename: str) -> str:
        """Escape a string for the C preprocessor."""
        return '"' + re.sub(r'([\\"])', r'\\\1', filename) + '"'

    def _lineno_preproc(self, source: str) -> str:
        """Insert #line directives between every line in `source` and return the result."""
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
            out.append('#line {} {}\n'.format(i + 1, escaped_filename))
            out.append(line + '\n')
        return ''.join(out)


def _make_lookup(extra_dirs: List[str]) -> TemplateLookup:
    dirs = extra_dirs + [pkg_resources.resource_filename(__name__, '')]
    return TemplateLookup(dirs, lexer_cls=LinenoLexer, strict_undefined=True)


# Cached for convenience
_lookup = _make_lookup([])


def build(context: AbstractContext, name: str,
          render_kws: Optional[Mapping[str, Any]] = None,
          extra_dirs: Optional[List[str]] = None,
          extra_flags: Optional[List[str]] = None,
          source: Optional[str] = None) -> AbstractProgram:
    """Build a source module from a mako template.

    Parameters
    ----------
    context
        Context for which to compile the code
    name
        Source file name, relative to the katsdpsigproc module
    render_kws
        Keyword arguments to pass to mako
    extra_dirs
        Extra directories to search for source code
    extra_flags
        Flags to pass to the compiler
    source
        If specified, provides the source, overriding `name`

    Returns
    -------
    program
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


def all_devices() -> List[AbstractDevice]:
    """Return a list of all discovered devices."""
    devices = []     # type: List[AbstractDevice]
    if have_cuda:
        pycuda.driver.init()
        devices.extend(cuda.Device.get_devices())
    if have_opencl:
        devices.extend(opencl.Device.get_devices())
    return devices


def candidate_devices(device_filter: Optional[Callable[[AbstractDevice], bool]] = None) \
        -> Sequence[AbstractDevice]:
    """Get devices that are considered for :func:`create_some_context`.

    Refer to :func:`create_some_context` for documentation of how this list is
    affected by `device_filter` and environment variables. If no matching
    devices are found, returns an empty list. If an environment variable is
    out of range, raises :exc:`RuntimeError`.
    """

    def parse_id(envar: str) -> Optional[int]:
        if envar in os.environ:
            try:
                num = int(os.environ[envar])
                if num >= 0:
                    return num
            except ValueError:
                pass
        return None

    def parse_id_list(envar: str, max_parts: int) -> Optional[List[int]]:
        """Split an environment variable of the form a[:b[:c...]] into its components.

        If the variable is not validly formatted, returns `None`.
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

    cuda_id = None         # type: Optional[int]
    opencl_id = None       # type: Optional[List[int]]
    device_id = None       # type: Optional[int]

    device_id = parse_id('KATSDPSIGPROC_DEVICE')
    if device_id is None:
        cuda_id = parse_id('CUDA_DEVICE')
        if cuda_id is None:
            # Fields are platform number and device number
            opencl_id = parse_id_list('PYOPENCL_CTX', 2)

    cuda_devices = []      # type: Sequence[cuda.Device]
    opencl_devices = []    # type: Sequence[Sequence[opencl.Device]]
    devices = []           # type: List[AbstractDevice]
    if have_cuda:
        pycuda.driver.init()
        cuda_devices = cuda.Device.get_devices()
    if have_opencl:
        opencl_devices = opencl.Device.get_devices_by_platform()
    if device_filter is not None:
        cuda_devices = list(filter(device_filter, cuda_devices))
        opencl_devices = [list(filter(device_filter, platform)) for platform in opencl_devices]

    try:
        if cuda_id is not None:
            devices.append(cuda_devices[cuda_id])
        elif opencl_id is not None:
            if len(opencl_id) == 1:
                devices.extend(opencl_devices[opencl_id[0]])
            else:
                devices.append(opencl_devices[opencl_id[0]][opencl_id[1]])
        else:
            devices = list(itertools.chain(cuda_devices, *opencl_devices))
            if device_id is not None:
                devices = [devices[device_id]]
    except IndexError:
        raise RuntimeError('Out-of-range device selected')
    return devices


def create_some_context(
        interactive: bool = True,
        device_filter: Optional[Callable[[AbstractDevice], bool]] = None) -> AbstractContext:
    """Create a single-device context, selecting a device automatically.

    This is similar to `pyopencl.create_some_context`. A number of environment
    variables can be set to limit the choice to a single device:

     - KATSDPSIGPROC_DEVICE: device number from amongst all devices
     - CUDA_DEVICE: CUDA device number (compatible with PyCUDA)
     - PYOPENCL_CTX: OpenCL platform and optionally device number (compatible with PyOpenCL)

    The first of these that is encountered takes effect. If it does not exist,
    an exception is thrown.

    Parameters
    ----------
    interactive
        If true, and `sys.stdin.isatty()` is true, and there are multiple
        choices, it will prompt the user. Otherwise, it will choose the first
        available device, favouring CUDA over OpenCL, then GPU over
        accelerators over other OpenCL devices.
    device_filter
        If specified, each device in turn is passed to it, and it must return
        True to keep the device as a candidate or False to reject it.

    Raises
    ------
    RuntimeError
        If no device could be found or the user made an invalid selection
    """
    def key(device: AbstractDevice) -> int:
        if device.is_cuda:
            return 100
        elif device.is_gpu:
            return 50
        elif device.is_accelerator:
            return 40
        else:
            return 30

    devices = candidate_devices(device_filter)
    if not devices:
        raise RuntimeError('No compute devices found')

    if interactive and len(devices) > 1 and sys.stdin.isatty():
        print("Select device:")
        for i, device in enumerate(devices):
            print(f"    [{i}]: {device.name} ({device.platform_name})")
        print()
        choice_str = input('Enter selection: ')
        try:
            choice = int(choice_str)
            if choice < 0:
                raise IndexError   # Otherwise Python's negative indexing kicks in
            device = devices[choice]
        except (ValueError, IndexError):
            raise RuntimeError('Invalid device number')
    else:
        device = max(devices, key=key)

    return device.make_context()


class HostArray(np.ndarray):
    """A restricted array class that can be used to initialise a :class:`DeviceArray`.

    It uses C ordering and allows padding, which is always in units of the
    dtype. It optionally uses pinned memory to allow fast transfer to and from
    device memory.

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
    shape
        Shape for the array
    dtype
        Data type for the array
    padded_shape
        Total size of memory allocation (defaults to `shape`)
    context
        If specified, the memory will be allocated in a way that allows
        efficient copies to and from this context.
    """

    def __new__(cls, shape: Tuple[int, ...], dtype: DTypeLike,
                padded_shape: Optional[Tuple[int, ...]] = None,
                context: Optional[AbstractContext] = None):
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
    def safe(cls, obj: np.ndarray) -> bool:
        """Determine whether self can be copied to/from a device directly."""
        try:
            return obj._owner is not None  # type: ignore
        except AttributeError:
            return False

    @overload
    @classmethod
    def padded_view(cls, obj: 'HostArray') -> np.ndarray:
        ...

    @overload
    @classmethod
    def padded_view(cls, obj: np.ndarray) -> Optional[np.ndarray]:    # noqa: F811
        ...

    @classmethod
    def padded_view(cls, obj: np.ndarray) -> Optional[np.ndarray]:    # noqa: F811
        """Retrieve the view of the full memory without padding.

        Returns `None` if `cls.safe(obj)` is `False`.
        """
        try:
            return obj._owner  # type: ignore
        except AttributeError:
            return None

    def __array_finalize__(self, obj: 'HostArray') -> None:
        if obj is not None:
            # View casting or created from a template: we cannot vouch for it.
            self._owner = None
            if isinstance(obj, HostArray):
                self.padded_shape = obj.padded_shape    # type: Tuple[int, ...]
            else:
                self.padded_shape = None


class DeviceArray:
    """A light-weight array-like wrapper around a device buffer.

    It that handles padding better than PyCUDA (which has very poor support).

    It only supports C-order arrays where the inner-most dimension is
    contiguous. Transfers are designed to use an :class:`HostArray` of the
    same shape and padding, but fall back to using a copy when
    necessary.

    Parameters
    ----------
    context
        Context in which to allocate the memory
    shape
        Shape for the usable data
    dtype
        Data type
    padded_shape
        Shape for memory allocation (defaults to `shape`)
    raw
        If specified, provides the backing memory. It must be created from
        the `allocate_raw` method of an allocator.
    """

    def __init__(self, context: AbstractContext, shape: Tuple[int, ...],
                 dtype: DTypeLike, padded_shape: Optional[Tuple[int, ...]] = None,
                 raw: Any = None) -> None:
        if padded_shape is None:
            padded_shape = shape
        assert len(shape) == len(padded_shape)
        assert np.all(np.greater_equal(padded_shape, shape))
        self._shape = shape
        self._dtype: np.dtype = np.dtype(dtype)
        self.padded_shape = padded_shape
        self.context = context
        self.buffer = context.allocate(padded_shape, dtype, raw)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self.shape)

    @property
    def strides(self) -> Tuple[int, ...]:
        """Return strides, as in numpy."""
        ans = [self.dtype.itemsize]
        for i in range(len(self.padded_shape) - 1, 0, -1):
            ans.append(ans[-1] * self.padded_shape[i])
        return tuple(reversed(ans))

    def _copyable(self, ary: np.ndarray) -> bool:
        """Whether `ary` can be copied to/from this array directly."""
        return (HostArray.safe(ary)
                and ary.dtype == self.dtype
                and ary.shape == self.shape
                and ary.padded_shape == self.padded_shape)  # type: ignore

    @classmethod
    def _contiguous(cls, ary: HostArray) -> np.ndarray:
        """Return a contiguous view of a copyable array.

        The return value is suitable for passing to PyCUDA or PyOpenCL
        functions (which require a contiguous view).
        """
        return HostArray.padded_view(ary)

    def empty_like(self) -> HostArray:
        """Return an array-like object that can be efficiently copied."""
        return HostArray(self.shape, self.dtype, self.padded_shape, context=self.context)

    def asarray_like(self, ary: np.ndarray) -> HostArray:
        """Return an array with the same content as `ary`, but the same memory layout as `self`."""
        assert ary.shape == self.shape
        if self._copyable(ary):
            return ary  # type: ignore
        tmp = self.empty_like()
        np.copyto(tmp, ary, casting='no')
        return tmp

    def set(self, command_queue: AbstractCommandQueue, ary: np.ndarray) -> None:
        """Copy synchronously from `ary` to self."""
        ary = self.asarray_like(ary)
        command_queue.enqueue_write_buffer(self.buffer, self._contiguous(ary))

    def get(self, command_queue: AbstractCommandQueue,
            ary: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy synchronously from self to `ary`.

        If `ary` is None, or if it is not suitable as a target, the copy is to
        a newly-allocated :class:`HostArray`. The actual target is returned.
        """
        if ary is None or not self._copyable(ary):
            ary = self.empty_like()
        assert isinstance(ary, HostArray)
        command_queue.enqueue_read_buffer(self.buffer, self._contiguous(ary))
        return ary

    def set_async(self, command_queue: AbstractCommandQueue, ary: np.ndarray) -> None:
        """Copy asynchronously from `ary` to self."""
        ary = self.asarray_like(ary)
        command_queue.enqueue_write_buffer(
            self.buffer, self._contiguous(ary), blocking=False)

    def get_async(self, command_queue: AbstractCommandQueue,
                  ary: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy asynchronously from self to `ary` (see :meth:`get`)."""
        if ary is None or not self._copyable(ary):
            ary = self.empty_like()
        assert isinstance(ary, HostArray)
        command_queue.enqueue_read_buffer(
            self.buffer, self._contiguous(ary), blocking=False)
        return ary

    @classmethod
    def _canonical_slice(cls, region: _Slice, shape: Tuple[int, ...],
                         strides: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        """Transform a slice selection into a form that is easier to consume internally.

        See :meth:`copy_region` for a description of what slice
        expressions are supported.

        Parameters
        ----------
        region : int, slice, `np.newaxis` or tuple of same
            Index expression generated by `np.s_` or `np.index_exp`
        shape : sequence
            Shape of the array being indexed
        strides : sequence
            Strides of the array being indexed

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
        copy_shape = []      # type: List[int]
        copy_strides = []
        axis = 0
        if not isinstance(region, tuple):
            region = (region,)
        for index in region:
            if index is np.newaxis:
                copy_shape.append(1)
                copy_strides.append(0)
            elif isinstance(index, slice):
                if axis >= len(shape):
                    raise IndexError('Too many axes in index expression')
                start, stop, stride = index.indices(shape[axis])
                if stride <= 0:
                    raise IndexError('Only positive strides are supported')
                origin += start * strides[axis]
                copy_shape.append((stop - start) // stride)
                if copy_shape[-1] <= 0:
                    raise IndexError('Empty slice selection')
                copy_strides.append(stride * strides[axis])
                axis += 1
            elif isinstance(index, int):
                if axis >= len(shape):
                    raise IndexError('Too many axes in index expression')
                if index < 0:
                    index += shape[axis]
                if index < 0 or index >= shape[axis]:
                    raise IndexError('Index out of range')
                origin += index * strides[axis]
                axis += 1
            else:
                raise TypeError('Invalid type in slice: {}'.format(type(index)))
        while axis < len(shape):
            copy_shape.append(shape[axis])
            copy_strides.append(strides[axis])
            axis += 1
        return origin, tuple(copy_shape), tuple(copy_strides)

    @classmethod
    def _region_transfer_params(
            cls, src: Any, dest: Any,
            src_region: _Slice,
            dest_region: _Slice) -> Tuple[int, int, Tuple[int, ...],
                                          Tuple[int, ...], Tuple[int, ...]]:
        """Compute offsets, strides and shape for a copy.

        Where possible, the dimension of the copy is reduced by exploiting
        contiguity. When interpreting the return values, the arrays should be
        thought of as byte arrays, ignoring any shape, padding or dtype of the
        originals.

        Parameters
        ----------
        src,dest : array-like
            Source and destination buffers, with `dtype, `shape` and `strides`
            attributes
        src_region,dest_region : int, slice, `np.newaxis` or tuple of same
            Index expression generated by `np.s_` or `np.index_exp`

        Returns
        -------
        src_origin,dest_origin : int
            Byte offset into the source/destination array for the start of the copy
        shape : tuple
            Shape for the copy, which may have fewer dimensions than the arrays.
        src_strides,dest_strides : tuple
            Byte strides for source/destination.

        Raises
        ------
        TypeError
            if the source and destination do not have the same dtype
        ValueError
            if the source and destination regions are not the same shape
        IndexError
            if the source or destination regions are unsupported or out-of-range
        """
        if src.dtype != dest.dtype:
            raise TypeError(f'dtypes do not match ({src.dtype} and {dest.dtype})')
        src_origin, src_shape, src_strides = cls._canonical_slice(
            src_region, src.shape, src.strides)
        dest_origin, dest_shape, dest_strides = cls._canonical_slice(
            dest_region, dest.shape, dest.strides)
        if src_shape != dest_shape:
            raise ValueError('Source and destination shapes for the copy do not match')
        # Search for axes that can be collapsed together
        new_shape = [src.dtype.itemsize]
        new_src_strides = [1]
        new_dest_strides = [1]
        for axis in range(len(src_shape) - 1, -1, -1):
            if src_shape[axis] == 1:
                continue    # Can just ignore this axis
            if (src_strides[axis] == new_shape[-1] * new_src_strides[-1]
                    and dest_strides[axis] == new_shape[-1] * new_dest_strides[-1]):
                new_shape[-1] *= src_shape[axis]
            else:
                new_shape.append(src_shape[axis])
                new_src_strides.append(src_strides[axis])
                new_dest_strides.append(dest_strides[axis])
        return (src_origin, dest_origin,
                tuple(new_shape), tuple(new_src_strides), tuple(new_dest_strides))

    @classmethod
    def _transfer_region(cls, func: Callable,
                         buffer1: Any, buffer2: Any,
                         origin1: int, origin2: int,
                         shape: Tuple[int, ...],
                         strides1: Tuple[int, ...], strides2: Tuple[int, ...],
                         **kwargs) -> None:
        """Wrap the command-queue's copy and handle high dimensions by looping."""
        if len(shape) > 3:
            for i in range(shape[-1]):
                cls._transfer_region(
                    func, buffer1, buffer2,
                    origin1 + strides1[-1] * i,
                    origin2 + strides2[-1] * i,
                    shape[:-1], strides1[:-1], strides2[:-1], **kwargs)
        else:
            func(buffer1, buffer2, origin1, origin2, shape, strides1, strides2, **kwargs)

    def copy_region(self, command_queue: AbstractCommandQueue, dest: 'DeviceArray',
                    src_region: _Slice, dest_region: _Slice) -> None:
        """Perform a device-to-device copy of a subregion of `self` to `dest`.

        If the source and destination memory overlap, the result is undefined.

        The regions to copy are specified using a subset of numpy array
        indexing syntax. The following are supported:

        - slices with positive strides
        - integers
        - :data:`np.newaxis <numpy.newaxis>`
        - If fewer indices than axes are specified, all elements on the
          remaining axes are used.

        Ellipses are not yet supported, but it would be straightforward to add
        support.

        Parameters
        ----------
        command_queue
            Command queue for the asynchronous operation.
        dest
            Target of the copy
        src_region,dest_region
            Index expressions constructed by :data:`np.s_ <numpy.s_>` or
            :data:`!np.index_exp`.

        Raises
        ------
        TypeError
            if the source and destination do not have the same dtype
        ValueError
            if the source and destination regions select different shapes
        IndexError
            if the source or destination regions are unsupported or out-of-range
        """
        src_origin, dest_origin, shape, src_strides, dest_strides = \
            self._region_transfer_params(self, dest, src_region, dest_region)
        self._transfer_region(
            command_queue.enqueue_copy_buffer_rect,
            self.buffer, dest.buffer, src_origin, dest_origin,
            shape, src_strides, dest_strides)

    def get_region(self, command_queue: AbstractCommandQueue, ary: np.ndarray,
                   device_region: _Slice, ary_region: _Slice, blocking: bool = True) -> None:
        """Perform a device-to-host copy of a subregion of `self` to `ary`.

        See :meth:`~DeviceArray.copy_region` for a description of how regions
        are specified.

        Parameters
        ----------
        command_queue
            Command queue for the operation.
        ary
            Target of the copy
        device_region,ary_region
            Index expressions constructed by :data:`np.s_ <numpy.s_>` or
            :data:`!np.index_exp`, to specify the source and target regions.
        blocking
            If false, the operation will be asynchronous.

        Raises
        ------
        TypeError
            if the source and destination do not have the same dtype
        ValueError
            if the source and destination regions select different shapes
        ValueError
            if `ary` is not a :class:`HostArray` or is a view of one
        IndexError
            if the source or destination regions are unsupported or out-of-range
        """
        if not HostArray.safe(ary):
            raise ValueError('Target region is not suitable for device-to-host copy')
        assert isinstance(ary, HostArray)
        device_origin, ary_origin, shape, device_strides, ary_strides = \
            self._region_transfer_params(self, ary, device_region, ary_region)
        self._transfer_region(
            command_queue.enqueue_read_buffer_rect,
            self.buffer, self._contiguous(ary), device_origin, ary_origin,
            shape, device_strides, ary_strides, blocking=blocking)

    def set_region(self, command_queue: AbstractCommandQueue, ary: np.ndarray,
                   device_region: _Slice, ary_region: _Slice, blocking: bool = True) -> None:
        """Perform a host-to-device copy of a subregion `ary` to `self`.

        See :meth:`~DeviceArray.copy_region` for a description of how regions
        are specified.

        Parameters
        ----------
        command_queue
            Command queue for the operation.
        ary
            Source of the copy
        device_region,ary_region
            Index expressions constructed by :data:`np.s_ <numpy.s_>` or
            :data:`!np.index_exp`, to specify the source and target regions.
        blocking
            If false, the operation will be asynchronous.

        Raises
        ------
        TypeError
            if the source and destination do not have the same dtype
        ValueError
            if the source and destination regions select different shapes
        IndexError
            if the source or destination regions are unsupported or out-of-range
        """
        if not HostArray.safe(ary):
            tmp = HostArray(ary[ary_region].shape, ary.dtype, context=self.context)
            np.copyto(tmp, ary[ary_region], casting='no')
            ary = tmp
            ary_region = np.s_[()]
        assert isinstance(ary, HostArray)
        ary_origin, device_origin, shape, ary_strides, device_strides = \
            self._region_transfer_params(ary, self, ary_region, device_region)
        self._transfer_region(
            command_queue.enqueue_write_buffer_rect,
            self.buffer, self._contiguous(ary), device_origin, ary_origin,
            shape, device_strides, ary_strides, blocking=blocking)

    def zero(self, command_queue: AbstractCommandQueue) -> None:
        """Memset with zeros (asynchronously)."""
        command_queue.enqueue_zero_buffer(self.buffer)


class SVMArray(HostArray, DeviceArray):
    """An array that uses shared virtual memory to be accessible from both the host and the device.

    Shared virtual memory is also known as managed memory (in CUDA).

    It should not be used as a source or target for copies to the device, since
    it already resides on the device.

    Due to limitations in PyOpenCL, this is currently only available for CUDA,
    and CUDA's restrictions on managed memory apply. Only the base array (not
    views) can be passed to kernels.

    Parameters
    ----------
    context
        Context in which to allocate the memory
    shape
        Shape for the array
    dtype
        Data type for the array
    padded_shape
        Total size of memory allocation (defaults to `shape`)
    raw : PyCUDA DeviceAllocation or PyOpenCL Buffer, optional
        If specified, provides the backing memory
    """

    def __new__(cls, context: AbstractContext, shape: Tuple[int, ...], dtype: DTypeLike,
                padded_shape: Optional[Tuple[int, ...]] = None, raw: Any = None) -> 'SVMArray':
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
    def buffer(self) -> np.ndarray:
        return self._owner  # type: ignore

    def _copyable(self, ary: np.ndarray) -> bool:
        """Whether `ary` can be copied to/from this array directly."""
        return (ary.dtype == self.dtype
                and ary.shape == self.shape)

    def set(self, command_queue: AbstractCommandQueue, ary: np.ndarray) -> None:
        """Copy synchronously from `ary` to self.

        For SVMArray, this is a CPU copy.
        """
        # Ensure that it synchronises with previous work in the queue
        command_queue.finish()
        self[...] = ary

    def get(self, command_queue: AbstractCommandQueue,
            ary: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy synchronously copy from self to `ary`.

        If `ary` is None, or if it is not suitable as a target, the copy is to
        a newly allocated :class:`HostArray`. The actual target is returned.
        For SVMArray, this is a CPU copy.
        """
        # Ensure that it synchronises with previous work in the queue
        command_queue.finish()
        if ary is None or not self._copyable(ary):
            return self.copy()
        else:
            ary[:] = self
            return ary

    def set_async(self, command_queue: AbstractCommandQueue, ary: np.ndarray) -> None:
        """Copy asynchronously from `ary` to self.

        This is implemented synchronously for SVMArray, but exists for
        compatibility.
        """
        self.set(command_queue, ary)

    def get_async(self, command_queue: AbstractCommandQueue,
                  ary: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy asynchronously from self to `ary` (see `get`).

        This is implemented synchronously for SVMArray, but exists for
        compatibility.
        """
        return self.get(command_queue, ary)

    def set_region(self, command_queue: AbstractCommandQueue, ary: np.ndarray,
                   device_region: _Slice, ary_region: _Slice, blocking: bool = True) -> None:
        # Ensure that it synchronises with previous work in the queue
        command_queue.finish()
        self[device_region] = ary[ary_region]

    def get_region(self, command_queue: AbstractCommandQueue, ary: np.ndarray,
                   device_region: _Slice, ary_region: _Slice, blocking: bool = True) -> None:
        # Ensure that it synchronises with previous work in the queue
        command_queue.finish()
        ary[ary_region] = self[device_region]


class AbstractAllocator(ABC, Generic[_RB]):
    """Interface for allocating device memory."""

    context = None    # type: AbstractContext

    @abstractmethod
    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 padded_shape: Optional[Tuple[int, ...]] = None,
                 raw: Optional[_RB] = None) -> DeviceArray:
        pass

    @abstractmethod
    def allocate_raw(self, n_bytes: int) -> _RB:
        pass


class DeviceAllocator(Generic[_B, _RB, _RS, _D, _P, _Q, _TQ], AbstractAllocator[_RB]):
    """Allocate :class:`DeviceArray` objects from a context."""

    def __init__(self, context: AbstractContext[_B, _RB, _RS, _D, _P, _Q, _TQ]) -> None:
        self.context = context

    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 padded_shape: Optional[Tuple[int, ...]] = None,
                 raw: Optional[_RB] = None) -> DeviceArray:
        return DeviceArray(self.context, shape, dtype, padded_shape, raw)

    def allocate_raw(self, n_bytes: int) -> _RB:
        return self.context.allocate_raw(n_bytes)


class SVMAllocator(Generic[_B, _RB, _RS, _D, _P, _Q, _TQ], AbstractAllocator[_RS]):
    """Allocate :class:`SVMArray` objects from a context."""

    def __init__(self, context: AbstractContext[_B, _RB, _RS, _D, _P, _Q, _TQ]) -> None:
        self.context = context

    def allocate(self, shape: Tuple[int, ...], dtype: DTypeLike,
                 padded_shape: Optional[Tuple[int, ...]] = None,
                 raw: Optional[_RS] = None) -> SVMArray:
        return SVMArray(self.context, shape, dtype, padded_shape, raw)

    def allocate_raw(self, n_bytes: int) -> _RS:
        return self.context.allocate_svm_raw(n_bytes)


class Dimension:
    """A single dimension of an :class:`IOSlot`.

    It represents padding and alignment requirements. Instances can be linked
    together, with all linked instances (whether directly or indirectly linked)
    exposing the intersection of their requirements. Internally this is
    represented using a union-find tree with path compression.

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
    size
        Size of the actual data, before padding
    min_padded_round
        The `min_padded_size` will be computed by rounding `size` up to a
        multiple of this.
    min_padded_size
        Minimum size of the padded allocation, overriding `min_padded_round`.
    alignment
        Padded size is required to be a multiple of this.
        This must be a power of 2.
    align_dtype
        If specified, it is a hint that this data type is the fastest-varying
        axis of a multidimensional array. The padded size may be chosen to be
        such that the stride is a multiple of `ALIGN_BYTES`, to ensure
        efficient access by GPUs. The hint will be ignored if `exact` is true.
    exact
        If true, padding is forbidden
    """

    ALIGN_BYTES = 128

    @classmethod
    def _is_power2(cls, value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def __init__(self, size: int,
                 min_padded_round: Optional[int] = None, min_padded_size: Optional[int] = None,
                 alignment: int = 1, align_dtype: Optional[DTypeLike] = None,
                 exact: bool = False) -> None:
        if min_padded_size is None:
            if min_padded_round is not None:
                min_padded_size = roundup(size, min_padded_round)
            else:
                min_padded_size = size

        if not self._is_power2(alignment):
            raise ValueError('alignment is not a power of 2')
        if min_padded_size < size:
            raise ValueError('padded size is less than size')

        self._parent = None     # type: Optional[Dimension]
        self._size = size
        self._min_padded_size = min_padded_size
        self._alignment = alignment
        self._alignment_hint = alignment
        self._exact = exact
        self._frozen = False
        if align_dtype is not None:
            self.add_align_dtype(align_dtype)

    def _root(self) -> 'Dimension':
        if self._parent is None:
            return self
        else:
            self._parent = self._parent._root()
            return self._parent

    @property
    def size(self) -> int:
        return self._root()._size

    @property
    def min_padded_size(self) -> int:
        return self._root()._min_padded_size

    @property
    def alignment(self) -> int:
        return self._root()._alignment

    @property
    def alignment_hint(self) -> int:
        return self._root()._alignment_hint

    @property
    def exact(self) -> bool:
        return self._root()._exact

    @property
    def frozen(self) -> bool:
        return self._root()._frozen

    def required_padded_size(self) -> int:
        """Return padded size required to satisfy this dimension."""
        root = self._root()
        size = roundup(root._min_padded_size, root._alignment)
        # If the size is less than the alignment hint, it could
        # waste a huge amount of memory to pad it
        if not root._exact and size >= root._alignment_hint:
            size = roundup(size, root._alignment_hint)
        return size

    def valid(self, padded_size: int) -> bool:
        """Return whether `size` is a valid padded size."""
        root = self._root()
        if root._exact:
            return padded_size == root.required_padded_size()
        else:
            return (padded_size >= root._min_padded_size
                    and padded_size % root._alignment == 0)

    def add_align_dtype(self, dtype: DTypeLike) -> None:
        """Add an alignment hint.

        Indicates that this will be used with an array whose fastest-varying
        dimension is of type `dtype`. If the size is not a power of 2, it is
        ignored.
        """
        if self.frozen:
            raise ValueError('cannot modify a frozen requirement')
        itemsize = np.dtype(dtype).itemsize
        if self._is_power2(itemsize):
            root = self._root()
            root._alignment_hint = max(root._alignment_hint, self.ALIGN_BYTES // itemsize)

    def link(self, other: 'Dimension') -> None:
        """Make both `self` and `other` reference a single shared requirement.

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

    def freeze(self) -> None:
        """Prevent further modifications."""
        self._root()._frozen = True


class IOSlotBase(ABC):
    """An input/output slot of an operation.

    A slot can be bound to storage, or can allocate storage itself. This base
    class is untyped and unshaped, so in most cases one will use
    :class:`IOSlot` instead.

    Slots are arranged in a tree, and only the root can be manipulated
    directly. The entire tree shares the same storage.
    A slot cannot be reattached to a new parent.
    """

    def __init__(self) -> None:
        self.is_root = True

    def check_root(self) -> None:
        """Check whether this slot is a root slot, and raise an exception if not."""
        if not self.is_root:
            raise ValueError('not a root slot')

    @abstractmethod
    def required_bytes(self) -> int:
        """Return number of bytes of device storage required."""

    @abstractmethod
    def is_bound(self):
        """Return whether storage is currently attached to this slot."""

    def attachable(self) -> bool:
        """Return whether this slot can be attached as a child to another."""
        return self.is_root and not self.is_bound()

    @abstractmethod
    def _allocate(self, allocator: AbstractAllocator[_RB], raw: Optional[_RB] = None,
                  *, bind: bool) -> Any:
        """Variant of :meth:`allocate` that does not check whether this is a root slot."""

    def allocate(self, allocator: AbstractAllocator[_RB], raw: Optional[_RB] = None,
                 *, bind: bool = True) -> Any:
        """Allocate and optionally bind a buffer satisfying the requirements.

        .. warning:: When `raw` is provided, there is no check that the storage is large enough.

        Parameters
        ----------
        allocator
            Memory allocator from which to obtain the memory
        raw : PyCUDA DeviceAllocation or PyOpenCL Buffer, optional
            Backing storage for the allocation
        bind
            If true (the default), the allocated buffer is immediately bound.
            If false, the buffer is returned and not bound, and can be bound
            later.

        Raises
        ------
        ValueError
            If this is not a root slot
        """
        self.check_root()
        return self._allocate(allocator, raw, bind=bind)

    @abstractmethod
    def _allocate_host(self, context: AbstractContext) -> HostArray:
        """Variant of :meth:`allocate_host` that does not check whether this is a root slot."""

    def allocate_host(self, context: AbstractContext) -> HostArray:
        """Allocate a HostArray compatible with this slot."""
        self.check_root()
        return self._allocate_host(context)


class IOSlot(IOSlotBase):
    """An input/output slot with type and shape information.

    It contains a reference to a buffer (initially `None`) and the shape, type,
    padding and alignment requirements for that buffer. It can allocate the
    buffer on request. Each dimension is represesented by a :class:`Dimension`.

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
    def _make_dimension(cls, dimension: Union[Dimension, int]) -> Dimension:
        if isinstance(dimension, Dimension):
            return dimension
        else:
            return Dimension(dimension)

    def __init__(self, dimensions: Tuple[Union[Dimension, int], ...], dtype: DTypeLike) -> None:
        super().__init__()
        self.dimensions = tuple([self._make_dimension(x) for x in dimensions])
        self.shape = tuple([x.size for x in self.dimensions])
        if len(self.dimensions) > 1:
            self.dimensions[-1].add_align_dtype(dtype)
        self.dtype: np.dtype = np.dtype(dtype)
        self.buffer = None       # type: Optional[DeviceArray]

    def is_bound(self) -> bool:
        return self.buffer is not None

    def validate(self, buffer: DeviceArray) -> None:
        """Check that `buffer` is suitable for binding.

        Parameters
        ----------
        buffer
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

    def _bind(self, buffer: Optional[DeviceArray]) -> None:
        """Variant of :meth:`bind` that does not check whether this is a root slot."""
        if buffer is not None:
            self.validate(buffer)
        self.buffer = buffer
        for dimension in self.dimensions:
            dimension.freeze()

    def bind(self, buffer: Optional[DeviceArray]) -> None:
        """Set the internal buffer reference.

        Always use this function rather than writing it directly.

        If the buffer is not `None`, it is validated (see :meth:`validate`).

        Parameters
        ----------
        buffer
            Buffer to store

        Raises
        ------
        ValueError
            If this is not a root slot
        """
        self.check_root()
        self._bind(buffer)

    def required_padded_shape(self) -> Tuple[int, ...]:
        """Return padded shape required to satisfy only this slot."""
        return tuple(x.required_padded_size() for x in self.dimensions)

    def required_bytes(self) -> int:
        return int(np.product(self.required_padded_shape()) * self.dtype.itemsize)

    def _allocate(self, allocator: AbstractAllocator[_RB],
                  raw: Optional[_RB] = None, *, bind: bool = True) -> DeviceArray:
        buffer = allocator.allocate(self.shape, self.dtype, self.required_padded_shape(), raw=raw)
        if bind:
            self._bind(buffer)
        return buffer

    def _allocate_host(self, context: AbstractContext) -> HostArray:
        return HostArray(self.shape, self.dtype, self.required_padded_shape(), context=context)

    def allocate(self, allocator: AbstractAllocator[_RB],
                 raw: Optional[_RB] = None, *, bind: bool = True) -> DeviceArray:
        # Overloaded just to restrict the type signature
        return super().allocate(allocator, raw, bind=bind)


class CompoundIOSlot(IOSlot):
    """IO slot that owns multiple child slots, and presents the combined requirement.

    The children must all have the same type and shape. This is used for
    connecting a single buffer to multiple operations.

    Parameters
    ----------
    children
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

    def __init__(self, children: Iterable[IOSlot]) -> None:
        self.children = list(children)
        if not len(self.children):
            raise ValueError('empty child list')
        shape = self.children[0].shape
        dtype = self.children[0].dtype
        dimensions = self.children[0].dimensions
        # Validate consistency and compute combined requirements
        for child in self.children:
            if not child.attachable():
                raise ValueError('child is not attachable')
            if child.shape != shape:
                raise ValueError('inconsistent shapes')
            if child.dtype != dtype:
                raise TypeError('inconsistent dtypes')
            for dimension in child.dimensions:
                if dimension.frozen:
                    raise ValueError('child has frozen dimensions')
        for child in self.children:
            for x, y in zip(dimensions, child.dimensions):
                x.link(y)
        super().__init__(dimensions, dtype)
        for child in self.children:
            child.is_root = False

    def _bind(self, buffer: Optional[DeviceArray]) -> None:
        super()._bind(buffer)
        for child in self.children:
            child._bind(buffer)


class AliasIOSlot(IOSlotBase):
    """Slot that allocates a single low-level buffer to back multiple children.

    The child slots need not have the same type or shape. This is typically
    used when logically distinct slots can share memory because they do not
    contain live data at the same time.

    Parameters
    ----------
    children

    Raises
    ------
    ValueError
        If `children` is empty or contains non-attachable elements
    """

    def __init__(self, children: Iterable[IOSlotBase]) -> None:
        super().__init__()
        self.children = list(children)
        self.raw = None       # type: Optional[Any]
        if not len(self.children):
            raise ValueError('empty child list')
        for child in self.children:
            if not child.attachable():
                raise ValueError('child is not attachable')
        for child in self.children:
            child.is_root = False

    def is_bound(self) -> bool:
        return self.raw is not None

    def required_bytes(self) -> int:
        return max([child.required_bytes() for child in self.children])

    def _allocate_host(self, context: AbstractContext) -> HostArray:
        return HostArray((self.required_bytes(),), np.uint8, context=context)

    def _allocate(self, allocator: AbstractAllocator[_RB],
                  raw: Optional[_RB] = None, *, bind: bool = True) -> _RB:
        if raw is None:
            raw = allocator.allocate_raw(self.required_bytes())
        if bind:
            for child in self.children:
                child._allocate(allocator, raw, bind=bind)
            self.raw = raw
        return raw


class Operation(ABC):
    """An instance of a device operation.

    Typically one first creates a template (which contains the program code,
    and is expensive to create) and then instantiates it for use with specific
    buffers.

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
    command_queue
        Command queue for the operation
    allocator
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

    def __init__(self, command_queue: AbstractCommandQueue,
                 allocator: Optional[AbstractAllocator] = None) -> None:
        if allocator is None:
            allocator = DeviceAllocator(command_queue.context)
        elif allocator.context is not None:
            if allocator.context is not command_queue.context:
                raise ValueError('command_queue and allocator have different contexts')
        self.slots = {}            # type: Dict[str, IOSlotBase]
        self.hidden_slots = {}     # type: Dict[str, IOSlotBase]
        self.command_queue = command_queue
        self.is_root = True
        self.allocator = allocator

    def bind(self, **kwargs) -> None:
        """Bind buffers to slots by keyword.

        Each keyword argument name specifies a slot name.

        Raises
        ------
        KeyError
            if a named slots does not exist
        TypeError
            if a named slot if an alias slot
        """
        for name, buffer in kwargs.items():
            slot = self.slots[name]
            if not isinstance(slot, IOSlot):
                raise TypeError(f'Slot {slot} is not an IOSlot')
            slot.bind(buffer)

    def ensure_all_bound(self) -> None:
        """Make sure that all slots have a buffer bound, allocating if necessary."""
        for slot in self.slots.values():
            if not slot.is_bound():
                slot.allocate(self.allocator)

    def buffer(self, name: str) -> DeviceArray:
        """Retrieve the buffer bound to a slot.

        It will consult both :attr:`slots` and :attr:`hidden_slots`.

        Parameters
        ----------
        name : str
            Name of the slot to access

        Returns
        -------
        :class:`DeviceArray`
            Buffer bound to slot `name`.

        Raises
        ------
        KeyError
            If no slot with this name exists
        TypeError
            If the slot exists but it is an alias slot
        ValueError
            If the slot exists but does not yet have a buffer bound
        """
        try:
            slot = self.slots[name]
        except KeyError:
            try:
                slot = self.hidden_slots[name]
            except KeyError:
                raise KeyError('no slot named ' + name)
        if isinstance(slot, IOSlot):
            if slot.buffer is None:
                raise ValueError('slot ' + name + ' has no buffer bound')
            return slot.buffer
        else:
            raise TypeError('slot ' + name + ' is an alias slot')

    def required_bytes(self) -> int:
        """Return number of bytes of device storage required."""
        return sum([x.required_bytes() for x in self.slots.values()])

    def parameters(self) -> Mapping[str, Any]:
        """Return dictionary of configuration options for this operation."""
        return {}

    @abstractmethod
    def _run(self) -> Any:
        raise NotImplementedError('abstract base class')

    def __call__(self, **kwargs) -> Any:
        """Run the operation.

        Any slots that are not already bound will allocate new buffers.
        Keyword arguments are passed to :meth:`bind`.
        """
        self.bind(**kwargs)
        self.ensure_all_bound()
        return self._run()


class OperationSequence(Operation):
    """Convenience class for an operation that is built up of smaller named operations.

    Slots are mapped to share data between these smaller operations.
    Initially, each slot named *slot* in a child named *op* is remapped to a
    parent slot named *op*:*slot*. After this, each set provided in `compounds`
    is removed from the slots and combined into a single compound slot.
    Finally, each set in `aliases` is removed and combined into an alias slot.

    For both `compounds` and `aliases`, if a child does not exist, it is
    skipped, and if none of the children exist, no action is taken.

    Parameters
    ----------
    command_queue
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

    def __init__(self, command_queue: AbstractCommandQueue,
                 operations: Iterable[Tuple[str, Operation]],
                 compounds: Optional[Mapping[str, Iterable[str]]] = None,
                 aliases: Optional[Mapping[str, Iterable[str]]] = None,
                 allocator: Optional[AbstractAllocator] = None) -> None:
        super().__init__(command_queue, allocator)
        self.operations = OrderedDict(operations)
        for (name, operation) in self.operations.items():
            if operation.command_queue is not command_queue:
                raise ValueError('child has a different command queue to the parent')
            if not operation.is_root:
                raise ValueError('child already has another parent')
            for (slot_name, slot) in operation.slots.items():
                self.slots[name + ':' + slot_name] = slot
        if compounds is not None:
            for (name, child_names) in compounds.items():
                children = self._extract_slots(child_names, False)
                if children:
                    for child in children:
                        if not isinstance(child, IOSlot):
                            raise TypeError(f'Children of {name} must all be IOSlots')
                    self.slots[name] = CompoundIOSlot(cast(Iterable[IOSlot], children))
        if aliases is not None:
            for (name, child_names) in aliases.items():
                children = self._extract_slots(child_names, True)
                if children:
                    self.slots[name] = AliasIOSlot(children)
        for operation in self.operations.values():
            operation.is_root = False

    def _extract_slots(self, names: Iterable[str], add_to_hidden: bool) -> List[IOSlotBase]:
        """Remove and return the slots with the given names."""
        ans = []      # type: List[IOSlotBase]
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

    def _run(self) -> None:
        for operation in self.operations.values():
            operation()


def _slot_label(slot: IOSlotBase, name: str, hide_detail: bool) -> str:
    """Produce contents of HTML label for a slot.

    If `hide_detail` is true, it will not show the shapes and dtype. This is used when
    the details will appear on a parent.
    """
    if isinstance(slot, AliasIOSlot):
        return f'<b>{name}</b><br/><font color="red">alias slot</font>'
    elif hide_detail:
        return f'<b>{name}</b>'
    else:
        assert isinstance(slot, IOSlot)
        shape = '×'.join(str(x) for x in slot.shape)
        if not shape:
            shape = 'scalar'
        padded_shape = '×'.join(str(x) for x in slot.required_padded_shape())
        dtype = str(np.dtype(slot.dtype))
        return '<b>{name}</b><br/>{shape}<br/>{padded_shape}<br/>{dtype}'.format(
            name=name, shape=shape, padded_shape=padded_shape, dtype=dtype)


def _visualize_operation(
        g: 'graphviz.Graph',
        operation: Operation,
        path: Tuple[str, ...],
        slot_map: MutableMapping[IOSlotBase, str],
        no_detail_slots: Set[IOSlotBase],
        edges: Set[Tuple[IOSlotBase, IOSlotBase]]) -> None:
    """Recursive implementation of :func:`visualize_operation`.

    Parameters
    ----------
    g
        Graph or subgraph to be updated with the operation. If the operation has
        children, it will be rendered as a subgraph, otherwise as a single slot
        with ports for the slots.
    operation
        Operation to add to the graph
    path
        Hierarchical name of `operation`. The last component is used for visual
        display, and the rest for generating unique node labels.
    slot_map
        Mapping from slots to node names (or port names). It is updated in place
        with all the slots in this operation and its descendants.
    no_detail_slots
        Slots that should not show shape and dtype details. It is updated in
        place with any descendant slots of this node.
    edges
        Edges to draw, as tuples of two :class:`IOSlotBase`s.
    """
    all_slots = {**operation.slots, **operation.hidden_slots}

    if isinstance(operation, OperationSequence):
        # Graphviz treats subgraphs with the prefix 'cluster_' specially. We
        # only want this for real subgraphs, not the root.
        node_name = 'cluster_' + '!'.join(path) if len(path) > 1 else 'root'
        for slot_name, slot in all_slots.items():
            if isinstance(slot, CompoundIOSlot):
                for child_slot in slot.children:
                    no_detail_slots.add(child_slot)
        with g.subgraph(name=node_name) as sub:
            sub.attr('graph', label=path[-1])
            for child_name, child in operation.operations.items():
                _visualize_operation(sub, child, path + (child_name,),
                                     slot_map, no_detail_slots, edges)
            for slot_name, slot in list(all_slots.items()):
                if slot not in slot_map:     # It's already represented in the child
                    # ":" has special meaning to graphviz, so we replace it with "+".
                    full_name = 'slot_{}_{}'.format('!'.join(path), slot_name.replace(':', '+'))
                    slot_map[slot] = full_name
                    slot_label = '<' + _slot_label(slot, slot_name, slot in no_detail_slots) + '>'
                    sub.node(full_name, label=slot_label, shape='box')
                else:
                    del all_slots[slot_name]
    else:
        node_name = 'op_' + '!'.join(path)
        label = io.StringIO()
        label.write('<<table border="1" cellborder="0" rows="*" columns="*">'
                    '<tr><td colspan="{n_slots}">{name}</td></tr><tr>'
                    .format(name=path[-1], n_slots=len(operation.slots)))
        for slot_name, slot in operation.slots.items():
            label.write('<td port="{name}"><font point-size="9"><i>{name}</i></font></td>'
                        .format(name=slot_name))
            slot_map[slot] = node_name + ':' + slot_name.replace(':', '+')
        label.write('</tr></table>>')
        g.node(node_name, label=label.getvalue(), shape='plain')

    for slot in all_slots.values():
        if hasattr(slot, 'children'):
            children = getattr(slot, 'children')  # type: Sequence[IOSlotBase]
            for child2 in children:
                edges.add((child2, slot))


def visualize_operation(operation: Operation, filename: str) -> None:
    """Write a visualization of an :class:`Operation` to file.

    This requires the `graphviz package`_ to be installed.

    .. _graphviz package: https://graphviz.readthedocs.io/en/stable/

    Parameters
    ----------
    operation
        Operation or operation sequence to visualize
    filename
        Base filename to write. It should end in ``.gv``, and a rendered
        PDF will automatically be produced with a filename ending in
        ``.gv.pdf``.
    """
    from graphviz import Graph

    g = Graph()
    g.attr('graph', clusterrank='local', ranksep='3', compound='yes')
    slot_map = {}        # type: Dict[IOSlotBase, str]
    edges = set()        # type: Set[Tuple[IOSlotBase, IOSlotBase]]
    _visualize_operation(g, operation, ('root',), slot_map, set(), edges)
    for a, b in edges:
        g.edge(slot_map[a], slot_map[b])
    g.render(filename)
