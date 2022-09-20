"""Utilities for testing with nosetests.

These are maintained for backwards compatibility. New code is encouraged to use
pytest instead.
"""

import functools
import inspect
import sys
from unittest import mock
from typing import Tuple, Optional, Callable, Awaitable, TypeVar

from decorator import decorator
# Don't import nose here, to allow docs to build even without nose installed.

from katsdpsigproc import accel, tune
from katsdpsigproc.abc import AbstractContext, AbstractCommandQueue

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_test_context = None           # type: Optional[AbstractContext]
_test_command_queue = None     # type: Optional[AbstractCommandQueue]
_test_initialized = False


def _prepare_device_test() -> Tuple[AbstractContext, AbstractCommandQueue]:
    from nose.plugins.skip import SkipTest

    global _test_initialized, _test_context, _test_command_queue
    if not _test_initialized:
        try:
            _test_context = accel.create_some_context(False)
            _test_command_queue = _test_context.create_command_queue()
            print("Testing on {} ({})".format(
                _test_context.device.name, _test_context.device.platform_name),
                file=sys.stderr)
        except RuntimeError:
            pass  # No devices available
        _test_initialized = True

    if not _test_context:
        raise SkipTest('CUDA/OpenCL not found')
    assert _test_command_queue is not None
    return _test_context, _test_command_queue


def _device_test_sync(test: Callable[..., _T]) -> Callable[..., _T]:
    @functools.wraps(test)
    def wrapper(*args, **kwargs) -> _T:
        context, command_queue = _prepare_device_test()
        with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner):
            args += (context, command_queue)
            # Make the context current (for CUDA contexts). Ideally the test
            # should not depend on this, but PyCUDA leaks memory if objects
            # are deleted without the context current.
            with context:
                return test(*args, **kwargs)
    return wrapper


def _device_test_async(test: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
    @functools.wraps(test)
    async def wrapper(*args, **kwargs) -> _T:
        context, command_queue = _prepare_device_test()
        with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner):
            args += (context, command_queue)
            with context:
                return await test(*args, **kwargs)
    return wrapper


def device_test(test: Callable[..., _T]) -> Callable[..., _T]:
    """Decorate an on-device test.

    It provides a context and command queue to the test, skipping it if a
    compute device is not available. It also disables autotuning, instead
    using the `test` value provided for the autotune function.

    If autotuning is desired, use :func:`force_autotune` inside (hence,
    afterwards on the decorator list) this one.
    """
    if inspect.iscoroutinefunction(test):
        return _device_test_async(test)    # type: ignore
    else:
        return _device_test_sync(test)


def cuda_test(test: _F) -> _F:
    """Skip a test if the device is not a CUDA device.

    Put this *after* :meth:`device_test`.
    """
    from nose.plugins.skip import SkipTest

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        global _test_context
        if not _test_context.device.is_cuda:
            raise SkipTest('Device is not a CUDA device')
        return test(*args, **kwargs)
    return wrapper     # type: ignore


@decorator
def force_autotune(test: Callable[..., _T], *args, **kw) -> _T:
    """Force autotuning for a test (decorator).

    It bypasses the autotuning cache so that the autotuning code always runs.
    """
    with mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.force_autotuner):
        return test(*args, **kw)


# Prevent nose from treating it as a test
device_test.__test__ = False         # type: ignore
cuda_test.__test__ = False           # type: ignore
