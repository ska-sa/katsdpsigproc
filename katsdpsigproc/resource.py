"""Utilities for scheduling device operations with trollius."""

import trollius
from trollius import From

@trollius.coroutine
def wait_until(future, when, loop=None):
    """Like :meth:`trollius.wait_for`, but with an absolute timeout."""
    def ready(*args):
        if not waiter.done():
            waiter.set_result(None)

    if loop is None:
        loop = trollius.get_event_loop()
    waiter = trollius.Future(loop=loop)
    timeout_handle = loop.call_at(when, ready)
    # Ensure the that future is really a future, not a coroutine object
    future = trollius.async(future, loop=loop)
    future.add_done_callback(ready)
    try:
        result = yield From(waiter)
        if future.done():
            raise trollius.Return(future.result())
        else:
            future.remove_done_callback(ready)
            future.cancel()
            raise trollius.TimeoutError()
    finally:
        timeout_handle.cancel()


@trollius.coroutine
def _async_wait_for_events(events, loop=None):
    def wait_for_events(events):
        for event in events:
            event.wait()
    if loop is None:
        loop = trollius.get_event_loop()
    if events:
        yield From(loop.run_in_executor(None, wait_for_events, events))


class ResourceAllocation(object):
    def __init__(self, start, end, value, loop):
        self._start = start
        self._end = end
        self._loop = loop
        self.value = value

    def wait(self):
        return self._start

    @trollius.coroutine
    def wait_events(self):
        events = yield From(self._start)
        yield From(_async_wait_for_events(events, loop=self._loop))

    def ready(self, events=None):
        if events is None:
            events = []
        self._end.set_result(events)

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self._end.done():
            if exc_type is not None:
                self._end.cancel()
            else:
                logger.warn('Resource allocation was not explicitly made ready')
                self.ready()


class Resource(object):
    """Abstraction of a contended resource, which may exist on a device.

    Passing of ownership is done via futures. Acquiring a resource is a
    non-blocking operation that returns two futures: a future to wait for
    before use, and a future to be signalled with a result when done. The
    value of each of these futures is a (possibly empty) list of device
    events which must be waited on before more device work is scheduled.
    """
    def __init__(self, value, loop=None):
        if loop is None:
            loop = trollius.get_event_loop()
        self._loop = loop
        self._future = trollius.Future(loop=loop)
        self._future.set_result([])
        self.value = value

    def acquire(self):
        old = self._future
        self._future = trollius.Future(loop=self._loop)
        return ResourceAllocation(old, self._future, self.value, loop=self._loop)


__all__ = ['wait_until', 'Resource', 'ResourceAllocation']
