################################################################################
# Copyright (c) 2016-2022, National Research Foundation (SARAO)
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

"""Utilities for scheduling device operations with asyncio."""

import asyncio
import logging
import collections
from typing import List, Deque, Awaitable, Iterable, Optional, Type, Generic, TypeVar
from types import TracebackType

from .abc import AbstractEvent


_T = TypeVar('_T')
_logger = logging.getLogger(__name__)


async def wait_until(future: Awaitable[_T], when: float,
                     loop: Optional[asyncio.AbstractEventLoop] = None) -> _T:
    """Like :func:`asyncio.wait_for`, but with an absolute timeout."""
    def ready(*args) -> None:
        if not waiter.done():
            waiter.set_result(None)

    if loop is None:
        loop = asyncio.get_event_loop()
    waiter = asyncio.Future(loop=loop)    # type: asyncio.Future[None]
    timeout_handle = loop.call_at(when, ready)
    # Ensure that the future is really a future, not a coroutine object
    future = asyncio.ensure_future(future, loop=loop)
    future.add_done_callback(ready)
    try:
        await waiter
        if future.done():
            return future.result()
        else:
            future.remove_done_callback(ready)
            future.cancel()
            raise asyncio.TimeoutError()
    finally:
        timeout_handle.cancel()


async def async_wait_for_events(events: Iterable[AbstractEvent],
                                loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Coroutine that waits for a list of device events."""
    def wait_for_events(events: Iterable[AbstractEvent]) -> None:
        for event in events:
            event.wait()
    if loop is None:
        loop = asyncio.get_event_loop()
    if events:
        await loop.run_in_executor(None, wait_for_events, events)


class ResourceAllocation(Generic[_T]):
    """A handle representing a future acquisition of a resource.

    There are two ways to make the acquisition current:

     1. Call :meth:`wait`, which returns a future. The result of this future
        is a list of device events that must complete before it is safe to use
        the resource; they can either be waited for on the host (for example,
        using ``run_in_executor``), or by enqueuing a device wait before
        device operations on the resource.
     2. The above steps (with a blocking host wait) can be combined using
        :meth:`wait_events`.

    Instances of this class should never be constructed directly. Instead, use
    :meth:`Resource.acquire`.

    This class implements the context manager protocol, providing the
    underlying resource as the return value. This handles some cleanup if the
    method using the resource raises an exception without releasing the
    resource cleanly.
    """

    def __init__(self,
                 start: 'asyncio.Future[List[AbstractEvent]]',
                 end: 'asyncio.Future[List[AbstractEvent]]',
                 value: _T, loop: asyncio.AbstractEventLoop) -> None:
        self._start = start
        self._end = end
        self._loop = loop
        self.value = value

    def wait(self) -> 'asyncio.Future[List[AbstractEvent]]':
        """Return a future that will be set to a list of device events to waited for."""
        return self._start

    async def wait_events(self) -> None:
        """Wait for previous use of the resource to be complete on the host.

        This is a coroutine.
        """
        events = await self._start
        await async_wait_for_events(events, loop=self._loop)

    def ready(self, events: Optional[List[AbstractEvent]] = None) -> None:
        """Indicate that we are done with the resource, and that subsequent acquirers may use it.

        Note that even if the caller decides that it doesn't need to use the
        resource, it must not call this until the resource is ready.

        Parameters
        ----------
        events
            Device events that must be waited on before the resource is ready
        """
        if events is None:
            events = []
        self._end.set_result(events)

    def __enter__(self) -> _T:
        return self.value

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        if not self._end.done():
            if exc_value is not None:
                self._end.set_exception(exc_value)
                # Prevent asyncio complaining that the exception was never
                # retrieved, because we're also propagating the exception
                # from the context manager block and it's not necessary for
                # the application to consume it a second time.
                self._end.exception()
            else:
                _logger.warning('Resource allocation was not explicitly made ready')
                self.ready()


class Resource(Generic[_T]):
    """Abstraction of a contended resource, which may exist on a device.

    Passing of ownership is done via futures. Acquiring a resource is a
    non-blocking operation that returns two futures: a future to wait for
    before use, and a future to be signalled with a result when done. The
    value of each of these futures is a (possibly empty) list of device
    events which must be waited on before more device work is scheduled.

    Parameters
    ----------
    value : object
        Underlying resource to manage
    loop
        Event loop used for asynchronous operations. If not specified, defaults
        to ``asyncio.get_event_loop()``.

    Attributes
    ----------
    value : object
        Underlying resource passed to the constructor
    """

    def __init__(self, value: _T, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if loop is None:
            loop = asyncio.get_event_loop()
        self._loop = loop
        self._future = asyncio.Future(loop=loop)  # type: asyncio.Future[List[AbstractEvent]]
        self._future.set_result([])
        self.value = value

    def acquire(self) -> ResourceAllocation[_T]:
        """Indicate intent to acquire the resource.

        This does not actually acquire the resource, but instead returns a
        handle that can be used to acquire and release it later. Acquisitions
        always occur in the order in which calls to :meth:`acquire` are made.

        See :class:`ResourceAllocation` for further details.
        """
        old = self._future
        self._future = asyncio.Future(loop=self._loop)
        return ResourceAllocation(old, self._future, self.value, loop=self._loop)


class JobQueue:
    """Maintain a list of in-flight asynchronous jobs."""

    def __init__(self) -> None:
        self._jobs = collections.deque()   # type: Deque[asyncio.Future]

    def add(self, job: Awaitable) -> None:
        """Append a job to the list.

        If `job` is a coroutine, it is automatically wrapped in a task.
        """
        self._jobs.append(asyncio.ensure_future(job))

    def clean(self) -> None:
        """Remove completed jobs from the front of the queue."""
        while self._jobs and self._jobs[0].done():
            self._jobs.popleft().result()     # Re-throws any exception

    async def finish(self, max_remaining: int = 0) -> None:
        """Wait for jobs to finish until there are at most `max_remaining` in the queue.

        This is a coroutine.
        """
        while len(self._jobs) > max_remaining:
            await self._jobs.popleft()

    def __len__(self) -> int:
        return len(self._jobs)

    def __bool__(self) -> bool:
        return bool(self._jobs)

    def __contains__(self, item: asyncio.Future) -> bool:
        return item in self._jobs


__all__ = ['wait_until', 'async_wait_for_events', 'Resource', 'ResourceAllocation', 'JobQueue']
