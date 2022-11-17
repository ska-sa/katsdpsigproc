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

"""Tests for :mod:`katsdpsigproc.resource`."""

import time
import asyncio
import queue
import logging
from typing import List

import pytest

from katsdpsigproc import resource
from katsdpsigproc.abc import AbstractEvent


class TestWaitUntil:
    async def test_result(self) -> None:
        """wait_until returns before the timeout if a result is set."""
        future = asyncio.Future()     # type: asyncio.Future[int]
        loop = asyncio.get_event_loop()
        loop.call_later(0.1, future.set_result, 42)
        result = await resource.wait_until(future, loop.time() + 1000000)
        assert result == 42

    async def test_already_set(self) -> None:
        """wait_until returns if a future has a result set before the call."""
        future = asyncio.Future()     # type: asyncio.Future[int]
        future.set_result(42)
        loop = asyncio.get_event_loop()
        result = await resource.wait_until(future, loop.time() + 1000000)
        assert result == 42

    async def test_exception(self) -> None:
        """wait_until rethrows an exception set on the future."""
        future = asyncio.Future()     # type: asyncio.Future[int]
        loop = asyncio.get_event_loop()
        loop.call_later(0.1, future.set_exception, ValueError('test'))
        with pytest.raises(ValueError):
            await resource.wait_until(future, loop.time() + 1000000)

    async def test_timeout(self) -> None:
        """wait_until throws `asyncio.TimeoutError` if it times out, and cancels the future."""
        future = asyncio.Future()     # type: asyncio.Future[int]
        loop = asyncio.get_event_loop()
        with pytest.raises(asyncio.TimeoutError):
            await resource.wait_until(future, loop.time() + 0.01)
        assert future.cancelled()

    async def test_shield(self) -> None:
        """wait_until does not cancel the future if it is wrapped in shield."""
        future = asyncio.Future()     # type: asyncio.Future[int]
        loop = asyncio.get_event_loop()
        with pytest.raises(asyncio.TimeoutError):
            await resource.wait_until(asyncio.shield(future), loop.time() + 0.01)
        assert not future.cancelled()


class DummyEvent(AbstractEvent):
    """Dummy version of katsdpsigproc.accel event.

    The :meth:`wait` method just sleeps for a small time and then appends the
    event to a queue.
    """

    def __init__(self, completed: queue.Queue) -> None:
        self.complete = False
        self.completed = completed

    def wait(self) -> None:
        if not self.complete:
            time.sleep(0.1)
            self.completed.put(self)
            self.complete = True

    def time_since(self, prior_event: 'DummyEvent') -> float:
        return 0.0

    def time_till(self, next_event: 'DummyEvent') -> float:
        return 0.0


class TestResource:
    def setup_method(self) -> None:
        self.completed = queue.Queue()      # type: queue.Queue[resource.ResourceAllocation[int]]

    async def _run_frame(self, acq: resource.ResourceAllocation[int],
                         event: AbstractEvent) -> None:
        with acq as value:
            assert value == 42
            await acq.wait_events()
            acq.ready([event])
            self.completed.put(acq)

    async def test_wait_events(self) -> None:
        """Test :meth:`.ResourceAllocation.wait_events`."""
        r = resource.Resource(42)
        a0 = r.acquire()
        a1 = r.acquire()
        e0 = DummyEvent(self.completed)
        e1 = DummyEvent(self.completed)
        run1 = asyncio.ensure_future(self._run_frame(a1, e1))
        run0 = asyncio.ensure_future(self._run_frame(a0, e0))
        await run0
        await run1
        order = []
        try:
            while True:
                order.append(self.completed.get_nowait())
        except queue.Empty:
            pass
        assert order == [a0, e0, a1]

    async def test_context_manager_exception(self) -> None:
        """Test using :class:`resource.ResourceAllocation` as a \
        context manager when an error is raised."""
        r = resource.Resource(None)
        a0 = r.acquire()
        a1 = r.acquire()
        with pytest.raises(RuntimeError):
            with a0:
                await a0.wait_events()
                raise RuntimeError('test exception')
        with pytest.raises(RuntimeError):
            with a1:
                await a1.wait_events()
                a1.ready()

    async def test_context_manager_no_ready(self, caplog) -> None:
        """Test using :class:`resource.ResourceAllocation` as a context \
        manager when the user does not call :meth:`.ResourceAllocation.ready`."""
        with caplog.at_level(logging.WARNING, logger='katsdpsigproc.resource'):
            r = resource.Resource(None)
            a0 = r.acquire()
            with a0:
                pass
        assert caplog.record_tuples == [(
            'katsdpsigproc.resource',
            logging.WARNING,
            'Resource allocation was not explicitly made ready'
        )]


class TestJobQueue:
    @pytest.fixture(autouse=True)
    def setup(self, event_loop) -> None:
        self.jobs = resource.JobQueue()
        self.finished = [event_loop.create_future()
                         for i in range(5)]      # type: List[asyncio.Future[int]]
        self.unfinished = [event_loop.create_future()
                           for i in range(5)]    # type: List[asyncio.Future[int]]
        for i, future in enumerate(self.finished):
            future.set_result(i)

    def test_clean(self) -> None:
        self.jobs.add(self.finished[0])
        self.jobs.add(self.unfinished[0])
        self.jobs.add(self.finished[1])
        self.jobs.clean()
        assert len(self.jobs._jobs) == 2

    async def _finish(self) -> None:
        for i, future in enumerate(self.unfinished):
            await asyncio.sleep(0.02)
            future.set_result(i)

    async def test_finish(self) -> None:
        self.jobs.add(self.finished[0])
        self.jobs.add(self.unfinished[0])
        self.jobs.add(self.unfinished[1])
        self.jobs.add(self.unfinished[2])
        finisher = asyncio.ensure_future(self._finish())
        await self.jobs.finish(max_remaining=1)
        assert self.unfinished[0].done()
        assert self.unfinished[1].done()
        assert not self.unfinished[2].done()
        assert len(self.jobs) == 1
        await finisher

    def test_nonzero(self) -> None:
        assert not self.jobs
        self.jobs.add(self.finished[0])
        assert self.jobs

    def test_len(self) -> None:
        assert len(self.jobs) == 0
        self.jobs.add(self.finished[0])
        self.jobs.add(self.unfinished[1])
        self.jobs.add(self.finished[1])
        assert len(self.jobs) == 3

    def test_contains(self) -> None:
        assert self.finished[0] not in self.jobs
        self.jobs.add(self.finished[0])
        assert self.finished[0] in self.jobs
        assert self.finished[1] not in self.jobs
