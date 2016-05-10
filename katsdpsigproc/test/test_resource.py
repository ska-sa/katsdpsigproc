"""Tests for :mod:`katsdpsigproc.resource`."""

from katsdpsigproc import resource
import functools
import decorator
import trollius
import time
import Queue
from trollius import From
from nose.tools import *
import mock


@nottest
def async_test(func):
    func = trollius.coroutine(func)
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.loop.run_until_complete(func(self, *args, **kwargs))
    return wrapper


class TestWaitUntil(object):
    def setup(self):
        self.loop = trollius.get_event_loop_policy().new_event_loop()

    def teardown(self):
        self.loop.close()

    @async_test
    def test_result(self):
        """wait_until returns before the timeout if a result is set"""
        future = trollius.Future(loop=self.loop)
        self.loop.call_later(0.1, future.set_result, 42)
        result = yield From(resource.wait_until(future, self.loop.time() + 1000000, loop=self.loop))
        assert_equal(42, result)

    @async_test
    def test_already_set(self):
        """wait_until returns if a future has a result set before the call"""
        future = trollius.Future(loop=self.loop)
        future.set_result(42)
        result = yield From(resource.wait_until(future, self.loop.time() + 1000000, loop=self.loop))
        assert_equal(42, result)

    @async_test
    def test_exception(self):
        """wait_until rethrows an exception set on the future"""
        future = trollius.Future(loop=self.loop)
        self.loop.call_later(0.1, future.set_exception, ValueError('test'))
        with assert_raises(ValueError):
            yield From(resource.wait_until(future, self.loop.time() + 1000000, loop=self.loop))

    @async_test
    def test_timeout(self):
        """wait_until throws `trollius.TimeoutError` if it times out, and cancels the future"""
        future = trollius.Future(loop=self.loop)
        with assert_raises(trollius.TimeoutError):
            yield From(resource.wait_until(future, self.loop.time() + 0.01, loop=self.loop))
        assert_true(future.cancelled())

    @async_test
    def test_shield(self):
        """wait_until does not cancel the future if it is wrapped in shield"""
        future = trollius.Future(loop=self.loop)
        with assert_raises(trollius.TimeoutError):
            yield From(resource.wait_until(trollius.shield(future), self.loop.time() + 0.01, loop=self.loop))
        assert_false(future.cancelled())


class DummyEvent(object):
    """Dummy version of katsdpsigproc.accel event, whose wait method just
    sleeps for a small time and then appends the event to a queue.
    """
    def __init__(self, completed):
        self.complete = False
        self.completed = completed

    def wait(self):
        if not self.complete:
            time.sleep(0.1)
            self.completed.put(self)
            self.complete = True


class TestResource(object):
    def setup(self):
        self.loop = trollius.get_event_loop_policy().new_event_loop()
        self.completed = Queue.Queue()

    def teardown(self):
        self.loop.close()

    @trollius.coroutine
    def _run_frame(self, acq, event):
        with acq as value:
            assert_equal(42, value)
            yield From(acq.wait_events())
            acq.ready([event])
            self.completed.put(acq)

    @async_test
    def test_wait_events(self):
        """Test :meth:`resource.ResourceAcquisition.wait_events`"""
        r = resource.Resource(42, loop=self.loop)
        a0 = r.acquire()
        a1 = r.acquire()
        e0 = DummyEvent(self.completed)
        e1 = DummyEvent(self.completed)
        run1 = trollius.async(self._run_frame(a1, e1), loop=self.loop)
        run0 = trollius.async(self._run_frame(a0, e0), loop=self.loop)
        yield From(run0)
        yield From(run1)
        order = []
        try:
            while True:
                order.append(self.completed.get_nowait())
        except Queue.Empty:
            pass
        assert_equal(order, [a0, e0, a1])

    @async_test
    def test_context_manager_exception(self):
        """Test using :class:`resource.ResourceAcquisition` as a context
        manager when an error is raised.
        """
        r = resource.Resource(None, loop=self.loop)
        a0 = r.acquire()
        a1 = r.acquire()
        with assert_raises(RuntimeError):
            with a0:
                yield From(a0.wait_events())
                raise RuntimeError('test exception')
        with assert_raises(trollius.CancelledError):
            with a1:
                yield From(a1.wait_events())
                a1.ready()

    @mock.patch('katsdpsigproc.resource._logger')
    @async_test
    def test_context_manager_no_ready(self, mock_logging):
        """Test using :class:`resource.ResourceAllocation` as a context
        manager when the user does not call
        :meth:`~resource.ResourceAllocation.ready`.
        """
        r = resource.Resource(None, loop=self.loop)
        a0 = r.acquire()
        with a0:
            pass
        mock_logging.warn.assert_called_once_with(
                'Resource allocation was not explicitly made ready')
