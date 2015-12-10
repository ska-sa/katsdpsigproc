import sys
import traceback
from nose.tools import assert_equal
import unittest2 as unittest
import mock
import sqlite3
import threading
from .. import tune

# nose.tools uses unittest rather than unittest2 for assertRaises, which
# means that it doesn't get the context manager version on Python 2.6
# (see https://github.com/nose-devs/nose/issues/25)
# We import from unittest2 directly to fix this, using the same trick
# as nose does.
class Dummy(unittest.TestCase):
    def nop():
        pass
_dummy = Dummy('nop')
assert_raises = _dummy.assertRaises

def test_autotune_basic():
    received = []
    received_lock = threading.Lock()
    def generate(a, b):
        with received_lock:
            received.append((a, b))
        return lambda iters: a * b

    best = tune.autotune(generate, time_limit=0.001, a=[1, 2], b=[7, 3])
    # Autotuning is parallel, so we can't assert anything about the order
    assert_equal([(1, 3), (1, 7), (2, 3), (2, 7)], sorted(received))
    assert_equal({'a': 1, 'b': 3}, best)

def test_autotune_empty():
    with assert_raises(ValueError):
        tune.autotune(lambda x, y: lambda iters: 0, x=[1, 2], y=[])

class CustomError(RuntimeError):
    pass

def test_autotune_some_raise():
    def generate(x):
        if x == 1:
            raise CustomError('x = 1')
        def measure(iters):
            if x == 3:
                raise CustomError('x = 3')
            return -x
        return measure
    best = tune.autotune(generate, x=[0, 1, 2, 3])
    assert_equal({'x': 2}, best)

def generate_raise(x):
    raise CustomError('x = {0}'.format(x))

def test_autotune_all_raise():
    exc_value = None
    with assert_raises(CustomError):
        exc_value = None
        try:
            tune.autotune(generate_raise, x=[1, 2, 3])
        except CustomError as e:
            exc_value = e
            raise

    assert_equal('x = 3', str(exc_value))
    # Check that the traceback refers to the original site, not
    # where it was re-raised
    frames = traceback.extract_tb(sys.exc_info()[2])
    assert_equal('generate_raise', frames[-1][2])

class TestAutotuner(object):
    """Tests for the `autotuner` decorator. We use mocking to substitute an
    in-memory database, which is made to persist for the test instead of
    being closed each time the decorator is called."""

    def setup(self):
        self.conn = sqlite3.connect(':memory:')
        self.__class__.autotune_mock = mock.Mock()

    def teardown(self):
        self.conn.close()
        del self.__class__.autotune_mock

    @classmethod
    @tune.autotuner(test={'a': 3, 'b': -1})
    def autotune(cls, context, param):
        return cls.autotune_mock(context, param)

    @mock.patch('katsdpsigproc.tune._close_db')
    @mock.patch('katsdpsigproc.tune._open_db')
    def test(self, open_db_mock, close_db_mock):
        open_db_mock.return_value = self.conn
        tuning = {'a': 1, 'b': 2}
        context = mock.NonCallableMock()
        context.device.driver_version = 'mock version'
        context.device.platform_name = 'mock platform'
        context.device.name = 'mock device'
        self.autotune_mock.return_value = tuning
        ret1 = self.autotune(context, 'xyz')
        ret2 = self.autotune(context, 'xyz')
        self.autotune_mock.assert_called_once_with(context, 'xyz')
        assert_equal(tuning, ret1)
        assert_equal(tuning, ret2)
