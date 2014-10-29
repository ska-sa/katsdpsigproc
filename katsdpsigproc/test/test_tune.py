import sys
import traceback
from nose.tools import assert_equal
import unittest2 as unittest
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
    def generate(a, b):
        received.append((a, b))
        return lambda iters: a * b

    best = tune.autotune(generate, time_limit=0.001, a=[1, 2], b=[7, 3])
    assert_equal([(1, 7), (1, 3), (2, 7), (2, 3)], received)
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
