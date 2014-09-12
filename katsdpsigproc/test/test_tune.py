import sys
import traceback
from nose.tools import assert_equal, assert_raises
from .. import tune

def test_autotune_basic():
    received = []
    def measure(a, b):
        received.append((a, b))
        return a * b

    best = tune.autotune(measure, a=[1, 2], b=[7, 3])
    assert_equal([(1, 7), (1, 3), (2, 7), (2, 3)], received)
    assert_equal({'a': 1, 'b': 3}, best)

def test_autotune_empty():
    with assert_raises(ValueError):
        tune.autotune(lambda x, y: 0, x=[1, 2], y=[])

class CustomError(RuntimeError):
    pass

def test_autotune_some_raise():
    def measure(x):
        if x == 1:
            raise CustomError('x = 1')
        return -x
    best = tune.autotune(measure, x=[0, 1, 2])
    assert_equal({'x': 2}, best)

def measure_raise(x):
    raise CustomError('x = {0}'.format(x))

def test_autotune_all_raise():
    exc_info = None
    with assert_raises(CustomError):
        exc_value = None
        try:
            tune.autotune(measure_raise, x=[1,2,3])
        except CustomError as e:
            exc_value = e
            raise

    assert_equal('x = 3', str(exc_value))
    # Check that the traceback refers to the original site, not
    # where it was re-raised
    frames = traceback.extract_tb(sys.exc_info()[2])
    assert_equal('measure_raise', frames[-1][2])
