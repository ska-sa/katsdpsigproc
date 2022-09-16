"""Tests for :mod:`katsdpsigproc.tune`."""

import os
import sys
import traceback
import threading
from unittest import mock
from unittest.mock import MagicMock
from typing import NoReturn, Any, Mapping

import pytest
import sqlite3

from katsdpsigproc import tune
from katsdpsigproc.abc import AbstractContext


def test_autotune_understanding_lambda() -> None:
    cartesian = []
    cartesian_lock = threading.Lock()

    def generate(x_dim, y_dim):
        with cartesian_lock:
            cartesian.append((x_dim, y_dim))
        return lambda scoring_function: (1+16*1024)-(x_dim * y_dim) if (x_dim * y_dim) < (1+16*1024) else (1+16*1024)

    opt = tune.autotune(generate, time_limit=0.1, x_dim = [2,8,64,128], y_dim = [4,16,32,256])
    assert_equal({'x_dim': 64, 'y_dim': 256}, opt)


@mock.patch.dict(os.environ, {"KATSDPSIGPROC_TUNE_MATCH": "nearest"})
def test_device_fallback() -> None:
    # autotune a set of values to a function, storing results as usual in the sql database
    # amend the results to point to a similar but different device
    # check that the autotune uses the cached version
    tune.autotune(lambda x, y: lambda prod: x*y, x=[1],y=[3])


def test_autotune_basic() -> None:
    received = []
    received_lock = threading.Lock()

    def generate(a, b):
        with received_lock:
            received.append((a, b))
        return lambda iters: a * b

    best = tune.autotune(generate, time_limit=0.001, a=[1, 2], b=[7, 3])
    # Autotuning is parallel, so we can't assert anything about the order
    assert sorted(received) == [(1, 3), (1, 7), (2, 3), (2, 7)]
    assert best == {'a': 1, 'b': 3}


def test_autotune_empty() -> None:
    with pytest.raises(ValueError):
        tune.autotune(lambda x, y: lambda iters: 0, x=[1, 2], y=[])


class CustomError(RuntimeError):
    ...


def test_autotune_some_raise() -> None:
    def generate(x):
        if x == 1:
            raise CustomError('x = 1')

        def measure(iters):
            if x == 3:
                raise CustomError('x = 3')
            return -x
        return measure
    best = tune.autotune(generate, x=[0, 1, 2, 3])
    assert best == {'x': 2}


def generate_raise(x: Any) -> NoReturn:
    raise CustomError(f'x = {x}')


def test_autotune_all_raise() -> None:
    exc_value = None
    exc_info = None
    with pytest.raises(CustomError):
        exc_value = None
        try:
            tune.autotune(generate_raise, x=[1, 2, 3])
        except CustomError as e:
            exc_value = e
            exc_info = sys.exc_info()
            raise

    assert str(exc_value) == 'x = 3'
    # Check that the traceback refers to the original site, not
    # where it was re-raised
    assert exc_info is not None
    frames = traceback.extract_tb(exc_info[2])
    assert frames[-1][2] == 'generate_raise'


class TestAutotuner:
    """Tests for the `autotuner` decorator.

    We use mocking to substitute an in-memory database, which is made to
    persist for the test instead of being closed each time the decorator is
    called.
    """

    autotune_mock = mock.Mock()

    def setup(self) -> None:
        self.conn = sqlite3.connect(':memory:')
        self.autotune_mock.reset()

    def teardown(self) -> None:
        self.conn.close()
        self.autotune_mock.reset()

    @classmethod
    @tune.autotuner(test={'a': 3, 'b': -1})
    def autotune(cls, context: AbstractContext, param: str) -> mock.Mock:
        return cls.autotune_mock(context, param)

    @mock.patch('katsdpsigproc.tune._close_db')
    @mock.patch('katsdpsigproc.tune._open_db')
    def test(self, open_db_mock: mock.Mock, close_db_mock: mock.Mock) -> None:
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
        assert ret1 == tuning
        assert ret2 == tuning


@mock.patch.dict(os.environ, {"KATSDPSIGPROC_TUNE_MATCH": "nearest"})
class TestDeviceFallback:
    """Tests for device fallback"""
    generate = mock.Mock()

    @classmethod
    @tune.autotuner(test={'a': 3, 'b': -1})
    def autotune(cls, context: AbstractContext, param: str) -> Mapping[str, Any]:
        return tune.autotune(generate, wgs=[32])
    
    @mock.patch('katsdpsigproc.tune._query')
    @mock.patch('katsdpsigproc.tune._fetch')
    def test(self, query_mock: mock.Mock, fetch_mock: mock.Mock) -> None: 
        context = mock.NonCallableMock()
        context.device.driver_version = 'mock version'
        context.device.platform_name = 'mock platform'
        context.device.name = 'mock device'
        self.autotune(context, 'WGS = [32]')
        

