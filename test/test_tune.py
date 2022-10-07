################################################################################
# Copyright (c) 2014-2022, National Research Foundation (SARAO)
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

"""Tests for :mod:`katsdpsigproc.tune`."""

import os
import threading
from unittest import mock
from typing import Any, Generator, Mapping, NoReturn

import pytest
import sqlite3

from katsdpsigproc import tune
from katsdpsigproc.abc import AbstractContext


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
    with pytest.raises(CustomError, match=r'^x = 3$') as exc_info:
        tune.autotune(generate_raise, x=[1, 2, 3])
    # Check that the traceback refers to the original site, not
    # where it was re-raised
    assert exc_info.traceback[-1].name == 'generate_raise'


class TestAutotuner:
    """Tests for the `autotuner` decorator.

    We use mocking to substitute an in-memory database, which is made to
    persist for the test instead of being closed each time the decorator is
    called.
    """

    autotune_mock = mock.Mock()

    @pytest.fixture
    def conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(':memory:')
        yield conn
        conn.close()

    @pytest.fixture(autouse=True)
    def _prepare_autotune_mock(self) -> Generator[None, None, None]:
        # The autotune method has to be a class method due to magic in the
        # autotuner decorator, and hence autotune_mock has to be at class
        # scope. But we want it reset for each test.
        self.autotune_mock.reset()
        yield
        self.autotune_mock.reset()

    @classmethod
    @tune.autotuner(test={'a': 3, 'b': -1})
    def autotune(cls, context: AbstractContext, param: str) -> mock.Mock:
        return cls.autotune_mock(context, param)

    @mock.patch('katsdpsigproc.tune._close_db')
    @mock.patch('katsdpsigproc.tune._open_db')
    def test(self, open_db_mock: mock.Mock, close_db_mock: mock.Mock,
             conn: sqlite3.Connection) -> None:
        open_db_mock.return_value = conn
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
    """Tests for device fallback
    + autotune a set of values to a function, storing results as usual in the sql database
    + amend the table to point to a similar but different device
    + check that the autotune is not called agian and uses the cached version
    """
    autotune_mock = mock.Mock()

    @pytest.fixture
    def conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(':memory:')
        yield conn
        conn.close()

    @pytest.fixture(autouse=True)
    def _prepare_autotune_mock(self) -> Generator[None, None, None]:
        self.autotune_mock.reset()
        yield
        self.autotune_mock.reset()

    @classmethod
    @tune.autotuner(test={'a': 3, 'b': -1})
    def autotune(cls, context: AbstractContext, param: str) -> Mapping[str, Any]:
        generate = mock.Mock()
        return tune.autotune(generate, wgs=[32])

    def test(self, query_mock: mock.Mock, fetch_mock: mock.Mock) -> None:
        context = mock.NonCallableMock()
        context.device.driver_version = 'CUDA'
        context.device.platform_name = 'CUDA:1000 Driver:10010'
        context.device.name = 'GeForce GTX TITAN X'
        self.autotune_mock.return_value = {'a': 1, 'b': 2}
        self.autotune(context, 'WGS = [32]')
        context.device.name = 'Another_device'
        self.autotune(context, 'WGS = [32]')
        self.autotune_mock.assert_called_once_with(context, 'WGS = [32]')


@mock.patch.dict(os.environ, {"KATSDPSIGPROC_TUNE_MATCH": "nearest"})
def test_device_fallback() -> None:
    cartesian = []
    cartesian_lock = threading.Lock()

    def generate(x_dim, y_dim):
        with cartesian_lock:
            cartesian.append((x_dim, y_dim))
        return lambda scoring_function: (1 + 16 * 1024) - (x_dim * y_dim) \
            if (x_dim * y_dim) < (1 + 16 * 1024) else (1 + 6 * 1024)

    opt = tune.autotune(generate, time_limit=0.1, x_dim=[2, 8, 64, 128], y_dim=[4, 16, 32, 256])
    assert opt == {'x_dim': 64, 'y_dim': 256}
