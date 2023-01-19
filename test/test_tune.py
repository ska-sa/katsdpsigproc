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

import threading
from unittest import mock
from typing import Any, Generator, Mapping, NoReturn, Optional

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
    pass


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
    def autotune(cls, context: AbstractContext, param: Optional[str]) -> Mapping[str, Any]:
        return cls.autotune_mock(context, param)

    @mock.patch('katsdpsigproc.tune._close_db')
    @mock.patch('katsdpsigproc.tune._open_db')
    @mock.patch('katsdpsigproc.tune.KATSDPSIGPROC_TUNE_MATCH', 'exact')
    def test_autotune_match(self, open_db_mock: mock.Mock, close_db_mock: mock.Mock,
                            conn: sqlite3.Connection) -> None:
        ''' Tests behaviour of both autotune matching options ('exact' and 'nearest'). '''
        open_db_mock.return_value = conn
        context = mock.NonCallableMock()

        # Check that autotuning is only performed once for a particular
        # context+arg_parameters set
        tuning_1 = {'a': 1, 'b': 2}
        self.autotune_mock.return_value = tuning_1
        context.device.driver_version = 'mock version'
        context.device.platform_name = 'mock platform'
        context.device.name = 'mock device'
        ret = self.autotune(context, 'xyz')
        ret2 = self.autotune(context, 'xyz')
        self.autotune_mock.assert_called_once_with(context, 'xyz')
        assert ret == ret2 == tuning_1

        # An empty arg_parameter set should not trigger autotuning
        ret = self.autotune(context, None)
        assert ret == tuning_1

        # Check that using a different context+arg_parameters set causes tuning to be
        # re-run. This also populates the database with another set
        tuning_2 = {'a': 3, 'b': 4}
        self.autotune_mock.return_value = tuning_2
        context.device.driver_version = 'another version'
        context.device.platform_name = 'another platform'
        context.device.name = 'another device'
        ret = self.autotune(context, 'xyz')
        assert ret == tuning_2

        # Now test incremental fallback on device  [ driver version | platform | name ]
        # when KATSDPSIGPROC_TUNE_MATCH is set to 'nearest'
        with mock.patch('katsdpsigproc.tune.KATSDPSIGPROC_TUNE_MATCH', 'nearest'):
            # autotuning re-runs if arg_parameters are different
            tuning_4 = {'a': 7, 'b': 8}
            self.autotune_mock.return_value = tuning_4
            context.device.driver_version = 'mock version'
            context.device.platform_name = 'mock platform'
            context.device.name = 'mock device'
            ret = self.autotune(context, 'zzz')
            assert ret == tuning_4

            # Re-run with different configurations making sure the cached values are returned if
            # they exist in any configuration, and that autotuning does not run.
            self.autotune_mock.side_effect = RuntimeError

            context.device.driver_version = 'abc'
            context.device.platform_name = 'mock platform'
            context.device.name = 'mock device'
            ret = self.autotune(context, 'xyz')
            assert ret == tuning_1

            context.device.driver_version = 'abc'
            context.device.platform_name = 'abc'
            context.device.name = 'another device'
            ret = self.autotune(context, 'xyz')
            assert ret == tuning_2

            context.device.driver_version = 'mock version'
            context.device.platform_name = 'mock platform'
            context.device.name = 'mock device'
            ret = self.autotune(context, 'xyz')
            assert ret == tuning_1

            # Autotuning does not run for empty arg_parameter sets
            ret = self.autotune(context, None)
            assert ret == tuning_1
