################################################################################
# Copyright (c) 2014-2020, National Research Foundation (SARAO)
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

"""Tools for autotuning algorithm parameters and caching the results.

Note that this is intended to apply only to parameters that affect
performance, such as work-group sizes, and not the outputs.

The design is that each computation class that implements autotuning will
provide a class method (typically called `autotune`) that takes a context and
user-supplied parameters (related to the problem rather than the
implementation), and returns a dictionary of values. A decorator is applied to
the autotuning method to cause the results to be cached.

The class may define an autotune_version class attribute to version the table.

At present, caching is implemented in an sqlite3 database. There is a table
corresponding to each autotuning method. The table has the following columns:

- `device_name`: string, name for the compute device
- `device_platform`: string, name for the compute device's platform
- `device_version`: version string for the driver
- `arg_*`: for each parameter passed to the autotuner
- `value_*`: for each value returned from the autotuner

The database is stored in the user cache directory.
"""

import itertools
import inspect
import os
import os.path
import enum
import time
import logging
import multiprocessing
import concurrent.futures
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    cast,
)

import appdirs
import sqlite3
import numpy as np
from decorator import decorator
from typing_extensions import Protocol

from .abc import AbstractContext, AbstractTuningCommandQueue


_logger = logging.getLogger(__name__)
_ScoreFunc = Callable[[int], float]
_T = TypeVar("_T")

KATSDPSIGPROC_TUNE_MATCH = os.getenv("KATSDPSIGPROC_TUNE_MATCH", "exact")
if KATSDPSIGPROC_TUNE_MATCH not in ["exact", "nearest"]:
    _logger.debug("KATSDPSIGPROC_TUNE_MATCH environment variable not one of [ exact | nearest ]: "
                  "setting to 'exact'.")
    KATSDPSIGPROC_TUNE_MATCH = "exact"


class _TuningFunc(Protocol):
    @property
    def __name__(self) -> str:
        ...

    def __call__(
        self, __cls: Type, __context: AbstractContext, *args: Any, **kwargs: Any
    ) -> Mapping[str, Any]:
        ...


def adapt_value(value: Any) -> Any:
    """Convert `value` to a type that can be used in sqlite3.

    This is not done through the sqlite3 adapter interface, because that is
    global rather than per-connection. This also only applies to lookup keys,
    not results, because it is not a symmetric relationship.
    """
    if isinstance(value, type) or isinstance(value, np.dtype):
        return repr(value)
    elif isinstance(value, enum.Enum):
        return value.name
    return value


def _db_keys(fn: _TuningFunc, args: Sequence, kwargs: Mapping) -> Dict[str, Any]:
    """Use the arguments passed to an autotuning function to generate a database key.

    .. seealso::

       :func:`autotuner`
    """
    # Extract the arguments passed to the wrapped function, by name
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    keys = {
        "arg_" + name: adapt_value(value)
        for (name, value) in itertools.islice(bound.arguments.items(), 2, None)
    }

    # Add information about the device
    device = args[1].device
    keys["device_name"] = device.name
    keys["device_platform"] = device.platform_name
    keys["device_version"] = device.driver_version
    return keys


def _query(
    conn: sqlite3.Connection, tablename: str, keys: Mapping[str, Any]
) -> Optional[sqlite3.Row]:
    query = f"SELECT * FROM {tablename} WHERE"
    query_args = []
    first = True
    for key, value in keys.items():
        if not first:
            query += " AND"
        first = False
        query += f" {key}=?"
        query_args.append(value)
    cursor = conn.cursor()
    cursor.execute(query, query_args)
    row = cursor.fetchone()
    return row


def _fetch(
    conn: sqlite3.Connection, tablename: str, keys: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """Fetch a cached record from the database.

    If the KATSDPSIGPROC_TUNE_MATCH environment variable is set to "nearest", and
    an exact record is not found, return the nearest match, by ignoring in turn
    device driver version, then device platform, then device name, or None
    otherwise (triggering autotuning).

    Parameters
    ----------
    conn
        Database connection
    tablename
        Name of the table to query
    keys
        Keys and values for the query
    """
    try:
        row = _query(conn, tablename, keys)
        if row is None:
            if KATSDPSIGPROC_TUNE_MATCH == "nearest":
                # Find the nearest autotune match -- if it exists -- by first ignoring
                # device driver version, then device platform, then device name.
                tune_keys = dict(keys)
                for key in ["device_version", "device_platform", "device_name"]:
                    _logger.debug("Retrying query by ignoring %s", key)
                    del tune_keys[key]
                    try:
                        row = _query(conn, tablename, tune_keys)
                    except sqlite3.Error:
                        _logger.debug("Query '%s' failed", exc_info=True)
                        pass
                    if row is not None:
                        break

        if row is not None:
            ans = {}
            for colname in row.keys():
                if colname.startswith("value_"):
                    # Truncate the 'value_' prefix
                    ans[colname[6:]] = row[colname]
            return ans
    except sqlite3.Error:
        # This could happen if the table does not exist yet
        _logger.debug("Query failed", exc_info=True)
        pass
    return None


def _create_table(
    conn: sqlite3.Connection,
    tablename: str,
    keys: Mapping[str, Any],
    values: Mapping[str, Any],
) -> None:
    command = f"CREATE TABLE IF NOT EXISTS {tablename} ("
    for name in itertools.chain(keys.keys(), values.keys()):
        command += name + " NOT NULL, "
    command += "PRIMARY KEY ({}) ON CONFLICT REPLACE)".format(", ".join(keys.keys()))
    conn.execute(command)


def _save(
    conn: sqlite3.Connection,
    tablename: str,
    keys: Mapping[str, Any],
    values: Mapping[str, Any],
) -> None:
    """ Write cached result into the table, creating it if it does not already exist."""
    _create_table(conn, tablename, keys, values)
    # Combine all fields
    entries = dict(keys)
    entries.update(values)
    command = "INSERT OR REPLACE INTO {}({}) VALUES ({})".format(
        tablename, ", ".join(entries.keys()), ", ".join(["?" for x in entries.keys()])
    )
    # Start transaction
    with conn:
        cursor = conn.cursor()
        cursor.execute(command, list(entries.values()))


def _open_db() -> sqlite3.Connection:
    cache_dir = appdirs.user_cache_dir("katsdpsigproc", "ska-sa")
    try:
        os.makedirs(cache_dir)
    except OSError:
        # This happens if the directory already exists. If we failed
        # to create it, the database open will fail.
        pass

    cache_file = os.path.join(cache_dir, "tuning.db")
    conn = sqlite3.connect(cache_file)
    return conn


def _close_db(conn: sqlite3.Connection) -> None:
    """Close a database.

    This is split into a separate function for the benefit of testing, because
    the close member of the object itself is read-only and hence cannot be
    patched.
    """
    conn.close()


def autotuner_impl(
    test: Mapping[str, Any], fn: _TuningFunc, *args: Any, **kwargs: Any
) -> Mapping[str, Any]:
    """Implement :func:`autotuner`.

    It is split into a separate function so that mocks can patch it.
    """
    cls = args[0]
    classname = f"{cls.__module__}.{cls.__name__}.{fn.__name__}"
    tablename = (
        classname.replace(".", "_") + "__" + str(getattr(cls, "autotune_version", 0))
    )
    keys = _db_keys(fn, args, kwargs)

    conn = _open_db()
    conn.row_factory = sqlite3.Row
    try:
        ans = _fetch(conn, tablename, keys)
        if ans is None:
            # Nothing found in the database, so we need to tune now
            _logger.info("Performing autotuning for %s with key %s", classname, keys)
            ans = fn(*args, **kwargs)
            values = {"value_" + key: value for (key, value) in ans.items()}
            _save(conn, tablename, keys, values)
        else:
            _logger.debug("Autotuning cache hit for %s with key %s", classname, keys)
    finally:
        _close_db(conn)
    return ans


def autotuner(test: Mapping[str, Any]) -> Callable[[_T], _T]:
    r"""Decorate a function to make it an autotuning function that caches the result.

    The function must take a class and a context as the first two arguments.
    The remaining arguments form a cache key, along with properties of the
    device and the name of the function.

    Every argument to the function must have a name, which implies that the
    \*args construct may not be used.

    Parameters
    ----------
    test : dictionary
        A value that will be returned by :func:`stub_autotuner`.
    """

    @decorator
    def autotuner(fn: _TuningFunc, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        r"""Decorate a function to make it an autotuning function that caches the result.

        The function must take a class and a context as the first two
        arguments. The remaining arguments form a cache key, along with
        properties of the device and the name of the function.

        Every argument to the function must have a name, which implies that the
        \*args construct may not be used.
        """
        return autotuner_impl(test, fn, *args, **kwargs)

    # decorator module doesn't have type annotations, so we need to force it
    return cast(Callable[[_T], _T], autotuner)


def force_autotuner(
    test: Mapping[str, Any], fn: _TuningFunc, *args: Any, **kwargs: Any
) -> Mapping[str, Any]:
    """Drop-in replacement for :func:`autotuner_impl` that does not do any caching.

    It is intended to be used with a mocking framework.
    """
    return fn(*args, **kwargs)


def stub_autotuner(
    test: Mapping[str, Any], fn: _TuningFunc, *args: Any, **kwargs: Any
) -> Mapping[str, Any]:
    """Drop-in replacement for :func:`autotuner_impl` that does not do any tuning.

    It instead returns the provided value. It is intended to be
    used with a mocking framework.
    """
    return test


def make_measure(
    queue: AbstractTuningCommandQueue, function: Callable[[], None]
) -> _ScoreFunc:
    """Generate a measurement function.

    The result can be returned by the function passed to :func:`autotune`. It
    calls `function` (with no arguments) the appropriate number of times and
    returns the averaged elapsed time as measured by `queue`.
    """

    def measure(iters):
        queue.start_tuning()
        for i in range(iters):
            function()
        return queue.stop_tuning() / iters

    return measure


def autotune(
    generate: Callable[..., Optional[_ScoreFunc]],
    time_limit: float = 0.1,
    threads: Optional[int] = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Run a number of tuning experiments and find the optimal combination of parameters.

    Each argument is a iterable. The `generate` function is passed each
    element of the Cartesian product (by keyword), and returns a callable.
    This callable is passed an iteration count, and returns a score: lower is
    better. If either `generate` or the function it returns raises an
    exception, it is suppressed. Returns a dictionary with the best combination
    of values.

    Instead of a callable, the generate function may return `None` to skip a
    configuration that it knows will be unsuitable. Throwing an exception has
    essentially the same effect (and is used in code written before returning
    `None` was allowed), but returning `None` is preferred since an exception
    just clutters the log.

    The scoring function should not do a warmup pass: that is handled by this
    function.

    Parameters
    ----------
    generate
        function that creates a scoring function
    time_limit
        amount of time to spend testing each configuration (excluding setup time)
    threads
        number of parallel calls to `generate` to make at a time (defaults to
        CPU count)

    Raises
    ------
    Exception
        if every combination throws an exception, the last exception is re-raised.
    ValueError
        if the Cartesian product is empty
    """
    opts = itertools.product(*kwargs.values())
    best = None  # type: Optional[Mapping[str, Any]]
    best_score = None  # type: Optional[float]
    exc = None  # type: Optional[Exception]
    if threads is None:
        try:
            threads = multiprocessing.cpu_count()
        except NotImplementedError:
            threads = 1
    with concurrent.futures.ThreadPoolExecutor(threads) as thread_pool:
        while True:
            batch = list(itertools.islice(opts, threads))
            if not batch:
                break
            batch_keywords = [dict(zip(kwargs.keys(), opt)) for opt in batch]
            futures = [
                thread_pool.submit(generate, **keywords) for keywords in batch_keywords
            ]
            concurrent.futures.wait(futures)
            for keywords, future in zip(batch_keywords, futures):
                try:
                    measure = future.result()
                    if measure is None:
                        continue
                    # Do a warmup pass
                    measure(1)
                    # Do an initial timing pass
                    start = time.time()
                    measure(1)
                    # Take a max to prevent divide-by-zero in very fast cases
                    elapsed = max(time.time() - start, 1e-4)
                    # Guess how many iterations we can do in the allotted time
                    iters = max(3, int(time_limit / elapsed))
                    score = measure(iters)
                    _logger.debug(
                        "Configuration %s scored %f in %d iterations",
                        keywords,
                        score,
                        iters,
                    )
                    if best_score is None or score < best_score:
                        best = keywords
                        best_score = score
                except Exception as e:
                    exc = e
                    _logger.debug(
                        "Caught exception while testing configuration %s",
                        keywords,
                        exc_info=True,
                    )
    if best is None:
        if exc is not None:
            raise exc
        else:
            raise ValueError("No options to test")
    return best
