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

from decorator import decorator
import itertools
import inspect
import appdirs
import sqlite3
import os
import os.path
import numpy as np
import time
import logging
import multiprocessing
import concurrent.futures

_logger = logging.getLogger(__name__)

def adapt_value(value):
    """Converts `value` to a type that can be used in sqlite3. This is
    not done through the sqlite3 adapter interface, because that is global
    rather than per-connection. This also only applies to lookup keys,
    not results, because it is not a symmetric relationship.
    """
    if isinstance(value, type) or isinstance(value, np.dtype):
        return repr(value)
    return value

def _db_keys(fn, args, kwargs):
    """Uses the arguments passed to an autotuning function (see
    :function:`autotuner`) to generate a database key.
    """
    argspec = inspect.getargspec(fn)
    # Extract the arguments passed to the wrapped function, by name
    named_args = dict(kwargs)
    for i in range(2, len(args)):
        named_args[argspec.args[i]] = args[i]
    keys = dict([('arg_' + key, adapt_value(value)) for (key, value) in named_args.iteritems()])

    # Add information about the device
    device = args[1].device
    keys['device_name'] = device.name
    keys['device_platform'] = device.platform_name
    keys['device_version'] = device.driver_version
    return keys

def _query(conn, tablename, keys):
    """Fetch a cached record from the database. If the record is not found,
    it will return None.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    tablename : str
        Name of the table to query
    keys : mapping
        Keys and values for the query
    """
    try:
        query = 'SELECT * FROM {0} WHERE'.format(tablename)
        query_args = []
        first = True
        for key, value in keys.iteritems():
            if not first:
                query += ' AND'
            first = False
            query += ' {0}=?'.format(key)
            query_args.append(value)
        cursor = conn.cursor()
        cursor.execute(query, query_args)
        row = cursor.fetchone()
        if row is not None:
            ans = {}
            for colname in row.keys():
                if colname.startswith('value_'):
                    # Truncate the 'value_' prefix
                    ans[colname[6:]] = row[colname]
            return ans
    except sqlite3.Error:
        # This could happen if the table does not exist yet
        _logger.debug("Query '%s' failed", query, exc_info=True)
        pass
    return None

def _create_table(conn, tablename, keys, values):
    command = 'CREATE TABLE IF NOT EXISTS {0} ('.format(tablename)
    for name in keys.keys() + values.keys():
        command += name + ' NOT NULL, '
    command += 'PRIMARY KEY ({0}) ON CONFLICT REPLACE)'.format(', '.join(keys.keys()))
    conn.execute(command)

def _save(conn, tablename, keys, values):
    """Write cached result into the table, creating it if it does not already
    exist.
    """
    _create_table(conn, tablename, keys, values)
    # Combine all fields
    entries = dict(keys)
    entries.update(values)
    command = 'INSERT OR REPLACE INTO {0}({1}) VALUES ({2})'.format(
            tablename,
            ', '.join(entries.keys()),
            ', '.join(['?' for x in entries.keys()]))
    # Start transaction
    with conn:
        cursor = conn.cursor()
        cursor.execute(command, entries.values())

def _open_db():
    cache_dir = appdirs.user_cache_dir('katsdpsigproc', 'ska-sa')
    try:
        os.makedirs(cache_dir)
    except OSError:
        # This happens if the directory already exists. If we failed
        # to create it, the database open will fail.
        pass

    cache_file = os.path.join(cache_dir, 'tuning.db')
    conn = sqlite3.connect(cache_file)
    return conn

def _close_db(conn):
    """Close a database. This is split into a separate function for the benefit
    of testing, because the close member of the object itself is read-only and
    hence cannot be patched."""
    conn.close()

def autotuner_impl(test, fn, *args, **kwargs):
    """Implementation of :func:`autotuner`. It is split into a separate
    function so that mocks can patch it."""
    cls = args[0]
    classname = '{0}.{1}.{2}'.format(cls.__module__, cls.__name__, fn.__name__)
    tablename = classname.replace('.', '_') + \
        '__' + str(getattr(cls, 'autotune_version', 0))
    keys = _db_keys(fn, args, kwargs)

    conn = _open_db()
    conn.row_factory = sqlite3.Row
    try:
        ans = _query(conn, tablename, keys)
        if ans is None:
            # Nothing found in the database, so we need to tune now
            _logger.info('Performing autotuning for %s with key %s', classname, keys)
            ans = fn(*args, **kwargs)
            values = dict([('value_' + key, value) for (key, value) in ans.iteritems()])
            _save(conn, tablename, keys, values)
        else:
            _logger.debug('Autotuning cache hit for %s with key %s', classname, keys)
    finally:
        _close_db(conn)
    return ans

def autotuner(test):
    r"""Decorator that marks a function as an autotuning function and caches
    the result. The function must take a class and a context as the first
    two arguments. The remaining arguments form a cache key, along with
    properties of the device and the name of the function.

    Every argument to the function must have a name, which implies that the
    \*args construct may not be used.

    Parameters
    ----------
    test : dictionary
        A value that will be returned by :func:`stub_autotuner`.
    """
    @decorator
    def autotuner(fn, *args, **kwargs):
        r"""Decorator that marks a function as an autotuning function and caches
        the result. The function must take a class and a context as the first
        two arguments. The remaining arguments form a cache key, along with
        properties of the device and the name of the function.

        Every argument to the function must have a name, which implies that the
        \*args construct may not be used.
        """
        return autotuner_impl(test, fn, *args, **kwargs)
    return autotuner

def force_autotuner(test, fn, *args, **kwargs):
    """Drop-in replacement for :func:`autotuner_impl` that does not do any
    caching. It is intended to be used with a mocking framework.
    """
    return fn(*args, **kwargs)

def stub_autotuner(test, fn, *args, **kwargs):
    """Drop-in replacement for :func:`autotuner_impl` that does not do
    any tuning, but instead returns the provided value. It is intended to be
    used with a mocking framework."""
    return test

def make_measure(queue, function):
    """Generates a measurement function that can be returned by the
    function passed to :func:`autotune`. It calls `function`
    (with no arguments) the appropriate number of times and returns
    the averaged elapsed time as measured by `queue`."""
    def measure(iters):
        queue.start_tuning()
        for i in range(iters):
            function()
        return queue.stop_tuning() / iters
    return measure

def autotune(generate, time_limit=0.1, threads=None, **kwargs):
    """Run a number of tuning experiments and find the optimal combination
    of parameters.

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
    generate : callable
        function that creates a scoring function
    time_limit : float
        amount of time to spend testing each configuration (excluding setup time)
    threads : int, optional
        number of parallel calls to `generate` to make at a time (defaults to
        CPU count)

    Raises
    ------
    Exception : if every combination throws an exception, the last exception
        is re-raised.
    ValueError : if the Cartesian product is empty
    """
    opts = itertools.product(*kwargs.values())
    best = None
    best_score = None
    had_exception = False
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
            futures = [thread_pool.submit(generate, **keywords) for keywords in batch_keywords]
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
                    _logger.debug("Configuration %s scored %f in %d iterations", keywords, score, iters)
                    if best_score is None or score < best_score:
                        best = keywords
                        best_score = score
                except Exception, e:
                    had_exception = True
                    _logger.debug("Caught exception while testing configuration %s", keywords, exc_info=True)
    if best is None:
        if had_exception:
            raise
        else:
            raise ValueError('No options to test')
    return best
