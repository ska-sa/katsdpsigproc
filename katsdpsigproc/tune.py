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
import sys
import numpy as np

def adapt_value(value):
    """Converts `value` to a type that can be used in sqlite3. This is
    not done through the sqlite3 adapter interface, because that is global
    rather than per-connection. This also only applies to lookup keys,
    not results, because it is not a symmetric relationship
    """
    if isinstance(value, type):
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

@decorator
def autotuner(fn, *args, **kwargs):
    """Decorator that marks a function as an autotuning function and caches
    the result. The function must take a class and a context as the first
    two arguments. The remaining arguments form a cache key, along with
    properties of the device and the name of the function.

    Every argument to the function must have a name, which implies that the
    *args construct may not be used.
    """
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
            ans = fn(*args, **kwargs)
            values = dict([('value_' + key, value) for (key, value) in ans.iteritems()])
            _save(conn, tablename, keys, values)
    finally:
        conn.close()
    return ans

def autotune(measure, **kwargs):
    """Run a number of tuning experiments and find the optimal combination
    of parameters.

    Each argument is a iterable. The `measure` function is passed each
    element of the Cartesian product (by keyword), and returns a score: lower
    is better. If the function raises an exception, it is suppressed. Returns a
    dictionary with the best combination of values.

    Raises
    ------
    Exception : if every combination throws an exception, the last exception
        is re-raised.
    ValueError : if the Cartesian product is empty
    """
    opts = itertools.product(*kwargs.values())
    best = None
    best_score = None
    last_exc_info = None
    for i in opts:
        try:
            kw = dict(zip(kwargs.keys(), i))
            score = measure(**kw)
            if best_score is None or score < best_score:
                best = kw
                best_score = score
        except Exception:
            last_exc_info = sys.exc_info()
    if best is None:
        if last_exc_info is None:
            raise RuntimeError('No options to test')
        else:
            raise last_exc_info[1], None, last_exc_info[2]
    return best
