Autotuning
==========
The kernels shown so far all have a fixed work-group size. However, it's not
easy to know in advance what the best work-group size is for a specific piece
of hardware, and even harder to know what it should be in code that may run on
multiple generations of hardware. Furthermore, work-group size is just the
most common tuning parameter, but there might be others.

A common approach to this problem is to use "autotuning": benchmark
different options on the fly and then use the best. There are two variants of
this approach, which I'll call "on-line" and "off-line". In on-line
autotuning, each time a kernel is invoked by the user, a different set of
parameters is tested; the benchmarking is a side effect of doing useful work,
but the useful work will have variable performance until the tuner has
converged. In off-line autotuning, benchmarking is done on a synthetic
workload and the optimal parameters are selected before doing any real work.

On-line autotuning has the advantage that it does not require a long wait
before useful work can be done, and also that the workload being benchmarked
will be representative. However, katsdpsigproc was originally developed for
use in real-time pipelines where highly variable performance during the tuning
phase is unacceptable. It thus only supports off-line autotuning.

Despite being off-line, the autotuning is transparent and automatic to users
of operations. If an operation template is constructed in a new configuration,
the autotuning will be run. The result is saved in a sqlite database so that
it will not need to be run again. Results are indexed by the device and driver
version so that old results will not be reused for new hardware for which they
might not be appropriate.

To take advantage of autotuning, authors of operations will need to add code
to their operation templates to specify the parameter space to search and the
code to benchmark. Let's see how that might look for our
:class:`!MultiplyTemplate` class (see :ref:`operation-templates`), to make it
automatically determine the work group size.

.. code:: python

    class MultiplyTemplate:
        def __init__(self, context, tuning=None):
            if tuning is None:
                tuning = self.autotune(context)
            self.wgs = tuning['wgs']
            self.program = build(context, '', source=SOURCE)

        @classmethod
        @katsdpsigproc.tune.autotuner(test={'wgs': 32})
        def autotune(cls, context):
            queue = context.create_tuning_command_queue()
            size = 1048576

            def generate(wgs):
                fn = cls(context, {'wgs': wgs}).instantiate(queue, size, 1)
                fn.ensure_all_bound()
                fn.buffer('data').zero(queue)
                return katsdpsigproc.tune.make_measure(queue, fn)

            return katsdpsigproc.tune.autotune(generate, wgs=[32, 64, 128, 256])

        def instantiate(self, queue, size, scale):
            return Multiply(self, queue, size, scale)

There is a fair amount of convention and boiler-plate here, so let's go
through it a step at a time.

- The constructor takes an extra `tuning` argument, defaulting to ``None``.
  Tuning parameters can be explicitly provided as a dictionary with string
  keys; in this case the only key used is ``'wgs'`` (short for work-group
  size). Users can thus override the tuning parameters, but this argument is
  really intended for internal use.

- A class method (:meth:`!autotune`) computes the optimal parameters for a given
  configuration. It has a decorator that tells the autotuning system to cache
  the result, similar to :func:`functools.lru_cache`. For our simple class there
  is no configuration, but if this function takes additional arguments they
  form part of the database key so that different configurations are tuned
  separately. These types need to be simple types like numbers and strings
  that can be serialized by :mod:`!sqlite3`, but there is support for enums and
  numpy dtypes. We'll come back to the ``test=`` part in the section on
  :ref:`autotune-testing`.

- The function uses :func:`katsdpsigproc.tune.autotune` to do the actual
  autotuning. It is passed a function to describe how to benchmark a
  specific set of parameters, and a keyword argument for each parameter to
  tune (whose names must match the argument names to :func:`!generate`) with a list
  of values to try. The autotuner is not particularly clever: given multiple
  parameters to tune, it tries all combinations, so if there are many
  parameters you need to be careful not to cause a combinatorial explosion
  that will take forever to test.

- The :func:`!generate` function sets up the benchmark for a specific value of
  `wgs` by constructing an instance of the class with the explicitly-provided
  tuning parameters. It also instantiates it (giving an instance of
  :class:`!Multiply`) with a size chosen to be large enough to reasonably
  exercise a GPU, and allocates buffers. It would be more efficient to
  allocate a single buffer once outside :func:`!generate` to be used for all
  possible values of `wgs`, but one needs to be careful that such a buffer is
  suitably padded for all cases.

  It then uses :func:`katsdpsigproc.tune.make_measure` to construct a
  benchmark function, which will return the performance of this configuration
  each time it is called. You could build your own benchmark function, but
  :func:`~katsdpsigproc.tune.make_measure` takes care of inserting markers
  into a command queue on either side of your operation and querying them to
  get the elapsed GPU time. The autotuning system will call the benchmark
  function multiple times to get an estimate of performance.

And that's it! The only change to the rest of the code is that the
:class:`!Multiply` kernel now needs to use ``template.wgs`` instead of
``template.WGS`` because it's no longer a Python constant.

Most of my autotuning functions look broadly similar to the above, but the
only part that really does any magical introspection is the
:func:`katsdpsigproc.tune.autotuner` decorator, and you can write the body of
your functions in completely different ways if you so choose.

Skipping combinations
---------------------
As mentioned, when multiple parameters are being tuned together, the tuner
will try all combinations, which can take an excessive amount of time. To test
only a smaller subset of combinations, one can return ``None`` from the
:func:`!generate` function to skip testing of that combination. This still
costs a Python function call so one should still avoid starting with a space
containing billions of combinations.

Some combinations might also lead to compiler errors, for example, because
they use too many registers. The autotuning system will gracefully skip
combinations that cause exceptions, so it is not necessary to catch and deal
with the compiler errors yourself. Not catching exceptions also means you'll
get a more useful error if you introduce a bug that causes *all* combinations
to fail.

Versioning
----------
Autotuning results are inserted into a SQL table whose name is based on the
fully-qualified name of the autotuning function, and has columns for the
device, platform, driver version, the arguments to the autotuning function,
and the dictionary keys in the result. This presents a problem if you want to
change the arguments or return keys from the function, because users who have
already run autotuning will get database errors when the columns don't match.
Furthermore, even if you don't change the interface, you might change the
implementation to such an extent that old autotuning results might no longer
be appropriate.

To solve these issues, the table name also includes a version number. It
defaults to zero, but can be overridden by define a class constant
:const:`!autotune_version`. Old results will *not* be removed from the
database, and might even still be used if the user downgrades back to the
previous version.

Overriding autotuning
---------------------
The default behaviour of katsdpsigproc's autotuning machinery is to autotune
for an inexact match between the GPU detected at runtime and the results
stored in the autotuning SQL table.

It is possible to request an inexact match in the autotuning lookup by
setting an environment variable, :envvar:`KATSDPSIGPROC_TUNE_MATCH`. If
:envvar:`KATSDPSIGPROC_TUNE_MATCH` is set to "nearest", the nearest match to the
current GPU in the autotuning SQL table will be returned, by ignoring in turn
the device driver, then platform, then device name. If no match is found,
autotuning will proceed. If `KATSDPSIGPROC_TUNE_MATCH` is set to "exact" (or
anything else), default behaviour will proceed.

.. _autotune-testing:

Testing
-------
The testing in general is addressed in :doc:`testing`, but it is worth noting
that autotuning causes some additional challenges in testing:

- One wants tests to be reproducible, but if different developers end up with
  different autotuning results, they will end up running different tests.

- The autotuning code should itself be tested, but once it has been run once
  the result will be cached and it will not run again.

To address these issues, the :ref:`context <fixture-context>` fixture disables
autotuning. Instead, your :func:`!autotune` function will return the result
specified with the `test` keyword argument to the :func:`.autotuner`
decorator. You should use an argument that is likely to work across a range of
devices.

To test the autotuning code itself, use the
:ref:`force_autotune <mark-force_autotune>` mark. It overrides the behavior
described above so that the autotuning function always runs with no caching.
