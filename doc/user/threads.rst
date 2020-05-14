Thread safety
=============

The package does not get extensively tested with multiple threads. Some parts
of it are likely to be thread-safe, but at present there are no guarantees. If
you have a multi-threaded use case in mind, you should first review the code
to check that it will be safe, and then file an issue to ask the maintainer to
add a guarantee that it will remain safe.

In general, it is not safe to perform two operations on an object in parallel
if at least one of them modifies the object (data race). One exception is the
autotuning cache: it is safe to run autotuning functions in parallel, although
it is not recommended as the same autotuning might end up running multiple
times, and running autotuning benchmarks in parallel will mess up their
timing.

Due to limitations in OpenCL, it is not thread-safe to make multiple calls to
the same kernel at the same time. Note that it is only the Python function
call that needs to be serialized; the execution of the kernels on the device
can proceed in parallel. It also only applies to a single Python object;
two kernel objects created by two different calls to
:meth:`.AbstractProgram.get_kernel` can be used in parallel. It's thus
recommended that this function is called again for each operation created from
an operation template.

It is also not safe to make two calls to the same :class:`.Operation` at the
same time as there are internal structures that may be modified by a call.
Operations should be cheap to manufacture from templates, so each thread
should have its own instance.
