Initialization
==============
Before any useful work can be done, one needs a context and (usually) a
command queue. A context is associated with a single device, and "owns" all
memory allocations. Applications will typically only need a single context,
unless they make use of multiple GPUs. A command queue is associated with a
context, and is used to submit work (both kernels and copies) to the GPU.
Because commands in a command queue are executed serially, high-performance
applications may need multiple command queues so that data transfers can be
done in parallel with computations.

While it's possible to use the API to enumerate devices yourself, the
following code is generally sufficient to create a context and a
command queue.

.. code:: python

  import katsdpsigproc.accel

  ctx = katsdpsigproc.accel.create_some_context()
  queue = ctx.create_command_queue()

If there are multiple devices found, it will prompt the user to select one.
Refer to :doc:`../installation` for instructions on setting an environment
variable to remember that choice. If you don't want the user to be prompted,
you can pass ``interactive=False`` to
:meth:`~katsdpsigproc.accel.create_some_context`, in which case the first
device found will be used. It is also possible to filter the devices that will
be considered. For example, to only consider CUDA and not OpenCL (maybe
because other parts of your code interact with CUDA-only libraries like
cuFFT), use

.. code:: python

  ctx = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

While it should be familiar if you've done extensive GPU programming before, it
is worth repeating: unless otherwise noted, commands issued to a single command
queue are executed serially but asynchronously.
