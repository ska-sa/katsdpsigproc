Kernels
=======
Now we know how to allocate buffers, but we haven't actually run any code on
the GPU. We'll start at a low level, and in later sections see how
katsdpsigproc's higher-level abstractions can remove some of the drudgery of
GPU programming.

GPU code is compiled on the fly as the
program is running. A context has a :meth:`~.AbstractContext.compile` method to
compile the code and return a :class:`.AbstractProgram`. At this level the
code needs to be written in either CUDA or OpenCL, depending on which type of
context you have. A program can consist of multiple kernels (functions written
to be called from the host and run on the device); use
:meth:`.AbstractProgram.get_kernel` to obtain the kernel with a specific
name. Finally, use :meth:`.AbstractCommandQueue.enqueue_kernel` to run the
code with arguments.

Let's jump in and see our first full example of running code on the GPU. It
will load random data into a buffer, triple every element, and transfer the
results back to the host. It will only work on OpenCL.

.. literalinclude:: examples/triple.py

This is now a fair amount of code, although the first half of it uses
functionality we've seen before. We're allocating an array of 50 floats, but
padding it to 64 because we're using a work group size of 32 work items and 50
is not a multiple of 32.

When enqueueing the kernel, we pass a number of parameters:

1. The kernel itself.
2. A list of kernel arguments. Each argument corresponds to one of the
   parameters of the :c:func:`!multiply` function in :const:`!SOURCE`. For
   pointers we need to provide a buffer. We pass ``buf.buffer``, which is the
   :class:`pyopencl.array.Array` that underpins the katsdpsigproc array. For
   scalar arguments we need to use sized numpy scalars rather than Python
   scalars, and they need to correspond to the type used in the kernel.
3. The "global size", which is the total number (and shape) of work items to
   run for this kernel. Note that this is similar to how OpenCL works, but a
   bit different to CUDA where one specifies the number of blocks rather than
   threads. In this case we've used the *padded* shape, because the global
   size must be a multiple of the local size.
4. The "local size", which is the number (and shape) of work items per work
   group (threads per block in CUDA terms).

Note that due to limitations in the underlying APIs (CUDA and OpenCL), there
is little error-checking done on the arguments. Specifying arguments with the
wrong type can easily lead to bizarre errors and crashes rather than helpful
exceptions.

There are two utility functions that can be helpful in computing the
sizes when it is necessary to add padding: :func:`.divup`
and :func:`.roundup`.

Templating
----------
A major disadvantage of the code above is that it only works on OpenCL,
because the device code is written in OpenCL C. What if we want to write code
that works on both OpenCL and CUDA? To do that we'll make use of a templating
engine and an include file that katsdpsigproc provides to paper over the more
trivial differences between OpenCL C and CUDA C. A templating engine is
similar in concept to the C preprocessor, but significantly more powerful. The
template engine used by katsdpsigproc is `mako`_.

.. _mako: https://docs.makotemplates.org/en/latest/

In later sections we'll also see other uses for templating, the main one being
to compile different variants of the same code, such as to specialize it for
different types. But for now let's modify our example so that it will work on
OpenCL and CUDA:

.. literalinclude:: examples/triple_templ.py

So what have we changed?

- The source code now contains that ``<%include>`` statement,
  which tells mako to include the content of :file:`port.mako` from
  katsdpsigproc. That file contains a lot of C macros and functions to provide
  portability between OpenCL and CUDA C. For example, it defines the
  :c:macro:`!KERNEL` and :c:macro:`!GLOBAL` macros, which expand to the
  appropriate keywords for each API. On CUDA it also defines the
  :c:func:`!get_global_id` function.
- Instead of :meth:`.AbstractContext.compile`, we're now using
  :meth:`~katsdpsigproc.accel.build` to compile the code. This is a
  higher-level function that passes the code through mako.

For the purposes of this example we're specifying an empty filename and
passing the source code directly, but in larger projects one would typically
store the kernel templates in the package and pass the filename to the build
function. Usage might look like

.. code:: python

  program = accel.build(
      ctx, 'my_kernel.mako', {'block': block},
      extra_dirs=[pkg_resources.resource_filename(__name__, '')]
  )

The dictionary argument provides variables that can be expanded in the
template, using ``${varname}`` syntax.
