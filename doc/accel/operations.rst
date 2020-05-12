Operations
==========
In the section on :doc:`kernels <kernels>` we saw how to compile and run a
simple kernel. However, working directly with kernels is not necessarily
convenient:

- If you always use the same kernel with the same buffers, you need to pass
  them all around in your code.
- A kernel may be designed or optimized for a specific work-group size, but
  that is not captured with the kernel.
- Kernels may impose alignment or padding requirements on the buffers used
  with them, but this is not captured programmatically.
- If you use a buffer with multiple kernels, they may each have different
  padding or alignment requirements, and you'll need to ensure that you pad
  the buffer allocation appropriately to satisfy all of them.
- Some algorithms require running multiple kernels, and it would be useful to
  have the same interface as for single-kernel algorithms.

"Operations" are higher-level constructs that address these shortcomings. In
the simplest case, an operation bundles a kernel together with information
about requirements on the buffers they're used with and currently-bound
buffers. Operations can also be composed to build more complex algorithms from
simple ones.

Let's see an example before diving into the details:

.. literalinclude:: ../examples/triple_op.py

The kernel ``SOURCE`` is unchanged from the previous example. However, we've
now encapsulated all the logic for multiplying a buffer by a constant into a
class ``Multiply``, which derives from :class:`.Operation`. The constructor
builds the kernel much as before; what's new is the use of :class:`.IOSlot` and
:class:`.Dimension`. We'll come to the details of those in the next section.

We also see that we're no longer allocating the buffer ourselves. Instead, the
call to :meth:`.Operation.ensure_all_bound` takes care of allocating memory
for all the associated buffers of an operation. We could still choose to
allocate our own buffers and :meth:`.Operation.bind` them; this can be useful
for example if using double-buffering so that one buffer is in use by the
operation while another is busy transferring data to or from the host.

To enqueue the operation, we simply call it as a function. Because the object
encapsulates all the information about how to run the operation and the
buffers involved, we don't need to provide any arguments.

Dimensions
----------
A :class:`.Dimension` specifies size and padding requirements along one axis of
a (possibly) multi-dimensional array. There is a subtle but
important rule about :class:`.Dimension`: a specific
:class:`.Dimension` object always resolves to a specific padded size. Thus, if
a single object is shared between two buffers, it not only gives them the same
requirements, it constrains them to have the *same* amount of padding. This
can be useful as it allows a single linear address to be used for multiple
buffers. But it can also lead to unnecessarily strict constraints, so in some
cases you may wish to construct two separate but identical dimension objects
rather than re-using one object.

The most common case for a padding requirement is to compensate for a kernel's
``global_size`` being rounded up to a multiple of the work group size. This is
seen in the example above: ``Dimension(size, self.WGS)`` rounds ``size`` up to
a multiple of ``self.WGS`` and makes it the minimum for the padded size
(larger padded sizes are still permitted, even if not a multiple of
``self.WGS``). There are other options that can be specified, including to
disallow padding (to ensure the buffer is contiguous); refer to the
constructor documentation for details.

Slots
-----
An :class:`.IOSlot` takes a tuple of dimensions to specify the
multi-dimensional shape of a buffer, and a dtype. For convenience, axes
without padding requirements may be specified just with an integer size
instead of a :class:`.Dimension`. It also has a currently-bound buffer
(initially ``None``).

For implementation reasons relating to composing operations, the buffer bound
to a slot should be retrieved with :meth:`.Operation.buffer` rather than via
the slots dictionary.

Operation templates
-------------------
The ``Multiply`` operation in our example is fine if we only want one of them,
but what if we want several with different sizes or scale factors? Each time
we instantiate a new one we will need to re-run the template engine and the
compiler, which is not very efficient. To avoid this overhead, it is
conventional to split out the static parts of each operation into an
"operation template" so that multiple :class:`.Operation` objects can share
the same program.

Let's see how that might look in our example:

.. literalinclude:: ../examples/triple_op_template.py

This doesn't use any new features of katsdpsigproc; we've simply refactored.
There are some conventions used:

- The template class has the same name as the operation class, with
  ``Template`` appended.
- It has an ``instantiate`` method which returns an instance of the operation,
  passing all its arguments through.
- The operation constructor takes the template as the first argument, and
  assigns it to a :attr:`template` attribute.
- The call to :meth:`~.AbstractProgram.get_kernel` is only done in the
  operation, not the template. This is because in PyOpenCL, two threads cannot
  safely call the same kernel object at the same time. With each operation
  having its own kernel object, it is safe for different threads to use
  different operations built from the same operation template.
