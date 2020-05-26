Asynchronous resource management
================================

For applications that are limited by host-device bandwidth it is often
necessary to perform copies, device computation and even host computation or
networking concurrently. The :mod:`katsdpsigproc.resource` module contains
some utilities to simplify management of resources to prevent race conditions
when used in conjunction with :mod:`asyncio`. For more detailed background,
refer to this `PyConZA talk`_ (or read the `slides`_).

.. _PyConZA talk: https://www.youtube.com/watch?v=9h_dD6OKgq4
.. _slides: https://speakerdeck.com/pyconza/juggling-gpu-tasks-with-asyncio-by-bruce-merry

Firstly, there is an asyncio-friendly way to wait for events
(:func:`.async_wait_for_events`) and a utility class to bound the amount of
in-flight work (:class:`.JobQueue`). The latter addresses a slightly different
problem to `aiojobs`_: aiojobs limits the amount of work executing, but does
not have a way to limit the *pending* work. A :class:`.JobQueue` allows the
caller to block until the backlog has fallen to a determined level.

.. _aiojobs: https://aiojobs.readthedocs.io/

A :class:`.Resource` is a lock with extra features. For convenience it holds a
reference to some object, so that you don't need to carry a lock and the
locked object around separately. In the simplest case, acquiring and releasing
the lock is done like this:

.. code:: python

    acq = resource.acquire()
    with acq as value:   # value is the object passed to Resource() constructor
        await acq.wait_events()
        ...              # Make use of the resource
        acq.ready()      # Releases the lock

That's a surprising amount of boilerplate for a lock. It's complicated because
it's flexible and allows for some use cases that aren't well-supported with a
standard lock.

Ordering control
----------------
Consider a real-time pipeline that receives chunks of data from the network,
processes them, and sends chunks of processed data back into the network. Some
parts of the processing can be done concurrently, but some resources are
shared and hence require locking. Ideally, we want chunk :math:`i` to always
get its turn at a shared resource before chunk :math:`i+1` (so that chunks
don't complete out-of-order), but standard locks don't guarantee this. Even
locks that guarantee first-in-first-out behavior (such as
:class:`.asyncio.Lock`) can't guarantee this, because chunk
:math:`i` might have been slow on a previous (parallel) step, and hence only
try to acquire the lock after chunk :math:`i+1` does.

Resources are always serviced in the order they call
:meth:`.Resource.acquire`. This call can be seen like taking a ticket with a
number at a bank: it guarantees your position in the queue, but you don't
actually need to be ready. You could go home again to collect the documents
you need for your transaction and come back later without losing your place in
the queue. Unlike in real life, you won't be skipped, even if the teller is
sitting idle waiting for you. That does mean that your ordering guarantee may
come at the price of lower utilization.

In our real-time pipeline example, one can ensure that chunks get their turns
by performing all the calls to :meth:`.Resource.acquire` as soon as
the chunk is received. It can then work through its processing, actually
waiting for the resources only as it needs access to them.

Device events
-------------
Astute readers might have noticed that the function to lock the resource is
called :meth:`~.ResourceAllocation.wait_events` rather than just
:meth:`!wait`, and also be wondering why there is an explicit
:meth:`~.ResourceAllocation.ready` function to unlock rather than letting
the context manager handle it. The answer is that there are built-in utilities
for synchronization using events (see the section on :doc:`sync` for a
description of events).

Let's say that a resource is a device buffer, and you're locking it so that
you can launch a kernel that writes to it. If you called
:meth:`~.ResourceAllocation.ready` immediately after enqueuing the kernel,
your code would have a bug because kernels execute asynchronously and hence
you're not actually done with the resource yet. You could use one of the
synchronization primitives to wait for the kernel to complete, but you may
have other work to be getting on with, and it would force all dependencies to
be resolved by the CPU. Instead, you can put an event into the command queue,
and pass it to :meth:`~.ResourceAllocation.ready`:

.. code:: python

    queue.enqueue_kernel(...)   # Or call an Operation
    event = queue.enqueue_marker()
    acq.ready([event])

This says "I'm done using the resource from Python code, but the next user
still has to wait for this event."

Now the name :meth:`~.ResourceAllocation.wait_events` makes more sense: it
waits not only for the previous user to call
:meth:`~.ResourceAllocation.ready`, but also for any events they supplied to
complete. That is still a CPU-side wait through; what if we want to use a
device-side wait? In that case we can just call
:meth:`.ResourceAllocation.wait`, which returns the list of events without
waiting for them. It is then the caller's responsibility, which can be handled
using any suitable method such as
:meth:`.AbstractCommandQueue.enqueue_wait_for_events`.
