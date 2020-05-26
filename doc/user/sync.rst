Synchronization
===============

Kernel executions are asynchronous, yet so far we haven't needed any
explicit synchronization. That's because

- a command queue is *in-order*, so there is no need to explicitly synchronize
  between sequential commands issued to the same command queue; and
- to get the results we've ended with :meth:`.DeviceArray.get`, which is
  synchronous.

However, we've alluded to the possibility of overlapping data transfers with
computation, which needs more sophisticated techniques involving multiple
command queues. We've seen :meth:`.DeviceArray.get_async` and
:meth:`.DeviceArray.set_async` to do asynchronous transfers, but how do we
tell when they're complete?

The simplest form of synchronization is :meth:`.AbstractCommandQueue.finish`,
which blocks until *all* commands issued on the command queue have completed.
A related function is :meth:`.AbstractCommandQueue.flush`, which ensures that
work has been submitted to the device (rather than being buffered up
somewhere); this is useful if you want to get on with some CPU work at the
same time as the device does its thing, as otherwise the device might not even
start until you call :meth:`~.AbstractCommandQueue.finish`.

Sometimes we don't want the CPU to block on all work in a command queue. A
common paradigm is to use one command queue for transferring data to the
device and a second for doing computation on the device, with a double buffer.
Once a block of data has been transferred, we want the device to be able to
start working on it immediately, without the CPU needing to be involved. In
other words, we want one command queue to block on some event from another
command queue.

To do this, we use *events*. These behave like CUDA events; in OpenCL they
would be called *markers*, and because function names are based on OpenCL, the
relevant method is :meth:`.AbstractCommandQueue.enqueue_marker`, which returns
an instance of :class:`.AbstractEvent`. An event is an item in a
command queue that does no work itself, but which can be waited for.

To wait for an event on the host, use :meth:`.AbstractEvent.wait`. To make a
command queue wait for an event before proceeding with subsequent commands,
use :meth:`.AbstractCommandQueue.enqueue_wait_for_events`.

Profiling
---------
If the command queue was created with profiling enabled, you can also use
events for simple profiling, namely measuring the time elapsed (on the GPU)
between two events. Note that if you are trying to tune your code, there may
be vendor tools that give far more insight.

To create a command queue that supports profiling, use
:meth:`.AbstractContext.create_tuning_command_queue` (instead of the usual
:meth:`~.AbstractContext.create_command_queue`). Then use
:meth:`.AbstractEvent.time_since` to get the time difference between events.
