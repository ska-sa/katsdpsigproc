Introduction
============

The core of katsdpsigproc is a framework to simplify programming for CUDA and
OpenCL, including writing code that will work on both APIs. OpenCL allows for
programming a range of devices, including CPUs, GPUs and FPGAs, but where it
simplifies the flow of text I will talk about GPUs as they're the most
commonly available accelerator.

GPU programming is a massive topic and attempting to provide background in it
is far beyond the scope of this user guide. I will thus assume that you are
familiar with either CUDA or OpenCL (but not necessarily PyCUDA or PyOpenCL —
that is not necessary to be able to use katsdpsigproc). The API naming tends
towards using OpenCL terminology, as it is more generic. The table below may
be handy for users more familiar with CUDA. If these terms are unfamiliar to
you, I recommend learning about them before trying to use katsdpsigproc.

================  ===============
CUDA              OpenCL
================  ===============
Thread            Work item
Thread block      Work group
Stream            Command queue
Local memory      Private memory
Shared memory     Local memory
Device memory     Global memory
================  ===============

The sections in the guide are organized from basics towards more advanced
features, and later sections often build on knowledge from earlier ones, so
new users are advised to read the guide in order rather than jumping ahead.
