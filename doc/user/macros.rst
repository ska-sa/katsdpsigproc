Macros for kernels
==================
In addition to the CUDA/OpenCL portability macros in :file:`port.mako`,
katsdpsigproc also provides a number of macros that help to write more advanced
kernels.

These macros make use of the more advanced parts of `Mako`_ syntax, and you
should probably read its manual alongside this one.

.. _Mako: https://docs.makotemplates.org

Transposition
-------------
Transposition can be done with the :class:`.Transpose` operation without
writing any kernel code; but one may wish to combine the transposition with
other code to save memory bandwidth. The code to do this is contained in
:file:`transpose_base.mako`, and can be loaded with

.. code::

    <%namespace name="transpose" file="transpose_base.mako"/>

You will need to choose three parameters: `block`, `vtx` and `vty`. The kernel
will run with `block` × `block` work groups, and each work group will transpose
a `tiley` × `tilex` region of the source to a `tilex × tiley` region of the
destination, where `tilex` is `block` · `vtx` and `tiley` is `block` · `vty`.
Typical `block` values are 8–32 while `vtx` and `vty` are small — at most 4,
and 1 is not unreasonable (the built-in transpose operation uses
:doc:`autotune` to select these parameters). The examples in this section all
assume that these parameters are passed as variables to the Mako template.

A structure is needed to hold some internal coordinates. Declare it like this:

.. code::

    <%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

This creates a struct named :c:struct:`!transpose_coords`; you could call it
whatever you want. Declare an instance of it in your kernel function, and
initialize it using the generated function
:c:func:`!transpose_coords_init_simple` (the name is generated from the
`class_name`):

.. code::

    transpose_coords coords;
    transpose_coords_init_simple(&coords);

As the name suggests, there is a more general version
:c:func:`!transpose_coords_init` which lets you provide your own information
about which coordinates the work item will work on, but it's beyond the scope
of this tutorial.

Your kernel will consist of two main phases:

1. Load data, transform it if needed, and store it in local memory.

2. Read the data from local memory (with a transposed access pattern),
   transform it further if needed, and write it out.

Thus, you'll need one or more local memory buffers to hold the intermediate
results. Declare the type (at global scope) like this:

.. code::

    <%transpose:transpose_data_class class_name="transpose_values" type="${ctype}" block="${block}" vtx="${vtx}" vty="${vty}"/>

where `ctype` is the C type of each element. It's advisable not to declare
your own struct for the `ctype` as it can hurt efficiency; rather use multiple
data classes (or multiple instances of the same one). When creating an
instance of the struct, remember to declare it as :c:macro:`!LOCAL_DECL`
e.g.

.. code::

    LOCAL_DECL transpose_values values;

Now we're ready to load a chunk of data. Here's a simple example (based on the
:class:`.Transpose` operation) that simply copies data into the local memory:

.. code::

    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        values.arr[${lr}][${lc}] = in[${r} * in_stride + ${c}];
    </%transpose:transpose_load>
    BARRIER();

The `args` determines the names of extra Mako variables that can be used
inside the block. With the names in the example, `(r, c)` are the coordinates
in the input data, and `(lr, lc)` are the coordinates to use in the
intermediate storage. Note that `in` and `in_stride` are not magical: they're
just arguments that were passed to this kernel. You could get the input from
multiple arrays or even synthesize it mathematically if there was a good
reason to.

The call to :c:macro:`!BARRIER` separates the load and store phases, and is
critical to ensure synchronization of the local memory.

The store phase (after the barrier) is very similar:

.. code::

    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        out[${r} * out_stride + ${c}] = values.arr[${lr}][${lc}];
    </%transpose:transpose_store>

Note that once again `(r, c)` are the coordinates for external storage and
`(lr, lc)` are the coordinates for internal storage; don't swap anything
around as it's handled for you.

The examples above are all based on the built-in :class:`Transpose` operation
(but slightly simplified), whose source is in :file:`transpose.mako`. It is
recommended that you look at the source (both the Python and kernel code) as
an example.
