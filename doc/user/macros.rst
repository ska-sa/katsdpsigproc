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

The examples above are all based on the built-in :class:`.Transpose` operation
(but slightly simplified), whose source is in :file:`transpose.mako`. It is
recommended that you look at the source (both the Python and kernel code) as
an example.

Reductions
----------
Reductions are operations like sum or max that repeatedly apply a binary
operator to turn an array of values into a single value. The namespace
:file:`wg_reduce.mako` provides macros for performing reductions within a
single work group. Larger-scale reductions are left to the user to implement
e.g. by recursively applying work-group reductions or by transferring the
results of a single round of reduction to the CPU.

For efficiency, the implementation assumes that the binary operator is
associative and commutative, and hence that the order of operations can be
juggled. An example of a non-commutative operator is matrix multiplication.
Floating-point addition is commutative, but not quite associative due to
rounding errors; thus, summation may return a different result to a CPU
summation, or to summation on a different GPU. However, for a specific system
it is deterministic i.e., is not affected by random timing.

This section covers the basics of defining a reduction across a whole work
group with 1D shape. It is also possible to partition a work group into
equally-sized pieces, each of which performs an independent reduction, but
that is beyond the scope of this tutorial. Refer to the comments in
:file:`wg_reduce.mako` for details.

Here's a kernel where each work group just sums a section of the
input array corresponding to the work group's global IDs. A `wgs` variable
must be passed to indicate the work group size.

.. literalinclude:: examples/sum.mako

Similar to the transposition helpers, we need to define a struct to hold
scratch data, with :c:macro:`!define_scratch`. It takes the type of data being
reduced, the number of items in the reduction, and the name of the struct to
define. We then define the function that does the reduction, using
:c:macro:`!define_function`. It takes those same arguments, as well as the
function name and a macro defining the binary operator. There are a few common
operators provided (:c:macro:`!op_plus`, :c:macro:`!op_min`,
:c:macro:`!op_max`, :c:macro:`!op_fmin`, :c:macro:`!op_fmax`), but you can
easily define your own. The macro takes two values and the type and generates
code for the result. For example, here is the definition of
:c:macro:`!op_plus` [1]_.

.. code::

    <%def name="op_plus(a, b, type)">((${a}) + (${b}))</%def>

.. [1] As with C preprocessor macros, parentheses should be used to ensure
   that operations are ordered correctly if the arguments are
   themselves expressions or if the macro result is used in a larger expression.

Synchronization
_______________
The function defined by :c:macro:`!define_function` contains barriers. It's
thus critical that all work items in the work group call the function
collectively.

Disabling broadcasting
______________________
By default, the return value is valid in all items in the work group. However,
frequently only one work item actually uses the value, as seen in the example
above. In this case one can pass ``broadcast=False`` to
:c:macro:`!define_function`, and only the first work item will receive the sum
(others will receive an undefined value). This improves performance as the
implementation naturally obtains the sum in the first work item and normally
requires an extra step to broadcast it to the others.

Shuffle optimization
____________________
In many cases the internals can be optimized using special CUDA instructions,
but it is not done by default because it is only safe under certain
conditions. The comments in :file:`wg_reduce.mako` have the full details, but
if you use a 1D kernel and pass ``get_local_id(0)`` as the index (as is done
in this example) then it is safe to enable this optimization. To do so, pass
``shuffle=True`` to *both* :c:macro:`!define_scratch` and
:c:macro:`!define_function`.
