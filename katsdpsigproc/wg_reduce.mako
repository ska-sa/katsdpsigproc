<%def name="op_plus(a, b, type)">((${a}) + (${b}))</%def>
<%def name="op_max(a, b, type)">max((${a}), (${b}))</%def>
<%def name="op_min(a, b, type)">min((${a}), (${b}))</%def>
<%def name="op_fmin(a, b, type)">
% if type.startswith('float') or type.startswith('double'):
fmin((${a}), (${b}))
% else:
min((${a}), (${b}))
% endif
</%def>
<%def name="op_fmax(a, b, type)">
% if type.startswith('float') or type.startswith('double'):
fmax((${a}), (${b}))
% else:
max((${a}), (${b}))
% endif
</%def>

## Defines the data structure for a reduction operation. See below for the
## meaning of allow_shuffle.
<%def name="define_scratch(type, size, scratch_type, allow_shuffle=False)">
<% small_shuffle = allow_shuffle and (size & (size - 1)) == 0 and size <= simd_group_size %>
typedef struct ${scratch_type}
{
#if !SHUFFLE_AVAILABLE || ${int(not small_shuffle)}
    ${type} data[${size}];
#endif
} ${scratch_type};
</%def>

/**
 * Cooperative reduction operation amongst @a size workitems. Only commutative
 * operations are currently supported.
 *
 * If @a allow_shuffle is True, the caller guarantees that
 * - the @a idx passed to the generated function is the linear thread ID modulo
 *   @a size.
 * In this case, the implementation @em may use Kepler shuffle instructions on
 * CUDA for communication, but it will not do so in all cases. The following
 * will prevent this fast path from being used
 * - not compiling for CUDA
 * - not compiling for Compute Capability 3.0 or higher
 * - size is not a power of 2
 *
 * @param type         The type of items to reduce.
 * @param size         Number of cooperating work-items
 * @param function     Name of the defined function
 * @param scratch_type Type defined with @c define_scratch
 * @param op           Binary operator (defaults to addition)
 * @param rake_width   Number of work-items that perform serial up-sweep.
 *                     Defaults to the SIMD group size.
 * @param allow_shuffle See above.
 */
<%def name="define_function(type, size, function, scratch_type, op=None, rake_width=None, allow_shuffle=False)">
<%
if op is None:
    op = op_plus
if rake_width is not None and allow_shuffle:
    raise ValueError("rake_width may not currently be set when allow_shuffle is used")
if rake_width is None:
    rake_width = simd_group_size
rake_width = min(rake_width, size)
use_shuffle = int(allow_shuffle and (size & (size - 1)) == 0)
%>

DEVICE_FN ${type} ${function}(${type} value, int idx, LOCAL ${scratch_type} *scratch)
{
% if size > rake_width:
    const int rake_width = ${rake_width};
    // Transfer values to shared memory for the raking warp to sum
    const bool first_rake = idx < rake_width;
    if (!first_rake)
        scratch->data[idx] = value;
    BARRIER();
% else:
    const bool first_rake = true;
% endif

    // Raking warp sums values from other warps
    if (first_rake)
    {
% if size > rake_width:
        const int full_chunks = ${size / rake_width};
        for (int i = 1; i < full_chunks; i++)
            value = ${op('value', 'scratch->data[idx + i * rake_width]', type)};
% if size % rake_width != 0:
        if (idx < ${size % rake_width})
            value = ${op('value', 'scratch->data[idx + full_chunks * rake_width]', type)};
% endif
% endif
        // If we're shuffling, we can do the final reduction inside the
        // first_rake test. Otherwise, it has to be outside to allow barriers
        // to be in uniform control flow.
#if SHUFFLE_AVAILABLE && ${use_shuffle}
        union
        {
            ${type} v;
            int a[sizeof(${type}) / sizeof(int)];
        } transfer;
        for (int i = ${rake_width} / 2; i >= 1; i /= 2)
        {
            transfer.v = value;
            for (int j = 0; j < sizeof(${type}) / sizeof(int); j++)
                transfer.a[j] = __shfl_down(transfer.a[j], i, ${size});
            value = ${op('value', 'transfer.v', type)};
        }
#else
        scratch->data[idx] = value;
#endif
    }

    // Rake reduction when shuffle is not available
#if !(SHUFFLE_AVAILABLE && ${use_shuffle})
    BARRIER();
<% N = rake_width %>
% while N > 1:
    // N = ${N}
    if (idx < ${N // 2})
    {
        value = ${op('value', 'scratch->data[idx + %d]' % ((N + 1) // 2), type)};
        scratch->data[idx] = value;
    }
    BARRIER();
<% N = (N + 1) // 2 %>
% endwhile
#endif

    // Broadcast result
    // TODO: allow user to specify that only idx==0 needs the result
#if SHUFFLE_AVAILABLE && ${use_shuffle} && ${int(size <= rake_width)}
    // Can reach all threads just using shuffle
    union
    {
        ${type} v;
        int a[sizeof(${type}) / sizeof(int)];
    } transfer;
    transfer.v = value;
    for (int j = 0; j < sizeof(${type}) / sizeof(int); j++)
        transfer.a[j] = __shfl(transfer.a[j], 0, ${size});
    return transfer.v;
#else
# if SHUFFLE_AVAILABLE && ${use_shuffle}
    // In the non-shuffle path, this has already happened in the loop
    if (idx == 0)
        scratch->data[0] = value;
    BARRIER();
# endif
    value = scratch->data[0];
    // This barrier is needed because the scratch might get reused immediately
    BARRIER();
    return value;
#endif
}
</%def>
