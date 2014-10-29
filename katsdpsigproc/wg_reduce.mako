<%def name="op_plus(a, b)">${a} + ${b}</%def>
<%def name="op_max(a, b)">max(${a}, ${b})</%def>
<%def name="op_min(a, b)">min(${a}, ${b})</%def>

<%def name="define_scratch(type, size, scratch_type)">
typedef struct ${scratch_type}
{
    ${type} data[${size}];
} ${scratch_type};
</%def>

/**
 * Cooperative reduction operation amongst @a size workitems. Only commutative
 * operations are currently supported.
 *
 * @param type         The type of items to reduce.
 * @param size         Number of cooperating work-items
 * @param function     Name of the defined function
 * @param scratch_type Type defined with @c define_scratch
 * @param op           Binary operator (defaults to addition)
 * @param rake_width   Number of work-items that perform serial up-sweep.
 *                     Defaults to the SIMD group size.
 */
<%def name="define_function(type, size, function, scratch_type, op=None, rake_width=None)">
<%
if op is None:
    op = op_plus
if rake_width is None:
    rake_width = simd_group_size
%>
DEVICE_FN ${type} ${function}(${type} value, int idx, LOCAL ${scratch_type} *scratch)
{
    // Raking warp phase
% if rake_width < size:
    const int rake_width = ${rake_width};
    if (idx >= rake_width)
        scratch->data[idx] = value;
    BARRIER();
    if (idx < rake_width)
    {
        const int full_chunks = ${size / rake_width};
        for (int i = 1; i < full_chunks; i++)
            value = ${op('value', 'scratch->data[idx + i * rake_width]')};
% if size % rake_width != 0:
        if (idx < ${size % rake_width})
            value = ${op('value', 'scratch->data[idx + full_chunks * rake_width]')};
% endif
        scratch->data[idx] = value;
    }
% else:
    scratch->data[idx] = value;
% endif
    BARRIER();

<% N = min(size, rake_width) %>
% while N > 1:
    // N = ${N}
    if (idx < ${N // 2})
    {
        value = ${op('value', 'scratch->data[idx + %d]' % ((N + 1) // 2))};
        scratch->data[idx] = value;
    }
    BARRIER();
<% N = (N + 1) // 2 %>
% endwhile
    ${type} result = scratch->data[0];
    // This barrier is needed because the scratch might get reused immediately
    BARRIER();
    return result;
}
</%def>
