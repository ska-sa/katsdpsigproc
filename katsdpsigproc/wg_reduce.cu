<%def name="op_plus(a, b)">${a} + ${b}</%def>
<%def name="op_max(a, b)">max(${a}, ${b})</%def>
<%def name="op_min(a, b)">min(${a}, ${b})</%def>

<%def name="define_scratch(type, size, scratch_type)">
typedef struct ${scratch_type}
{
    ${type} data[${size}];
} ${scratch_type};
</%def>

<%def name="define_function(type, size, function, scratch_type, op=None)">
<% if op is None: op = op_plus %>
__device__ ${type} ${function}(${type} value, int idx, ${scratch_type} *scratch)
{
    scratch->data[idx] = value;
    __syncthreads();
<% N = size %>
% while N > 1:
    // N = ${N}
    if (idx < ${N // 2})
    {
        value = ${op('value', 'scratch->data[idx + %d]' % ((N + 1) // 2))};
        scratch->data[idx] = value;
    }
    __syncthreads();
<% N = (N + 1) // 2 %>
% endwhile
    return scratch->data[0];
}
</%def>
