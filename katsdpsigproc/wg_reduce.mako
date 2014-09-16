<%def name="op_plus(a, b)">${a} + ${b}</%def>
<%def name="op_max(a, b)">max(${a}, ${b})</%def>
<%def name="op_min(a, b)">min(${a}, ${b})</%def>

<%def name="define_scratch(type, size, scratch_type)">
typedef struct ${scratch_type}
{
    ${type} data[${size}];
} ${scratch_type};
</%def>

<%def name="define_function(type, size, function, scratch_type, op=None, rake_width=32)">
<%
if op is None:
    op = op_plus
if rake_width is None or size % rake_width != 0:
    rake_width = size
%>
DEVICE_FN ${type} ${function}(${type} value, int idx, LOCAL ${scratch_type} *scratch)
{
    scratch->data[idx] = value;
    BARRIER();
    // Raking warp phase
% if rake_width < size:
    const int rake_width = ${rake_width};
    if (idx < rake_width)
    {
        for (int i = 1; i < ${size / rake_width}; i++)
            value = ${op('value', 'scratch->data[idx + i * rake_width]')};
    }
    scratch->data[idx] = value;
    BARRIER();
% endif
<% N = rake_width %>
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
