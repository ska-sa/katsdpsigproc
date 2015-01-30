/**
 * @file
 *
 * Device-wide reduction operation, built on top of wg_reduce.mako.
 *
 * Parameters:
 * - type: data type
 * - wgsx, wgsy: number of workitems per workgroup in each dimension
 * - op: expression for combining two values
 * - identity: identity value for @a op
 * - extra_code: arbitrary extra code to add, for use by @a op
 */

<%include file="port.mako"/>
<%namespace name="wg_reduce" file="wg_reduce.mako"/>

${extra_code}

DEVICE_FN ${type} op(${type} a, ${type} b)
{
    return ${op};
}

<%def name="custom_op(a, b, type)">(op((${a}), (${b})))</%def>

${wg_reduce.define_scratch(type, wgsx, 'scratch_t')}
${wg_reduce.define_function(type, wgsx, 'reduce_local', 'scratch_t', custom_op)}

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void hreduce(
    const GLOBAL ${type} * RESTRICT in,
    GLOBAL ${type} * RESTRICT out,
    int start_col,
    int n_cols,
    int in_stride)
{
    LOCAL_DECL scratch_t scratch[${wgsy}];
    int row = get_global_id(1);
    int lid = get_local_id(0);
    int offset = in_stride * row + start_col;

    // Each workitem reduces some elements from its row
    // TODO: could create a separate kernel for the narrow case, to avoid
    // the branch
    ${type} value;
    if (lid >= n_cols)
        value = ${identity};
    else
    {
        value = in[offset + lid];
        for (int c = lid + ${wgsx}; c < n_cols; c += ${wgsx})
        {
            ${type} next = in[offset + c];
            value = op(value, next);
        }
    }

    // Reduce the per-workitem values
    value = reduce_local(value, lid, &scratch[get_local_id(1)]);

    // Write back result
    if (lid == 0)
        out[row] = value;
}
