<%include file="port.mako"/>
<%namespace name="wg_reduce" file="wg_reduce.mako"/>

${wg_reduce.define_scratch('int', wgs, 'scratch_t')}
${wg_reduce.define_function('int', wgs, 'reduce_helper', 'scratch_t', wg_reduce.op_plus)}

KERNEL void reduce(const GLOBAL int *in, GLOBAL int *out)
{
    LOCAL_DECL scratch_t scratch;
    int lid = get_local_id(0);
    int value = in[get_global_id(0)];
    int sum = reduce_helper(value, lid, &scratch);
    if (lid == 0)
        out[get_group_id(0)] = sum;
}
