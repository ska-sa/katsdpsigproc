/**
 * Thin wrapper around wg_reduce.mako to allow it to be tested from host code.
 */

<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%wg_reduce:define_scratch type="int" size="${size}" scratch_type="scratch_t" allow_shuffle="${allow_shuffle}"/>
<%wg_reduce:define_function type="int" size="${size}" function="reduce_add" scratch_type="scratch_t" allow_shuffle="${allow_shuffle}" broadcast="${broadcast}"/>
<%wg_reduce:define_function type="int" size="${size}" function="reduce_max" scratch_type="scratch_t" op="${wg_reduce.op_max}" allow_shuffle="${allow_shuffle}" broadcast="${broadcast}"/>

<%def name="test_function(function)">
KERNEL void test_${function}(GLOBAL const int *in, GLOBAL int *out)
{
    LOCAL_DECL scratch_t scratch[${rows}];
    int gid = get_global_id(1) * get_local_size(0) + get_global_id(0);
    int value = in[gid];
    int reduced = ${function}(value, get_local_id(0), &scratch[get_local_id(1)]);
    out[gid] = reduced;
}
</%def>

${test_function('reduce_add')}
${test_function('reduce_max')}
