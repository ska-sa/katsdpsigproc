/**
 * Thin wrapper around wg_reduce.mako to allow it to be tested from host code.
 */

<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

<%wg_reduce:define_scratch type="int" size="${size}" scratch_type="scratch_t"/>
<%wg_reduce:define_function type="int" size="${size}" function="reduce_add" scratch_type="scratch_t"/>
<%wg_reduce:define_function type="int" size="${size}" function="reduce_max" scratch_type="scratch_t" op="${wg_reduce.op_max}"/>

KERNEL void test_reduce_add(GLOBAL const int *in, GLOBAL int *out)
{
    LOCAL_DECL scratch_t scratch;
    int gid = get_global_id(0);
    int value = in[gid];
    int reduced = reduce_add(value, get_local_id(0), &scratch);
    if (gid == 0)
        *out = reduced;
}

KERNEL void test_reduce_max(GLOBAL const int *in, GLOBAL int *out)
{
    LOCAL_DECL scratch_t scratch;
    int gid = get_global_id(0);
    int value = in[gid];
    int reduced = reduce_max(value, get_local_id(0), &scratch);
    if (gid == 0)
        *out = reduced;
}
