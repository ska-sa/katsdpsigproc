/**
 * Copy one array to another. This is equivalent to clEnqueueCopyBuffer,
 * and is only provided for benchmarking implementations.
 */

<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1) void copy(
    GLOBAL ${ctype} * RESTRICT out,
    const GLOBAL ${ctype} * RESTRICT in,
    unsigned int size)
{
    unsigned int gid = get_global_id(0);
    if (gid < size)
        out[gid] = in[gid];
}
