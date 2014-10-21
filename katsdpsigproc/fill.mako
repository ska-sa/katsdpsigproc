/**
 * Fill an array with a constant value. This is similar to clEnqueueFillBuffer,
 * but it uses a kernel so that it works on OpenCL 1.1 and on CUDA.
 */

<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1) void fill(
    GLOBAL ${ctype} * RESTRICT out, unsigned int size, ${ctype} value)
{
    unsigned int gid = get_global_id(0);
    if (gid < size)
        out[gid] = value;
}
