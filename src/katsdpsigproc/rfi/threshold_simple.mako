/**
 * Apply a global threshold to each sample independently.
 *
 * Mako template arguments:
 *  - wgsx, wgsy
 *  - flag_value
 */

<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void threshold_simple(
    GLOBAL const float * RESTRICT deviations,
    GLOBAL const float * RESTRICT noise,
    GLOBAL unsigned char * RESTRICT flags,
    int stride,
    float n_sigma)
{
    int bl = get_global_id(0);
    int channel = get_global_id(1);
    // TODO: thread coarsening would allow threshold to be reused multiple times
    float threshold = n_sigma * noise[bl];
    int addr = channel * stride + bl;
    flags[addr] = (deviations[addr] > threshold) ? ${flag_value} : 0;
}
