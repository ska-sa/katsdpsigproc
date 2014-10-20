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
    int deviations_stride,
    int flags_stride,
    float n_sigma)
{
    int bl = get_global_id(0);
    int channel = get_global_id(1);
    // TODO: thread coarsening would allow threshold to be reused multiple times
    float threshold = n_sigma * noise[bl];
    int deviations_addr = channel * deviations_stride + bl;
    int flags_addr = channel * flags_stride + bl;
    flags[flags_addr] = (deviations[deviations_addr] > threshold) ? ${flag_value} : 0;
}
