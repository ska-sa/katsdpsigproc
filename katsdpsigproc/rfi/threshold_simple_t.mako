/**
 * Apply a global threshold to each sample independently,
 * with transposed data.
 *
 * Mako template arguments:
 *  - wgsx, wgsy
 *  - flag_value
 */

<%include file="/port.mako"/>

KERNEL void REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) threshold_simple_t(
    GLOBAL const float * RESTRICT deviations,
    GLOBAL const float * RESTRICT noise,
    GLOBAL unsigned char * RESTRICT flags,
    unsigned int stride,
    float n_sigma)
{
    int bl = get_global_id(1);
    int channel = get_global_id(0);
    // TODO: threshold could be loaded by one thread
    // and broadcast through local memory
    float threshold = n_sigma * noise[bl];
    int addr = bl * stride + channel;
    flags[addr] = (deviations[addr] > threshold) ? ${flag_value} : 0;
}
