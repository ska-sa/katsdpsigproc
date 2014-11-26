// Note: does not compile as-is: must be run through mako

/**
 * @file
 *
 * Median-of-absolute-deviations thresholder. It takes the data in
 * transposed form (baseline-major), and each workgroup loads an
 * entire baseline into registers.
 *
 * Medians are found by a binary search to find the value with the
 * required rank. This is done over binary representation of the
 * floating-point values, exploiting the fact that the IEEE-754 encoding
 * of positive floating-point values has the same ordering as the values
 * themselves.
 */

<%include file="/port.mako"/>

#define VT ${vt}
#define WGSX ${wgsx}

<%namespace name="wg_reduce" file="/wg_reduce.mako"/>
<%namespace name="rank" file="/rank.mako"/>

<%rank:ranker_serial class_name="ranker_abs_serial" type="float">
    <%def name="foreach(self)">
        #pragma unroll
        for (int i = 0; i < VT; i++)
        {
            ${caller.body('(%s)->values[i]' % (self,))}
        }
    </%def>
    float values[VT];
</%rank:ranker_serial>

DEVICE_FN void ranker_abs_serial_init(
    ranker_abs_serial *self,
    GLOBAL const float *data, int start, int step, int N)
{
    int p = start;
    for (int i = 0; i < VT; i++)
    {
        self->values[i] = (p < N) ? fabs(data[p]) : FLT_MAX;
        p += step;
    }
}

<%rank:ranker_parallel class_name="ranker_abs_parallel" serial_class="ranker_abs_serial" type="float" size="${wgsx}">
    <%def name="thread_id(self)">
        get_local_id(0)
    </%def>
</%rank:ranker_parallel>

DEVICE_FN void ranker_abs_parallel_init(
    ranker_abs_parallel *self,
    GLOBAL const float * RESTRICT data, int start, int step, int N,
    LOCAL ranker_abs_parallel_scratch *scratch)
{
    ranker_abs_serial_init(&self->serial, data, start, step, N);
    self->scratch = scratch;
}

<%rank:median_non_zero_float ranker_class="ranker_abs_parallel" uniform="${True}"/>

KERNEL REQD_WORK_GROUP_SIZE(WGSX, 1, 1) void madnz_t(
    GLOBAL const float * RESTRICT in,
    GLOBAL float * RESTRICT noise,
    int channels, int stride)
{
    LOCAL_DECL ranker_abs_parallel_scratch scratch;

    int bl = get_group_id(1);
    ranker_abs_parallel ranker;
    ranker_abs_parallel_init(
        &ranker, in + bl * stride, get_local_id(0),
        WGSX, channels, &scratch);
    float s = 1.4826f * median_non_zero_float(&ranker, channels);
    if (get_local_id(0) == 0)
        noise[bl] = s;
}
