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

#include <float.h> // For FLT_MAX

#define VT ${vt}
#define WGSX ${wgsx}

<%namespace name="wg_reduce" file="/wg_reduce.mako"/>
<%namespace name="rank" file="/rank.mako"/>
<%namespace name="common" file="threshold_mad_common.mako"/>

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

__device__ void ranker_abs_serial_init(
    ranker_abs_serial *self,
    const float *data, int start, int step, int N)
{
    int p = start;
    for (int i = 0; i < VT; i++)
    {
        self->values[i] = (p < N) ? fabs(data[p]) : FLT_MAX;
        p += step;
    }
}

<%rank:ranker_parallel class_name="ranker_abs_parallel" serial_class="ranker_abs_serial" type="float" size="${wgsx}"/>

__device__ void ranker_abs_parallel_init(
    ranker_abs_parallel *self,
    const float *data, int start, int step, int N,
    ranker_abs_parallel_scratch *scratch, int tid)
{
    ranker_abs_serial_init(&self->serial, data, start, step, N);
    self->scratch = scratch;
    self->tid = tid;
}

<%common:median_non_zero ranker_class="ranker_abs_parallel"/>

__global__ void __launch_bounds__(WGSX) threshold_mad_t(
    const float * __restrict in, unsigned char * __restrict flags,
    int channels, int stride, float factor)
{
    __shared__ ranker_abs_parallel_scratch scratch;

    int bl = blockIdx.y;
    ranker_abs_parallel ranker;
    ranker_abs_parallel_init(
        &ranker, in + bl * stride, threadIdx.x,
        WGSX, channels, &scratch, threadIdx.x);
    float threshold = factor * median_non_zero(&ranker, channels);
    for (int i = threadIdx.x; i < channels; i += WGSX)
    {
        int addr = bl * stride + i;
        flags[addr] = (in[addr] > threshold) ? ${flag_value} : 0;
    }
}
