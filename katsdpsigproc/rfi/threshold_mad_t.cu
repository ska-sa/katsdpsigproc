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

#define VT ${vt}
#define WGSX ${wgsx}

<%namespace name="wg_reduce" file="/wg_reduce.cu"/>
<%namespace name="rank" file="/rank.cu"/>
<%include file="threshold_mad_common.cu"/>

<%rank:ranker_serial class_name="RankerAbsSerial" type="int">
    <%def name="foreach()">
        #pragma unroll
        for (int i = 0; i < VT; i++)
        {
            ${caller.body('values[i]')}
        }
    </%def>

private:
    int values[VT];

public:
    __device__ RankerAbsSerial(
        const float *data, int start, int step, int N)
    {
        int p = start;
        for (int i = 0; i < VT; i++)
        {
            values[i] = (p < N) ? abs_int(data[p]) : 0x7FFFFFFF;
            p += step;
        }
    }
</%rank:ranker_serial>

<%rank:ranker_parallel class_name="RankerAbsParallel" serial_class="RankerAbsSerial" type="int" size="${wgsx}">
public:
    __device__ RankerAbsParallel(
        const float *data, int start, int step, int N,
        Scratch *scratch, int tid)
        : serial(data, start, step, N), scratch(scratch), tid(tid) {}
</%rank:ranker_parallel>

extern "C"
{

__global__ void __launch_bounds__(WGSX) threshold_mad_t(
    const float * __restrict in, unsigned char * __restrict flags,
    int channels, int stride, float factor)
{
    __shared__ RankerAbsParallel::Scratch scratch;

    int bl = blockIdx.y;
    RankerAbsParallel ranker(in + bl * stride, threadIdx.x,
                             WGSX, channels, &scratch, threadIdx.x);
    float threshold = factor * median_abs_impl(ranker, channels);
    for (int i = threadIdx.x; i < channels; i += WGSX)
    {
        int addr = bl * stride + i;
        flags[addr] = (in[addr] > threshold) ? ${flag_value} : 0;
    }
}

} // extern "C"
