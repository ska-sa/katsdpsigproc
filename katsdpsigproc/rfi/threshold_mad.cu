// Note: does not compile as-is: must be run through mako

/**
 * @file
 *
 * Median-of-absolute-deviations thresholder. The current design has some
 * number of threads (in the same workgroup) cooperatively computing the
 * median for each channel. However, the threads in each warp are spread
 * across baselines, to ensure efficient memory accesses.
 *
 * This prevents an entire channel from being loaded into the register
 * file, which makes the implementation bandwidth-heavy. A better
 * approach may be to transpose the data and use a workgroup per
 * channel.
 *
 * Medians are found by a binary search to find the value with the
 * required rank. This is done over binary representation of the
 * floating-point values, exploiting the fact that the IEEE-754 encoding
 * of positive floating-point values has the same ordering as the values
 * themselves.
 */

<%namespace name="rank" file="/rank.cu"/>
<%include file="threshold_mad_common.cu"/>

/**
 * Encapsulates a section of a strided (non-contiguous) 1D array.
 */
struct ArrayPiece
{
    const float *in;
    int start;
    int end;
    int stride;

    __device__ ArrayPiece() {}
    __device__ ArrayPiece(const float *in, int start, int end, int stride)
        : in(in), start(start), end(end), stride(stride)
    {
    }

    __device__ float operator[](int idx) const
    {
        return in[idx * stride];
    }
};

<%rank:ranker_serial class_name="RankerAbsSerial" type="int">
    <%def name="foreach()">
        for (int i = piece.start; i < piece.end; i++)
        {
            ${caller.body('abs_int(piece[i])')}
        }
    </%def>

private:
    ArrayPiece piece;

public:
    __device__ RankerAbsSerial(const ArrayPiece &piece) : piece(piece) {}
</%rank:ranker_serial>

<%rank:ranker_parallel class_name="RankerAbsParallel" serial_class="RankerAbsSerial" type="int" size="${wgsy}">
public:
    __device__ RankerAbsParallel(const ArrayPiece &piece, Scratch *scratch, int tid)
        : serial(piece), scratch(scratch), tid(tid) {}
</%rank:ranker_parallel>

extern "C"
{

__global__ void __launch_bounds__(${wgsx * wgsy}) threshold_mad(
    const float * __restrict in, unsigned char * __restrict flags,
    int channels, int stride, float factor,
    int VT)
{
    __shared__ RankerAbsParallel::Scratch scratch[${wgsx}];

    int bl = blockDim.x * blockIdx.x + threadIdx.x;
    int start = threadIdx.y * VT;
    int end = min(start + VT, channels);
    ArrayPiece piece(in + bl, start, end, stride);
    RankerAbsParallel ranker(piece, scratch + threadIdx.x, threadIdx.y);
    float threshold = factor * median_abs_impl(ranker, channels);
    for (int i = piece.start; i < piece.end; i++)
        flags[bl + i * stride] = (piece[i] > threshold) ? ${flag_value} : 0;
}

} // extern "C"
