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

    /**
     * Absolute value, reinterpreted as an int. The sign bit is
     * explicitly masked off to ensure the compiler doesn't take
     * any shortcuts with negative zero.
     */
    __device__ int abs_int(int idx) const
    {
        return __float_as_int(in[idx * stride]) & 0x7fffffff;
    }

    __device__ float operator[](int idx) const
    {
        return in[idx * stride];
    }
};

/**
 * Computes the rank of a value relative to a section of an array, in
 * absolute value. It also provides related functionality.
 */
class RankerAbsSerial
{
private:
    ArrayPiece piece;

public:
    __device__ RankerAbsSerial(const ArrayPiece &piece) : piece(piece) {}

    /// Count the number of zero elements
    __device__ int zeros() const
    {
        int c = 0;
        for (int i = piece.start; i < piece.end; i++)
            c += piece[i] == 0.0f;
        return c;
    }

    /**
     * Count the number of elements which absolute value is strictly
     * less than @a value. The value is provided as a bit pattern rather
     * than a float.
     */
    __device__ int rank(int value) const
    {
        int r = 0;
        for (int i = piece.start; i < piece.end; i++)
            r += piece.abs_int(i) < value;
        return r;
    }

    /**
     * Return the bit pattern of the largest absolute value that is
     * strictly less than @a limit (also a bit pattern). Returns 0
     * if there isn't one.
     */
    __device__ int max_below(int limit) const
    {
        int ans = 0;
        for (int i = piece.start; i < piece.end; i++)
        {
            int cur = piece.abs_int(i);
            if (cur > ans && cur < limit)
                ans = cur;
        }
        return ans;
    }

    /**
     * Indicates whether this is the first segment of the array.
     */
    __device__ bool first() const
    {
        return piece.start == 0;
    }
};

/**
 * Provides the same interface as @ref RankerAbsSerial, but for
 * collective operation between cooperating threads. The threads
 * are expected to partition the entire array (in particular,
 * exactly one of them must start at the beginning).
 */
class RankerAbsParallel
{
private:
    /// Underlying implementation
    RankerAbsSerial serial;
    /// Shared memory scratch space for reducing counts
    int *scratch;

public:
    __device__ RankerAbsParallel(const ArrayPiece &piece, int *scratch)
        : serial(piece), scratch(scratch) {}

    __device__ int zeros() const
    {
        if (serial.first())
            scratch[0] = 0;
        __syncthreads();

        int c = serial.zeros();
        atomicAdd(scratch, c);
        __syncthreads();

        return scratch[0];
    }

    __device__ int rank(int value) const
    {
        if (serial.first())
            scratch[0] = 0;
        __syncthreads();

        int r = serial.rank(value);
        atomicAdd(scratch, r);
        __syncthreads();

        return scratch[0];
    }

    __device__ int max_below(int limit) const
    {
        if (serial.first())
            scratch[0] = 0;
        __syncthreads();

        int s = serial.max_below(limit);
        atomicMax(scratch, s);
        __syncthreads();

        return scratch[0];
    }
};

/**
 * Return the element of @a in whose absolute values has rank @a rank.
 *
 * If @a halfway is true, then it returns the average of ranks @a rank and @a
 * rank - 1.
 */
template<typename Ranker>
__device__ float find_rank_abs(const Ranker &ranker, int rank, bool halfway)
{
    int cur = 0;
    for (int i = 30; i >= 0; i--)
    {
        int test = cur | (1 << i);
        int r = ranker.rank(test);
        if (r <= rank)
            cur = test;
    }

    float result = __int_as_float(cur);
    if (halfway)
    {
        int r = ranker.rank(cur);
        if (r == rank)
        {
            float prev = __int_as_float(ranker.max_below(cur));
            result = (result + prev) * 0.5f;
        }
    }
    return result;
}

/**
 * Finds the median absolute value in @a in, ignoring zeros.
 */
template<typename Ranker>
__device__ float median_abs_impl(const Ranker &ranker, int N)
{
    int zeros = ranker.zeros();
    int rank2 = N + zeros;
    return find_rank_abs(ranker, rank2 / 2, !(rank2 & 1));
}

extern "C"
{

__global__ void __launch_bounds__(${wgsx * wgsy}) threshold_mad(
    const float * __restrict in, unsigned char * __restrict flags,
    int channels, int stride, float factor,
    int VT)
{
    __shared__ int scratch[${wgsx}];

    int bl = blockDim.x * blockIdx.x + threadIdx.x;
    int start = threadIdx.y * VT;
    int end = min(start + VT, channels);
    ArrayPiece piece(in + bl, start, end, stride);
    RankerAbsParallel ranker(piece, scratch + threadIdx.x);
    float threshold = factor * median_abs_impl(ranker, channels);
    for (int i = piece.start; i < piece.end; i++)
        flags[bl + i * stride] = (piece[i] > threshold) ? ${flag_value} : 0;
}

} // extern "C"
