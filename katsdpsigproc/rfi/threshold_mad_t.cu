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

#include <cub/cub.cuh>
#define VT ${vt}
#define WGSX ${wgsx}

typedef cub::BlockReduce<int, WGSX> Reduce;
struct Scratch
{
    typename Reduce::TempStorage reduce;
    int output;
};

__device__ static int abs_int(float x)
{
    return __float_as_int(x) & 0x7FFFFFFF;
}

/**
 * Computes the rank of a value relative to a section of an array, in
 * absolute value. It also provides related functionality.
 */
class RankerAbsSerial
{
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

    /// Count the number of zero elements
    __device__ int zeros() const
    {
        int c = 0;
        for (int i = 0; i < VT; i++)
            c += values[i] == 0;
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
        for (int i = 0; i < VT; i++)
            r += values[i] < value;
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
        for (int i = 0; i < VT; i++)
        {
            int cur = values[i];
            if (cur > ans && cur < limit)
                ans = cur;
        }
        return ans;
    }
};

/**
 * Provides the same interface as @ref RankerAbsSerial, but for
 * collective operation between cooperating threads. The threads
 * are expected to partition the entire array.
 */
class RankerAbsParallel
{
private:
    /// Underlying implementation
    RankerAbsSerial serial;
    /// Shared memory scratch space for reducing counts
    Scratch *scratch;
    bool first;

    __device__ int sum(int value) const
    {
        int total = Reduce(scratch->reduce).Sum(value);
        if (first)
            scratch->output = total;
        __syncthreads();
        return scratch->output;
    }

    __device__ int max(int value) const
    {
        int top = Reduce(scratch->reduce).Reduce(value, cub::Max());
        if (first)
            scratch->output = top;
        __syncthreads();
        return scratch->output;
    }

public:
    __device__ RankerAbsParallel(
        const float *data, int start, int step, int N,
        Scratch *scratch, bool first)
        : serial(data, start, step, N), scratch(scratch), first(first) {}

    __device__ int zeros() const
    {
        int c = serial.zeros();
        return sum(c);
    }

    __device__ int rank(int value) const
    {
        int r = serial.rank(value);
        return sum(r);
    }

    __device__ int max_below(int limit) const
    {
        int s = serial.max_below(limit);
        return max(s);
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

__global__ void __launch_bounds__(WGSX) threshold_mad_t(
    const float * __restrict in, unsigned char * __restrict flags,
    int channels, int stride, float factor)
{
    __shared__ Scratch scratch;

    int bl = blockIdx.y;
    RankerAbsParallel ranker(in + bl * stride, threadIdx.x,
                             WGSX, channels, &scratch, threadIdx.x == 0);
    float threshold = factor * median_abs_impl(ranker, channels);
    for (int i = threadIdx.x; i < channels; i += WGSX)
    {
        int addr = bl * stride + i;
        flags[addr] = (in[addr] > threshold) ? ${flag_value} : 0;
    }
}

} // extern "C"
