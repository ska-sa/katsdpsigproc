// Note: does not compile as-is: must be run through mako

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
 * Computes the rank of a value relative to a section of an array, in absolute value.
 *
 * TODO: discuss __float_as_int.
 */
class RankerAbsSerial
{
private:
    ArrayPiece piece;

public:
    __device__ RankerAbsSerial(const ArrayPiece &piece) : piece(piece) {}

    __device__ int zeros() const
    {
        int c = 0;
        for (int i = piece.start; i < piece.end; i++)
            c += piece[i] == 0.0f;
        return c;
    }

    __device__ int rank(int value) const
    {
        int r = 0;
        for (int i = piece.start; i < piece.end; i++)
            r += piece.abs_int(i) < value;
        return r;
    }

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

    __device__ bool first() const
    {
        return piece.start == 0;
    }
};

class RankerAbsParallel
{
private:
    RankerAbsSerial serial;
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
