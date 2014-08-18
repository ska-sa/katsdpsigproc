// Note: does not compile as-is: must be run through mako

#include <math.h>

__device__ static float absc(float2 v)
{
    return hypotf(v.x, v.y);
}

/**
 * Serial sliding-window median filter. New elements are added using
 * @ref slide, and the median of the last @a WIDTH elements is retrieved
 * by calling @ref get.
 */
template<typename T, int WIDTH>
class MedianFilter
{
private:
    /**
     * History of samples. @a samples[0] is the oldest.
     */
    T samples[WIDTH];

    /**
     * Rank of each sample (0 being smallest). When there are ties, the
     * oldest sample is considered smaller.
     */
    int rank[WIDTH];

public:
    /**
     * Initialise the filter using zero-valued samples.
     */
    __device__ MedianFilter();

    /// Return the median of the current samples
    __device__ T get() const;

    /// Return the middle of the current samples
    __device__ T center() const;

    /// Add a new sample, dropping the oldest
    __device__ void slide(T new_sample);
};

template<typename T, int WIDTH>
__device__ MedianFilter<T, WIDTH>::MedianFilter()
{
    for (int i = 0; i < WIDTH; i++)
    {
        samples[i] = 0;
        rank[i] = i;
    }
}

template<typename T, int WIDTH>
__device__ T MedianFilter<T, WIDTH>::get() const
{
    const int H = WIDTH / 2;
    T result = 0;
    for (int j = 0; j < WIDTH; j++)
    {
        result = (rank[j] == H) ? samples[j] : result;
    }
    return result;
}

template<typename T, int WIDTH>
__device__ T MedianFilter<T, WIDTH>::center() const
{
    return samples[WIDTH / 2];
}

template<typename T, int WIDTH>
__device__ void MedianFilter<T, WIDTH>::slide(T new_sample)
{
    T old_sample = samples[0];
    int new_rank = WIDTH - 1;
#pragma unroll
    for (int j = 0; j < WIDTH - 1; j++)
    {
        samples[j] = samples[j + 1];
        int cmp = (new_sample < samples[j]);
        // TODO: only need to compare new_sample and samples[j] once
        rank[j] = rank[j + 1] + cmp - (old_sample <= samples[j]);
        new_rank -= cmp;
    }
    samples[WIDTH - 1] = new_sample;
    rank[WIDTH - 1] = new_rank;
}

template<typename T, typename T2, int WIDTH>
__device__ static void medfilt_serial_sliding(
    const T2 * __restrict in, T * __restrict out,
    int first, int last, int N, int stride)
{
    const int H = WIDTH / 2;
    MedianFilter<T, WIDTH> filter;

    // Load the initial window, substituting zeros beyond the ends.
    // These is no need for this on the leading edge, because the
    // constructor initialises with zero samples.
    for (int i = max(0, first - H); i < min(first + H, N); i++)
        filter.slide(absc(in[i * stride]));
    for (int i = N; i < first + H; i++)
        filter.slide(0);

    for (int i = first; i < min(last, N - H); i++)
    {
        filter.slide(absc(in[(i + H) * stride]));
        out[i * stride] = filter.center() - filter.get();
    }
    for (int i = max(first, N - H); i < last; i++)
    {
        filter.slide(0);
        out[i * stride] = filter.center() - filter.get();
    }
}

extern "C"
{

/**
 * Apply median filter to each baseline. The input data are stored channel-major,
 * baseline minor, with a separation of @a stride between rows. Each workitem produces
 * (up to) @a VT channels of output. The input must be suitably zero-padded.
 */
__global__ void __launch_bounds__(${wgs}) background_median_filter(
    const float2 * __restrict in, float * __restrict out,
    int channels, int stride, int VT)
{
    const int WIDTH = ${width};
    int bl = blockDim.x * blockIdx.x + threadIdx.x;
    int sub = blockDim.y * blockIdx.y + threadIdx.y;
    int start_channel = sub * VT;
    int end_channel = min(start_channel + VT, channels);
    medfilt_serial_sliding<float, float2, WIDTH>(in + bl, out + bl, start_channel, end_channel, channels, stride);
}

} // extern C
