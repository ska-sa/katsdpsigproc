// Note: does not compile as-is: must be run through mako

/**
 * @file
 *
 * Applies a median filter to each channel. Each workgroup processes a
 * section of one channel and multiple baselines. At present, each
 * channel is handled by separate threads from possibly several
 * workgroups, although it may be beneficial to group some of those
 * threads into the same workground purely for occupancy reasons. Each
 * thread has significant startup overhead in this implementation, so
 * there is a balance between using more threads for greater parallelism
 * vs fewer threads for reduced overhead.
 *
 * Each thread processes a section of a channel using a sliding-window
 * median filter.
 */

#include <math.h>

#define WIDTH ${width}

/// Complex absolute value
__device__ static float absc(float2 v)
{
    return hypotf(v.x, v.y);
}

/**
 * Serial sliding-window median filter. New elements are added using
 * @ref slide, and the median of the last @a WIDTH elements is retrieved
 * by calling @ref get.
 */
class MedianFilter
{
private:
    /**
     * History of samples. @a samples[0] is the oldest.
     */
    float samples[WIDTH];

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
    __device__ float get() const;

    /// Return the middle of the current samples
    __device__ float center() const;

    /// Add a new sample, dropping the oldest
    __device__ void slide(float new_sample);
};

__device__ MedianFilter::MedianFilter()
{
    for (int i = 0; i < WIDTH; i++)
    {
        samples[i] = 0;
        rank[i] = i;
    }
}

__device__ float MedianFilter::get() const
{
    const int H = WIDTH / 2;
    float result = 0.0f;
    for (int j = 0; j < WIDTH; j++)
    {
        result = (rank[j] == H) ? samples[j] : result;
    }
    return result;
}

__device__ float MedianFilter::center() const
{
    return samples[WIDTH / 2];
}

__device__ void MedianFilter::slide(float new_sample)
{
    float old_sample = samples[0];
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

/**
 * Apply the median filter on a single thread. The range of output
 * values to produce is [@a first, @a last), out of a total array of
 * size @a N.
 */
__device__ static void medfilt_serial_sliding(
    const float2 * __restrict in, float * __restrict out,
    int first, int last, int N, int stride)
{
    const int H = WIDTH / 2;
    MedianFilter filter;

    // Load the initial window, substituting zeros beyond the ends.
    // These is no need for this on the leading edge, because the
    // constructor initialises with zero samples.
    for (int i = max(0, first - H); i < min(first + H, N); i++)
        filter.slide(absc(in[i * stride]));
    for (int i = N; i < first + H; i++)
        filter.slide(0.0f);

    for (int i = first; i < min(last, N - H); i++)
    {
        filter.slide(absc(in[(i + H) * stride]));
        out[i * stride] = filter.center() - filter.get();
    }
    for (int i = max(first, N - H); i < last; i++)
    {
        filter.slide(0.0f);
        out[i * stride] = filter.center() - filter.get();
    }
}

extern "C"
{

/**
 * Apply median filter to each baseline. The input data are stored
 * channel-major, baseline minor, with a separation of @a stride between
 * rows. Each workitem produces (up to) @a VT channels of output. The
 * input must be suitably padded (in the baseline access) for the number
 * of threads.
 */
__global__ void __launch_bounds__(${wgs}) background_median_filter(
    const float2 * __restrict in, float * __restrict out,
    int channels, int stride, int VT)
{
    int bl = blockDim.x * blockIdx.x + threadIdx.x;
    int sub = blockDim.y * blockIdx.y + threadIdx.y;
    int start_channel = sub * VT;
    int end_channel = min(start_channel + VT, channels);
    medfilt_serial_sliding(in + bl, out + bl, start_channel, end_channel, channels, stride);
}

} // extern C
