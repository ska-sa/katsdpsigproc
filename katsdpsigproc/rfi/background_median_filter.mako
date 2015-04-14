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
 * Each thread processes a contiguous part of the band for a baseline, using
 * a sliding-window median filter.
 */

<%include file="/port.mako"/>

#define WIDTH ${width}

% if is_amplitude:

typedef float in_type;
DEVICE_FN float amplitude(in_type v)
{
    return v;
}

% else:

typedef float2 in_type;
/// Amplitude of a visibility
DEVICE_FN float amplitude(in_type v)
{
    return hypot(v.x, v.y);
}

% endif

/**
 * Serial sliding-window median filter. New elements are added using
 * @ref slide, and the median of the last @a WIDTH elements is retrieved
 * by calling @ref get.
 */
typedef struct median_filter
{
    /**
     * History of samples. @a samples[0] is the oldest.
     */
    float samples[WIDTH];

    /**
     * Rank of each sample (0 being smallest). When there are ties, the
     * oldest sample is considered smaller.
     */
    int rank[WIDTH];
} median_filter;

/**
 * Initialise the filter using zero-valued samples.
 */
DEVICE_FN void median_filter_init(median_filter *self)
{
    for (int i = 0; i < WIDTH; i++)
    {
        self->samples[i] = 0;
        self->rank[i] = i;
    }
}

/// Return the median of the current samples
DEVICE_FN float median_filter_get(const median_filter *self)
{
    const int H = WIDTH / 2;
    float result = 0.0f;
    for (int j = 0; j < WIDTH; j++)
    {
        result = (self->rank[j] == H) ? self->samples[j] : result;
    }
    return result;
}

DEVICE_FN float median_filter_center(const median_filter *self)
{
    return self->samples[WIDTH / 2];
}

DEVICE_FN void median_filter_slide(median_filter *self, float new_sample)
{
    float old_sample = self->samples[0];
    int new_rank = WIDTH - 1;
#pragma unroll
    for (int j = 0; j < WIDTH - 1; j++)
    {
        self->samples[j] = self->samples[j + 1];
        int cmp = (new_sample < self->samples[j]);
        self->rank[j] = self->rank[j + 1] + cmp - (old_sample <= self->samples[j]);
        new_rank -= cmp;
    }
    self->samples[WIDTH - 1] = new_sample;
    self->rank[WIDTH - 1] = new_rank;
}

/**
 * Apply the median filter on a single thread. The range of output
 * values to produce is [@a first, @a last), out of a total array of
 * size @a N.
 */
DEVICE_FN static void medfilt_serial_sliding(
    const GLOBAL in_type * RESTRICT in, GLOBAL float * __restrict out,
    int first, int last, int N, int stride)
{
    const int H = WIDTH / 2;
    median_filter filter;
    median_filter_init(&filter);

    // Load the initial window, substituting zeros beyond the ends.
    // These is no need for this on the leading edge, because the
    // constructor initialises with zero samples.
    for (int i = max(0, first - H); i < min(first + H, N); i++)
        median_filter_slide(&filter, amplitude(in[i * stride]));
    for (int i = N; i < first + H; i++)
        median_filter_slide(&filter, 0.0f);

    for (int i = first; i < min(last, N - H); i++)
    {
        median_filter_slide(&filter, amplitude(in[(i + H) * stride]));
        out[i * stride] = median_filter_center(&filter) - median_filter_get(&filter);
    }
    for (int i = max(first, N - H); i < last; i++)
    {
        median_filter_slide(&filter, 0.0f);
        out[i * stride] = median_filter_center(&filter) - median_filter_get(&filter);
    }
}

/**
 * Apply median filter to each baseline. The input data are stored
 * channel-major, baseline minor, with a separation of @a stride between
 * rows. Each workitem produces (up to) @a VT channels of output. The
 * input and output must be suitably padded (in the baseline axis) for the
 * number of threads.
 */
KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1) void background_median_filter(
    const GLOBAL in_type * RESTRICT in, GLOBAL float * RESTRICT out,
    int channels, int stride, int VT)
{
    int bl = get_global_id(0);
    int sub = get_global_id(1);
    int start_channel = sub * VT;
    int end_channel = min(start_channel + VT, channels);
    medfilt_serial_sliding(
        in + bl, out + bl,
        start_channel, end_channel, channels, stride);
}
