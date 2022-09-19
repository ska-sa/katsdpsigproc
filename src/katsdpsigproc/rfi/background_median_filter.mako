/*******************************************************************************
 * Copyright (c) 2014-2019, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

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
 *
 * Optionally, per-channel flags can be passed in, indicating known-bad values
 * that should be ignored for computing the background. Any non-zero flag value
 * indicates that the corresponding visibility should be ignored.
 *
 * If is_amplitude is true, then the inputs are the amplitudes rather than
 * complex values. In this case, negative values are also treated as flagged,
 * regardless of whether flag booleans are provided.
 *
 * The output is the difference between the original and filtered amplitudes.
 * Flagged values in the input are mapped to zero in the output.
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
 *
 * Flagged values are indicated by passing a negative value. Only the unflagged
 * values are considered when computing the median. If there are an even number
 * of values, the average of the two medians is returned. If there are no valid
 * values, the result is undefined.
 */
typedef struct median_filter
{
    /**
     * History of samples. @a samples[0] is the oldest.
     */
    float samples[WIDTH];

    /**
     * Rank of each sample (0 being largest). When there are ties, the
     * oldest sample is considered larger.
     */
    int rank[WIDTH];

    /// Number of elements in samples that are non-negative
    int num_valid;
} median_filter;

/**
 * Initialise the filter using invalid samples.
 */
DEVICE_FN void median_filter_init(median_filter *self)
{
    for (int i = 0; i < WIDTH; i++)
    {
        self->samples[i] = -1;
        self->rank[i] = i;
    }
    self->num_valid = 0;
}

/// Return the median of the current samples
DEVICE_FN float median_filter_get(const median_filter *self)
{
    int lo = (self->num_valid - 1) / 2;
    int hi = self->num_valid / 2;
    float result = 0.0f;
    for (int j = 0; j < WIDTH; j++)
    {
        if (self->rank[j] == lo)
            result += self->samples[j];
        if (self->rank[j] == hi)
            result += self->samples[j];
    }
    return result * 0.5f;
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
        int cmp = (new_sample > self->samples[j]);
        self->rank[j] = self->rank[j + 1] + cmp - (old_sample >= self->samples[j]);
        new_rank -= cmp;
    }
    self->samples[WIDTH - 1] = new_sample;
    self->rank[WIDTH - 1] = new_rank;
    self->num_valid += (new_sample >= 0.0f) - (old_sample >= 0.0f);
}

/**
 * Apply the median filter on a single thread. The range of output
 * values to produce is [@a first, @a last), out of a total array of
 * size @a N.
 */
DEVICE_FN static void medfilt_serial_sliding(
    const GLOBAL in_type * RESTRICT in, GLOBAL float * RESTRICT out,
% if use_flags:
    const GLOBAL unsigned char * RESTRICT flags,
% endif
    int first, int last, int N, int stride)
{
% if use_flags == BackgroundFlags.FULL:
#define AMPLITUDE(idx) (flags[(idx) * stride] ? -1.0f : amplitude(in[(idx) * stride]))
% elif use_flags == BackgroundFlags.CHANNEL:
#define AMPLITUDE(idx) (flags[(idx)] ? -1.0f : amplitude(in[(idx) * stride]))
% else:
#define AMPLITUDE(idx) amplitude(in[(idx) * stride])
% endif
    const int H = WIDTH / 2;
    median_filter filter;
    median_filter_init(&filter);

    // Load the initial window, substituting invalid values beyond the ends.
    // These is no need for this on the leading edge, because the
    // constructor initialises with invalid samples.
    for (int i = max(0, first - H); i < min(first + H, N); i++)
        median_filter_slide(&filter, AMPLITUDE(i));
    for (int i = N; i < first + H; i++)
        median_filter_slide(&filter, -1.0f);

    for (int i = first; i < min(last, N - H); i++)
    {
        median_filter_slide(&filter, AMPLITUDE(i + H));
        float cur = median_filter_center(&filter);
        out[i * stride] = cur >= 0.0f ? cur - median_filter_get(&filter) : 0.0f;
    }
    for (int i = max(first, N - H); i < last; i++)
    {
        median_filter_slide(&filter, -1.0f);
        float cur = median_filter_center(&filter);
        out[i * stride] = cur >= 0.0f ? cur - median_filter_get(&filter) : 0.0f;
    }
#undef AMPLITUDE
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
% if use_flags:
    const GLOBAL unsigned char * flags,
% endif
    int channels, int stride,
    int VT)
{
    int bl = get_global_id(0);
    int sub = get_global_id(1);
    int start_channel = sub * VT;
    int end_channel = min(start_channel + VT, channels);
    medfilt_serial_sliding(
        in + bl, out + bl,
% if use_flags == BackgroundFlags.CHANNEL:
        flags,
% elif use_flags == BackgroundFlags.FULL:
        flags + bl,
% endif
        start_channel, end_channel, channels, stride);
}
