/**
 * @file
 *
 * Thresholder using Sum-Threshold method (Offringa 2010). It takes the data in
 * transposed form (baseline-major).
 *
 * Each workgroup processes a contiguous portion of a baseline. Sums are
 * computed in local memory, Kogge-Stone style, and flags are dispersed in a
 * similar way. Each work-group produces CHUNK output values, starting with
 * WGS * VT input values. It actually computes an output value for each
 * input, but the ones on the edge depend on undefined values and so are
 * meaningless.  The VT values per workitem are strided, so that global and
 * local memory accesses are contiguous.
 *
 * At the time of writing, performance on a GeForce 770 GTX is limited by local
 * memory bandwidth. It may be possible that using a different mapping of
 * workitems to data would allow more efficient summation by doing some of the
 * work internally to a thread, instead of between threads. CUDA-specific
 * shuffle instructions could also help. However, this algorithm will probably
 * not be the bottleneck.
 *
 * The profiler also indicates that there are some global memory alignment
 * issues. This could be fixed by ensuring an aligned stride, plus using a
 * CHUNK size that is a multiple of a suitable power of 2 (which might
 * require wasting some work-items in each work-group) and hence doing more
 * work.
 *
 * Mako parameters:
 *  - wgs: work group size
 *  - vt: number of items to process per work item
 *  - windows: number of window sizes to use
 *  - flag_value
 */

<%include file="/port.mako"/>

<%
max_window = 2 ** (windows - 1)
edge_size = 2 ** windows - windows - 1
chunk = wgs - 2 * edge_size
%>

#define WGS ${wgs}
#define VT ${vt}
#define MAX_WINDOW ${max_window}
#define EDGE_SIZE ${edge_size}
#define CHUNK (WGS * VT - 2 * EDGE_SIZE)

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void threshold_sum(
    GLOBAL const float * RESTRICT deviations,
    GLOBAL const float * RESTRICT noise,
    GLOBAL unsigned char * RESTRICT flags,
    int channels,
    int stride
% for w in range(windows):
    , float n_sigma${w}
% endfor
    )
{
    LOCAL_DECL union
    {
        // The extra MAX_WINDOW has no useful data, but allows some conditionals
        // to be avoided
        float sum[WGS * VT + MAX_WINDOW];
        int flag[WGS * VT + MAX_WINDOW];
    } lcl;

    int lid = get_local_id(0);
    int chunk_start = get_group_id(0) * CHUNK;
    int channel0 = chunk_start - EDGE_SIZE + lid;

    int bl = get_global_id(1);
    float bl_noise = noise[bl];

    float value[VT];
    bool flag[VT];
    for (int j = 0; j < VT; j++)
    {
        int channel = j * WGS + channel0;
        value[j] = (channel >= 0 && channel < channels) ? deviations[bl * stride + channel] : 0.0f;
        flag[j] = false;
    }

% for w in range(windows):
    {
        const int w = ${w};
        const int window = 1 << w;
        float threshold = bl_noise * n_sigma${w};
        float sum[VT];
        for (int j = 0; j < VT; j++)
            sum[j] = flag[j] ? threshold : value[j];
% for i in range(w):
        {
            for (int j = 0; j < VT; j++)
                lcl.sum[lid + (j * WGS + MAX_WINDOW)] = sum[j];
            BARRIER();
            for (int j = 0; j < VT; j++)
                sum[j] += lcl.sum[lid + (j * WGS + MAX_WINDOW - (1 << ${i}))];
            BARRIER();
        }
% endfor
        threshold *= window; // threshold on sum, instead of average
        bool new_flag[VT];
        for (int j = 0; j < VT; j++)
            new_flag[j] = sum[j] > threshold;
% for i in range(w):
        {
            for (int j = 0; j < VT; j++)
                lcl.flag[lid + j * WGS] = new_flag[j];
            BARRIER();
            for (int j = 0; j < VT; j++)
                new_flag[j] |= lcl.flag[lid + (j * WGS + (1 << ${i}))];
% if w < windows - 1 or i < w - 1:
            BARRIER(); // skipped on very final iteration
% endif
        }
% endfor
        for (int j = 0; j < VT; j++)
            flag[j] |= new_flag[j];
    }
% endfor

    int chunk_end = min(chunk_start + CHUNK, channels);
    for (int j = 0; j < VT; j++)
    {
        int channel = channel0 + j * WGS;
        if (channel >= chunk_start && channel < chunk_end)
        {
            flags[bl * stride + channel] = flag[j] ? ${flag_value} : 0;
        }
    }
}
