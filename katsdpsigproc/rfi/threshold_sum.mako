/**
 * @file
 *
 * Thresholder using Sum-Threshold method (Offringa 2010). It takes the data in
 * transposed form (baseline-major).
 *
 * Each workgroup processes a contiguous portion of a baseline. Sums are
 * computed in local memory, Kogge-Stone style, and flags are dispersed in a
 * similar way.
 *
 * Mako parameters:
 *  - wgsx, wgsy: work group size (typically wgsy=1, but larger is supported)
 *  - windows: number of window sizes to use
 *  - flag_value
 */

<%include file="/port.mako"/>

<%
max_window = 2 ** (windows - 1)
edge_size = 2 ** windows - windows - 1
chunk = wgsx - 2 * edge_size
%>

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define MAX_WINDOW ${max_window}
#define EDGE_SIZE ${edge_size}
#define CHUNK (WGSX - 2 * EDGE_SIZE)

KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1) void threshold_sum(
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
        float sum[WGSX + MAX_WINDOW];
        int flag[WGSX + MAX_WINDOW];
    } lcl;

    int lid = get_local_id(0);
    int chunk_start = get_group_id(0) * CHUNK;
    int channel = chunk_start - EDGE_SIZE + lid;

    int bl = get_global_id(1);
    float bl_noise = noise[bl];
    int addr = bl * stride + channel;

    float value = (channel >= 0 && channel < channels) ? deviations[addr] : 0.0f;
    bool flag = false;

% for w in range(windows):
    {
        const int w = ${w};
        const int window = 1 << w;
        float threshold = bl_noise * n_sigma${w};
        float sum = flag ? threshold : value;
% for i in range(w):
        {
            lcl.sum[lid + MAX_WINDOW] = sum;
            BARRIER();
            sum += lcl.sum[lid + (MAX_WINDOW - (1 << ${i}))];
            BARRIER();
        }
% endfor
        bool new_flag = sum > threshold * window;
% for i in range(w):
        {
            lcl.flag[lid] = new_flag;
            BARRIER();
            new_flag |= lcl.flag[lid + (1 << ${i})];
% if w < windows - 1 or i < w - 1:
            BARRIER(); // skipped on very final iteration
% endif
        }
% endfor
        flag |= new_flag;
    }
% endfor

    int chunk_end = min(chunk_start + CHUNK, channels);
    if (channel >= chunk_start && channel < chunk_end)
        flags[addr] = flag ? ${flag_value} : 0;
}
