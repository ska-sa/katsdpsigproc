/*******************************************************************************
 * Copyright (c) 2014, National Research Foundation (SARAO)
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

<%include file="/port.mako"/>

#define VT ${vt}
#define WGSX ${wgsx}

<%namespace name="wg_reduce" file="/wg_reduce.mako"/>
<%namespace name="rank" file="/rank.mako"/>

<%rank:ranker_serial_store class_name="ranker_abs_serial" type="float" size="VT"/>

DEVICE_FN void ranker_abs_serial_init(
    ranker_abs_serial *self,
    GLOBAL const float *data, int start, int step, int N)
{
    int p = start;
    for (int i = 0; i < VT; i++)
    {
        self->values[i] = (p < N) ? fabs(data[p]) : NAN;
        p += step;
    }
}

<%rank:ranker_parallel class_name="ranker_abs_parallel" serial_class="ranker_abs_serial" type="float" size="${wgsx}">
    <%def name="thread_id(self)">
        get_local_id(0)
    </%def>
</%rank:ranker_parallel>

DEVICE_FN void ranker_abs_parallel_init(
    ranker_abs_parallel *self,
    GLOBAL const float * RESTRICT data, int start, int step, int N,
    LOCAL ranker_abs_parallel_scratch *scratch)
{
    ranker_abs_serial_init(&self->serial, data, start, step, N);
    self->scratch = scratch;
}

<%rank:median_non_zero_float ranker_class="ranker_abs_parallel" uniform="${True}"/>

KERNEL REQD_WORK_GROUP_SIZE(WGSX, 1, 1) void madnz_t(
    GLOBAL const float * RESTRICT in,
    GLOBAL float * RESTRICT noise,
    int channels, int stride)
{
    LOCAL_DECL ranker_abs_parallel_scratch scratch;

    int bl = get_group_id(1);
    ranker_abs_parallel ranker;
    ranker_abs_parallel_init(
        &ranker, in + bl * stride, get_local_id(0),
        WGSX, channels, &scratch);
    float s = 1.4826f * median_non_zero_float(&ranker, channels);
    if (get_local_id(0) == 0)
        noise[bl] = s;
}
