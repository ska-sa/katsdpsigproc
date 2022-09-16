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

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>

/**
 * Encapsulates a section of a strided (non-contiguous) 1D array.
 */
typedef struct array_piece
{
    const GLOBAL float *in;
    int start;
    int end;
    int stride;
} array_piece;

DEVICE_FN void array_piece_init(
    array_piece *self,
    const GLOBAL float *in, int start, int end, int stride)
{
    self->in = in;
    self->start = start;
    self->end = end;
    self->stride = stride;
}

DEVICE_FN float array_piece_get(const array_piece *self, int idx)
{
    return self->in[idx * self->stride];
}

<%rank:ranker_serial class_name="ranker_abs_serial" type="float">
    <%def name="foreach(self, start=0, stop=None)">
        <%
        if stop is None:
            stop = '({0})->piece.end'.format(self)
        else:
            stop = '({0})->piece.start + ({1})'.format(self, stop)
        %>
        for (int i = (${self})->piece.start + (${start}); i < ${stop}; i++)
        {
            ${caller.body('fabs(array_piece_get(&(%s)->piece, i))' % (self,))}
        }
    </%def>
    array_piece piece;
</%rank:ranker_serial>

DEVICE_FN void ranker_abs_serial_init(ranker_abs_serial *self, const array_piece *piece)
{
    self->piece = *piece;
}

<%rank:ranker_parallel class_name="ranker_abs_parallel" serial_class="ranker_abs_serial" type="float" size="${wgsy}">
    <%def name="thread_id(self)">
        get_local_id(1)
    </%def>
</%rank:ranker_parallel>
DEVICE_FN void ranker_abs_parallel_init(
    ranker_abs_parallel *self,
    const array_piece *piece,
    LOCAL ranker_abs_parallel_scratch *scratch)
{
    ranker_abs_serial_init(&self->serial, piece);
    self->scratch = scratch;
}

<%rank:median_non_zero_float ranker_class="ranker_abs_parallel" uniform="${False}"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void madnz(
    const GLOBAL float * RESTRICT in,
    GLOBAL float * RESTRICT noise,
    int channels, int stride,
    int VT)
{
    LOCAL_DECL ranker_abs_parallel_scratch scratch[${wgsx}];

    int bl = get_global_id(0);
    int start = get_local_id(1) * VT;
    int end = min(start + VT, channels);
    array_piece piece;
    array_piece_init(&piece, in + bl, start, end, stride);
    ranker_abs_parallel ranker;
    ranker_abs_parallel_init(&ranker, &piece, scratch + get_local_id(0));
    float s = 1.4826 * median_non_zero_float(&ranker, channels);
    if (start == 0)
        noise[bl] = s;
}
