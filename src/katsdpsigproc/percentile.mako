/*******************************************************************************
 * Copyright (c) 2014-2015, National Research Foundation (SARAO)
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

/**
 * @file
 *
 * Wrapper around rank-finding functions.
 */

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>
#define VT ${vt}

<%def name="define_rankers(type)">

% if is_amplitude:
<% in_type = type %>
DEVICE_FN ${type} amplitude(${in_type} value)
{
    return value; // it is already an amplitude
}

DEVICE_FN ${type} fix_amplitude(${type} value)
{
    return value; // it is already an amplitude
}

% else:

<% in_type = type + '2' %>
/// Compute the amplitude (actually squared amplitude, for efficiency)
DEVICE_FN ${type} amplitude(${in_type} value)
{
    return fma(value.x, value.x, value.y * value.y);
}

/// Turns a value computed by amplitude() into a real amplitude
DEVICE_FN ${type} fix_amplitude(${type} value)
{
    return sqrt(value);
}

% endif

<%rank:ranker_serial_store class_name="ranker_serial_${type}" type="${type}" size="VT"/>

DEVICE_FN void ranker_serial_${type}_init(
    ranker_serial_${type} *self,
    GLOBAL const ${in_type} *data, int start, int step, int N)
{
    int p = start;
    for (int i = 0; i < VT; i++)
    {
        self->values[i] = (p < N) ? amplitude(data[p]) : NAN;
        p += step;
    }
}

<%rank:ranker_parallel class_name="ranker_parallel_${type}" serial_class="ranker_serial_${type}" type="${type}" size="${size}">
    <%def name="thread_id(self)">
        get_local_id(0)
    </%def>
</%rank:ranker_parallel>

DEVICE_FN void ranker_parallel_${type}_init(
    ranker_parallel_${type} *self,
    GLOBAL const ${in_type} * RESTRICT data, int start, int step, int N,
    LOCAL ranker_parallel_${type}_scratch *scratch)
{
    ranker_serial_${type}_init(&self->serial, data, start, step, N);
    self->scratch = scratch;
}
</%def>

<%self:define_rankers type="float"/>

<%rank:find_min_float ranker_class="ranker_parallel_float"/>
<%rank:find_max_float ranker_class="ranker_parallel_float"/>

<%rank:find_rank_float ranker_class="ranker_parallel_float" uniform="True"/>

% if is_amplitude:
<% in_type = 'float' %>
% else:
<% in_type = 'float2' %>
% endif

/**
 * Computes the [0,100,25,75,50] percentiles of the two dimensional array.
 * The lower index element is chosen, and no interpolation is performed.
 * Percentiles are calculated along the columns axis, independently per row,
 * in columns [first_col, first_col + n_cols).
 * Warning: assumes 'in' contains positive numbers only.
 * The shape of 'in' is simplistically (nrows,ncols) however padding is allowed, 
 * therefore the true shape is (number of rows, 'in_stride') but only data up to number of columns is considered.
 * Input data in its flattened form is [row0col0, row0col1, row0col2,.., ..padding, row1col0, row1col1, ...]
 * 'in_stride' indexes row1col0 to account for padding
 * 'out' is of shape (5, nrows of input)
 *
 * Each workgroup processes one or more complete rows.
 */
KERNEL REQD_WORK_GROUP_SIZE(${size}, ${wgsy}, 1) void percentile5_float(
    GLOBAL const ${in_type} * RESTRICT in,
    GLOBAL float * RESTRICT out, int in_stride, int out_stride,
    int first_col, int n_cols)
{
    LOCAL_DECL ranker_parallel_float_scratch scratch[${wgsy}];
    ranker_parallel_float ranker;
    int lid = get_local_id(0);//thread id within processing element
    int ly = get_local_id(1); //row within the set of rows handled by one workgroup
    int row = get_global_id(1);//row within the input and output arrays
    ranker_parallel_float_init(&ranker, in + row * in_stride + first_col,
        get_local_id(0), ${size}, n_cols, &scratch[ly]);

    float perc[5];
    perc[0] = find_min_float(&ranker);
    perc[1] = find_max_float(&ranker);
    perc[2] = find_rank_float(&ranker, (n_cols-1)/4, false);
    perc[3] = find_rank_float(&ranker, ((n_cols-1)*3)/4, false);
    perc[4] = find_rank_float(&ranker, (n_cols-1)/2, false);

    if (lid == 0)
    {
        for (int i = 0; i < 5; i++)
            out[row + i * out_stride] = fix_amplitude(perc[i]);
    }
}
