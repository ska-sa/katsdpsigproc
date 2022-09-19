/*******************************************************************************
 * Copyright (c) 2015-2017, National Research Foundation (SARAO)
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
 * Kernel function for summing floating point arrays using a mask.
 */

<%include file="/port.mako"/>

/**
 * Computes the masked sum of input array
 * Row corresponds to channel, column to baseline
 * Sum is calculated along row axis (channels), independently per column (baseline)
 * The shape of 'in' is simplistically (nrows,ncols) however padding is allowed, 
 * therefore the true shape is (number of rows, 'in_stride') but only data up to number of columns is considered.
 * Input data in its flattened form is [row0col0, row0col1, row0col2,.., ..padding, row1col0, row1col1, ...]
 * 'in_stride' indexes row1col0 to account for padding
 * 'out' is of shape (ncols of input)
 *
 * If @a use_amplitudes is true, then the sum is taken over the amplitudes of
 * the input complex values, rather than the complex values themselves.
 */
KERNEL REQD_WORK_GROUP_SIZE(${size}, 1, 1) void maskedsum_float(
    GLOBAL const float2 * RESTRICT in, GLOBAL const float * RESTRICT in_mask,
% if use_amplitudes:
    GLOBAL float * RESTRICT out,
% else:
    GLOBAL float2 * RESTRICT out,
% endif
    int in_stride,
    int Nrows)
{
    int col = get_global_id(0);//block id of processing element
    int row, rowcoloffset;
% if use_amplitudes:
    float value = 0.0f;
% else:
    float2 value;
    value.x = 0.0f;
    value.y = 0.0f;
% endif
    for (row = 0, rowcoloffset = col; row < Nrows; row++, rowcoloffset += in_stride)
    {
        float2 c = in[rowcoloffset];
% if use_amplitudes:
        value = fma(in_mask[row], sqrt(c.x * c.x + c.y * c.y), value);
% else:
        value.x = fma(in_mask[row], c.x, value.x);
        value.y = fma(in_mask[row], c.y, value.y);
% endif
    }
    out[col] = value;
}
