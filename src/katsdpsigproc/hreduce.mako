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
 * Device-wide reduction operation, built on top of wg_reduce.mako. Each
 * workgroup is 2D, and each row of the workgroup handles one row of data.
 * The elements in the row are partitioned amongst the workitems (in a strided
 * manner), and the workitem reduces its assigned elements. These partial
 * reductions are then passed to @ref wg_reduce.mako for a work-group local
 * reduction to obtain a final reduced value.
 *
 * Parameters:
 * - type: data type
 * - wgsx, wgsy: number of workitems per workgroup in each dimension
 * - op: expression for combining two values
 * - identity: identity value for @a op
 * - extra_code: arbitrary extra code to add, for use by @a op
 */

<%include file="port.mako"/>
<%namespace name="wg_reduce" file="wg_reduce.mako"/>

${extra_code}

DEVICE_FN ${type} op(${type} a, ${type} b)
{
    return ${op};
}

<%def name="custom_op(a, b, type)">(op((${a}), (${b})))</%def>

${wg_reduce.define_scratch(type, wgsx, 'scratch_t')}
${wg_reduce.define_function(type, wgsx, 'reduce_local', 'scratch_t', custom_op)}

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void hreduce(
    const GLOBAL ${type} * RESTRICT in,
    GLOBAL ${type} * RESTRICT out,
    int start_col,
    int n_cols,
    int in_stride)
{
    LOCAL_DECL scratch_t scratch[${wgsy}];
    int row = get_global_id(1);
    int lid = get_local_id(0);
    int offset = in_stride * row + start_col;

    // Each workitem reduces some elements from its row
    // TODO: could create a separate kernel for the narrow case, to avoid
    // the branch
    ${type} value;
    if (lid >= n_cols)
        value = ${identity};
    else
    {
        value = in[offset + lid];
        for (int c = lid + ${wgsx}; c < n_cols; c += ${wgsx})
        {
            ${type} next = in[offset + c];
            value = op(value, next);
        }
    }

    // Reduce the per-workitem values
    value = reduce_local(value, lid, &scratch[get_local_id(1)]);

    // Write back result
    if (lid == 0)
        out[row] = value;
}
