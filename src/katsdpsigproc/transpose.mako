/*******************************************************************************
 * Copyright (c) 2014-2020, National Research Foundation (SARAO)
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
 * 2D transposition kernel. A square block of work items loads a tile of
 * data from global memory into local memory, and then it is written back
 * again. The tile may be multiple times the size of the work group, in
 * which case it is loaded as several subtiles. This amortises some fixed
 * costs.
 *
 * The tiles are mapped in a "diagonal" fashion, which prevents some
 * imbalances between memory channels.
 *
 * It could potentially benefit from specializations for the case where
 * padding is available, to avoid the need for conditional accesses.
 *
 * Mako parameters:
 * - @a ctype: C type of the elements
 * - @a block: Each thread block processes @a block x @a block elements
 * - @a vtx, vty: number of items per thread in each dimension
 */

<%include file="/port.mako"/>
<%namespace name="transpose" file="transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_values" type="${ctype}" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void transpose(
    GLOBAL ${ctype} *out,
    const GLOBAL ${ctype} * RESTRICT in,
    int in_rows,
    int in_cols,
    int out_stride,
    int in_stride)
{
    LOCAL_DECL transpose_values values;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);

    // Load a chunk into shared memory
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        if (${r} < in_rows && ${c} < in_cols)
        {
            values.arr[${lr}][${lc}] = in[${r} * in_stride + ${c}];
        }
    </%transpose:transpose_load>

    BARRIER();

    // Write chunk back to global memory, transposed
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        if (${r} < in_cols && ${c} < in_rows)
        {
            out[${r} * out_stride + ${c}] = values.arr[${lr}][${lc}];
        }
    </%transpose:transpose_store>
}
