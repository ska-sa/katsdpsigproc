/*******************************************************************************
 * Copyright (c) 2014-2017, National Research Foundation (SARAO)
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
 * 2D transposition metakernel. A square block of work items loads a tile of
 * data from global memory into local memory, and then it is written back
 * again. The tile may be multiple times the size of the work group, in
 * which case it is loaded as several subtiles. This amortises some fixed
 * costs.
 *
 * The tiles are mapped in a "diagonal" fashion, which prevents some
 * imbalances between memory channels.
 *
 * This file defines tools for creating custom kernels that do transposition
 * combined with other operations. For a ready-to-use implementation, see
 * @ref transpose.mako.
 */

<%def name="transpose_data_class(class_name, type, block, vtx, vty)">
/// Stores actual data (allocate in local memory)
typedef struct
{
    // The inner dimension is padded so that column-major accesses will
    // hit different banks, for 4-byte banks and 1, 2 or 4-byte elements.
    ${type} arr[${block * vty}][${block * vtx} + (sizeof(${type}) > 4 ? 1 : 4 / sizeof(${type}))];
} ${class_name};
</%def>

<%def name="transpose_coords_class(class_name, block, vtx, vty)">
#ifndef TRANSPOSE_COORDS_DEFINED
#define TRANSPOSE_COORDS_DEFINED
/// Stores addressing information (allocate in private memory)
typedef struct
{
    int lx;    /// local X coordinate within a block
    int ly;    /// local Y coordinate within a block
    int in_row0; /// first row in input for the block
    int in_col0; /// first row in output for the block
} transpose_coords;
#endif

/**
 * Prepare coordinates.
 *
 * @param local_x, local_y  Local coordinates within a block (lx should be fastest varying)
 * @param block_x, block_y  Coordinates of the block
 * @param blocks_y          Number of y blocks
 */
DEVICE_FN void ${class_name}_init(
    transpose_coords *coords,
    int local_x, int local_y, int block_x, int block_y, int blocks_y)
{
    coords->lx = local_x;
    coords->ly = local_y;
    // Compute origin of the tile, with diagonal addressing
    coords->in_row0 = (block_x + block_y) % blocks_y * ${block * vty};
    coords->in_col0 = block_x * ${block * vtx};
}

/// Prepare coordinates, using the local and group coordinates as parameters
DEVICE_FN void ${class_name}_init_simple(transpose_coords *coords)
{
    ${class_name}_init(
        coords,
        get_local_id(0), get_local_id(1),
        get_group_id(0), get_group_id(1),
        get_num_groups(1));
}
</%def>

/**
 * Load data (using the caller's body) and put it in local memory.
 */
<%def name="transpose_load(coords, block, vtx, vty)">
{
    int transpose_r[${vty}];
    int transpose_c[${vtx}];
% for i in range(vty):
    transpose_r[${i}] = ${coords}.in_row0 + ${i * block} + ${coords}.ly;
% endfor
% for i in range(vtx):
    transpose_c[${i}] = ${coords}.in_col0 + ${i * block} + ${coords}.lx;
% endfor

% for y in range(vty):
% for x in range(vtx):
    {
        ${caller.body(
            "(transpose_r[{y}])".format(y=y),
            "(transpose_c[{x}])".format(x=x),
            "({coords}.ly + {yofs})".format(coords=coords, yofs=y * block),
            "({coords}.lx + {xofs})".format(coords=coords, xofs=x * block))}
    }
% endfor
% endfor
}
</%def>

<%def name="transpose_store(coords, block, vtx, vty)">
{
    int transpose_r[${vtx}];
    int transpose_c[${vty}];
% for i in range(vtx):
    transpose_r[${i}] = ${coords}.in_col0 + ${i * block} + ${coords}.ly;
% endfor
% for i in range(vty):
    transpose_c[${i}] = ${coords}.in_row0 + ${i * block} + ${coords}.lx;
% endfor

% for y in range(vtx):
% for x in range(vty):
    {
        ${caller.body(
            "(transpose_r[{y}])".format(y=y),
            "(transpose_c[{x}])".format(x=x),
            "({coords}.lx + {xofs})".format(coords=coords, xofs=x * block),
            "({coords}.ly + {yofs})".format(coords=coords, yofs=y * block))}
    }
% endfor
% endfor
}
</%def>
