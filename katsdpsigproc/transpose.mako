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
 * imbalanced between memory channels.
 *
 * It could potentially benefit from specialisations for the case where
 * padding is available, to avoid the need for conditional accesses.
 *
 * Mako parameters:
 * - @a ctype: C type of the elements
 * - @a block: Each thread block processes @a block x @a block elements
 * - @a vtx, vty: number of items per thread in each dimension
 */

<%include file="/port.mako"/>

#define BLOCK ${block}
#define VTX ${vtx}
#define VTY ${vty}
#define TILEX (VTX * BLOCK)
#define TILEY (VTY * BLOCK)
typedef ${ctype} T;

KERNEL REQD_WORK_GROUP_SIZE(BLOCK, BLOCK, 1) void transpose(
    GLOBAL T *out,
    const GLOBAL T * RESTRICT in,
    int in_rows,
    int in_cols,
    int out_stride,
    int in_stride)
{
    // The inner dimension is padded so that column-major accesses will
    // hit different banks, for 4-byte banks and 1, 2 or 4-byte elements.
    LOCAL_DECL T arr[TILEY][TILEX + (sizeof(T) > 4 ? 1 : 4 / sizeof(T))];

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Compute origin of the tile, with diagonal addressing
    int in_row0 = (get_group_id(0) + get_group_id(1)) % get_num_groups(1) * TILEY;
    int in_col0 = get_group_id(0) * TILEX;

    // Load a chunk into shared memory
    {
        int r[VTY];
        int c[VTX];
% for i in range(vty):
        r[${i}] = in_row0 + ${i} * BLOCK + ly;
% endfor
% for i in range(vtx):
        c[${i}] = in_col0 + ${i} * BLOCK + lx;
% endfor

% for y in range(vty):
% for x in range(vtx):
        if (r[${y}] < in_rows && c[${x}] < in_cols)
            arr[ly + ${y} * BLOCK][lx + ${x} * BLOCK] =
                in[r[${y}] * in_stride + c[${x}]];
% endfor
% endfor
    }

    BARRIER();

    // Write chunk back to global memory, transposed
    {
        int r[VTX];
        int c[VTY];
% for i in range(vtx):
        r[${i}] = in_col0 + ${i} * BLOCK + ly;
% endfor
% for i in range(vty):
        c[${i}] = in_row0 + ${i} * BLOCK + lx;
% endfor

% for y in range(vtx):
% for x in range(vty):
        if (r[${y}] < in_cols && c[${x}] < in_rows)
            out[r[${y}] * out_stride + c[${x}]] =
                arr[lx + ${x} * BLOCK][ly + ${y} * BLOCK];
% endfor
% endfor
    }
}
