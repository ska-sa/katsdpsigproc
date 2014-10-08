/**
 * @file
 *
 * Simple transposition kernel. It is not particularly optimal, and could benefit from some
 * thread coarsening, and possibly some specialisation to eliminate branches if there is
 * suitable padding.
 *
 * Mako parameters:
 * - @a ctype: C type of the elements
 * - @a block: Each thread block processes @a block x @a block elements
 */

<%include file="/port.mako"/>

#define BLOCK ${block}
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
    LOCAL_DECL T arr[BLOCK][BLOCK + (sizeof(T) > 4 ? 1 : 4 / sizeof(T))];

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Load a chunk into shared memory
    int in_row0 = get_group_id(1) * BLOCK;
    int in_row = in_row0 + ly;
    int in_col0 = get_group_id(0) * BLOCK;
    int in_col = in_col0 + lx;
    if (in_row < in_rows && in_col < in_cols)
        arr[ly][lx] = in[in_row * in_stride + in_col];

    BARRIER();

    // Write chunk bank to global memory, transposed
    int out_row = in_col0 + ly;
    int out_col = in_row0 + lx;
    if (out_row < in_cols && out_col < in_rows)
        out[out_row * out_stride + out_col] = arr[lx][ly];
}
