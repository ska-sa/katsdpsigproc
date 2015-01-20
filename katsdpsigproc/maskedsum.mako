/**
 * @file
 *
 * Kernel function for summing floating point arrays using a mask
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
 */
KERNEL REQD_WORK_GROUP_SIZE(${size}, 1, 1) void maskedsum_float(
    GLOBAL const float * RESTRICT in, GLOBAL const float * RESTRICT in_mask,
    GLOBAL float * RESTRICT out, int in_stride,
    int Nrows)
{
    int blockid = get_global_id(1);//block id of processing element 
    int col = blockid*${size}+get_local_id(0);//thread id within processing element
    int row,rowcoloffset;
    float value=0.0;
    for (row=0,rowcoloffset=col;row<Nrows;row++,rowcoloffset+=in_stride)
        value = fma(in_mask[row], in[rowcoloffset], value); //value+=in_mask[row]*in[rowcoloffset];
    out[col]=value;
}
