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
KERNEL REQD_WORK_GROUP_SIZE(${size}, 1, 1) void maskedsum_float2(
    GLOBAL const float2 * RESTRICT in, GLOBAL const float * RESTRICT in_mask,
    GLOBAL float2 * RESTRICT out, int in_stride,
    int Nrows)
{
    int col = get_global_id(0);//block id of processing element
    int row,rowcoloffset;
    float2 value;
    value.x=0.0;
    value.y=0.0;
    for (row=0,rowcoloffset=col;row<Nrows;row++,rowcoloffset+=in_stride)
    {
        value.x = fma(in_mask[row], in[rowcoloffset].x, value.x);
        value.y = fma(in_mask[row], in[rowcoloffset].y, value.y);
    }
    out[col]=value;
}

/**
 * Computes the masked sum of the magnitude of the complex valued input array
 * Row corresponds to channel, column to baseline
 * Sum is calculated along row axis (channels), independently per column (baseline)
 * The shape of 'in' is simplistically (nrows,ncols) however padding is allowed,
 * therefore the true shape is (number of rows, 'in_stride') but only data up to number of columns is considered.
 * Input data in its flattened form is [row0col0, row0col1, row0col2,.., ..padding, row1col0, row1col1, ...]
 * 'in_stride' indexes row1col0 to account for padding
 * 'out' is of shape (ncols of input)
 */
KERNEL REQD_WORK_GROUP_SIZE(${size}, 1, 1) void maskedsumabs_float(
    GLOBAL const float2 * RESTRICT in, GLOBAL const float * RESTRICT in_mask,
    GLOBAL float * RESTRICT out, int in_stride,
    int Nrows)
{
    int col = get_global_id(0);//block id of processing element
    int row,rowcoloffset;
    float value;
    value=0.0;
    for (row=0,rowcoloffset=col;row<Nrows;row++,rowcoloffset+=in_stride)
    {
        value = fma(in_mask[row], sqrt(in[rowcoloffset].x*in[rowcoloffset].x+in[rowcoloffset].y*in[rowcoloffset].y), value);
    }
    out[col]=value;
}
