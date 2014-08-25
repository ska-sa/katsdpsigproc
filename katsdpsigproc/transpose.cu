#define BLOCK 32

typedef ${ctype} T;

__global__ void transpose(
    T *out,
    const T * __restrict in,
    int in_rows,
    int in_cols,
    int out_stride,
    int in_stride)
{
    __shared__ T arr[BLOCK][BLOCK + 1];

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int in_row0 = blockIdx.y * blockDim.y;
    int in_row = in_row0 + ly;
    int in_col0 = blockIdx.x * blockDim.x;
    int in_col = in_col0 + lx;
    if (in_row < in_rows && in_col < in_cols)
        arr[ly][lx] = in[in_row * in_stride + in_col];

    __syncthreads();

    int out_row = in_col0 + ly;
    int out_col = in_row0 + lx;
    if (out_row < in_cols && out_col < in_rows)
        out[out_row * out_stride + out_col] = arr[lx][ly];
}
