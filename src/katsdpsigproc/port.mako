/*******************************************************************************
 * Copyright (c) 2014-2021, National Research Foundation (SARAO)
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

## Macros that abstract the differences between OpenCL and CUDA.

#ifndef PORT_MAKO
#define PORT_MAKO

#ifdef __OPENCL_VERSION__

#define DEVICE_FN
#define KERNEL __kernel
#define GLOBAL_DECL __global
#define GLOBAL __global
#define LOCAL_DECL __local
#define LOCAL __local
#define BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define RESTRICT restrict
#define REQD_WORK_GROUP_SIZE(x, y, z) __attribute__((reqd_work_group_size(x, y, z)))
#define SHUFFLE_AVAILABLE 0

% for type in ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'float']:

DEVICE_FN ${type}2 make_${type}2(${type} x, ${type} y)
{
    return (${type}2) (x, y);
}

DEVICE_FN ${type}4 make_${type}4(${type} x, ${type} y, ${type} z, ${type} w)
{
    return (${type}4) (x, y, z, w);
}

% endfor

#else

#include <math.h>
#include <float.h>
#include <stdio.h>

/* System headers may provide some of these, but it's OS dependent. It's
 * legal to repeat typedefs, so make sure that they're all available for
 * consistency with OpenCL.
 */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#define DEVICE_FN __device__
#define KERNEL __global__
#define GLOBAL_DECL __global__
#define GLOBAL
#define LOCAL_DECL __shared__
#define LOCAL
#define BARRIER() __syncthreads()
#define RESTRICT __restrict
#define REQD_WORK_GROUP_SIZE(x, y, z) __launch_bounds__((x) * (y) * (z))

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
# define SHUFFLE_AVAILABLE 1
#else
# define SHUFFLE_AVAILABLE 0
#endif

__device__ static inline unsigned int get_local_id(int dim)
{
    return dim == 0 ? threadIdx.x : dim == 1 ? threadIdx.y : threadIdx.z;
}

__device__ static inline unsigned int get_group_id(int dim)
{
    return dim == 0 ? blockIdx.x : dim == 1 ? blockIdx.y : blockIdx.z;
}

__device__ static inline unsigned int get_local_size(int dim)
{
    return dim == 0 ? blockDim.x : dim == 1 ? blockDim.y : blockDim.z;
}

__device__ static inline unsigned int get_global_id(int dim)
{
    return get_group_id(dim) * get_local_size(dim) + get_local_id(dim);
}

__device__ static inline unsigned int get_num_groups(int dim)
{
    return dim == 0 ? gridDim.x : dim == 1 ? gridDim.y : gridDim.z;
}

__device__ static inline float as_float(unsigned int x)
{
    return __int_as_float(x);
}

__device__ static inline int as_int(float x)
{
    return __float_as_int(x);
}

#endif /* CUDA */
#endif /* PORT_MAKO */
