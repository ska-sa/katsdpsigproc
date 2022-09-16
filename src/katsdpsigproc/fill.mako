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
 * Fill an array with a constant value. This is similar to clEnqueueFillBuffer,
 * but it uses a kernel so that it works on OpenCL 1.1 and on CUDA.
 */

<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1) void fill(
    GLOBAL ${ctype} * RESTRICT out, unsigned int size, ${ctype} value)
{
    unsigned int gid = get_global_id(0);
    if (gid < size)
        out[gid] = value;
}
