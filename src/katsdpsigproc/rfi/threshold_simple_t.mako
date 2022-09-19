/*******************************************************************************
 * Copyright (c) 2014, National Research Foundation (SARAO)
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
 * Apply a global threshold to each sample independently,
 * with transposed data.
 *
 * Mako template arguments:
 *  - wgsx, wgsy
 *  - flag_value
 */

<%include file="/port.mako"/>

KERNEL void REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) threshold_simple_t(
    GLOBAL const float * RESTRICT deviations,
    GLOBAL const float * RESTRICT noise,
    GLOBAL unsigned char * RESTRICT flags,
    unsigned int stride,
    float n_sigma)
{
    int bl = get_global_id(1);
    int channel = get_global_id(0);
    // TODO: threshold could be loaded by one thread
    // and broadcast through local memory
    float threshold = n_sigma * noise[bl];
    int addr = bl * stride + channel;
    flags[addr] = (deviations[addr] > threshold) ? ${flag_value} : 0;
}
