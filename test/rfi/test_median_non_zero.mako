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
 * @file
 *
 * Test for code in @ref threshold_mad_common.mako.
 */

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>
<%namespace name="common" file="../threshold_mad_common.mako"/>

<%rank:ranker_serial class_name="ranker_serial" type="float">
    <%def name="foreach(self)">
        for (int i = 0; i < (${self})->N; i++)
        {
            ${caller.body('(%s)->data[i]' % (self,))}
        }
    </%def>
    GLOBAL const float *data;
    int N;
</%rank:ranker_serial>

<%rank:ranker_parallel class_name="ranker_parallel" serial_class="ranker_serial" type="float" size="${size}">
    <%def name="thread_id(self)">
        get_local_id(0)
    </%def>
</%rank:ranker_parallel>

<%common:median_non_zero ranker_class="ranker_parallel" uniform="True" prefix="uniform_"/>
<%common:median_non_zero ranker_class="ranker_parallel" uniform="False" prefix="nonuniform_"/>

/**
 * Computes the median of the non-zero elements of an array, using two
 * variants of the algorithm. The two outputs are returned.
 *
 * Only a single workgroup is used.
 */
KERNEL void test_median_non_zero(
    GLOBAL const float * RESTRICT in,
    GLOBAL float * RESTRICT out,
    int N)
{
    LOCAL_DECL ranker_parallel_scratch scratch;
    ranker_parallel ranker;
    int lid = get_local_id(0);
    int start = N * lid / ${size};
    int end = N * (lid + 1) / ${size};
    ranker.serial.data = in + start;
    ranker.serial.N = end - start;
    ranker.scratch = &scratch;
    out[0] = uniform_median_non_zero(&ranker, N);
    out[1] = nonuniform_median_non_zero(&ranker, N);
}
