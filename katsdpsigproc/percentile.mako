/**
 * @file
 *
 * Wrapper around rank-finding functions, to test them from the host.
 */

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>

<%def name="define_rankers(type)">
<%rank:ranker_serial class_name="ranker_serial_${type}" type="${type}">
    <%def name="foreach(self)">
        for (int i = 0; i < ${self}->N; i++)
        {
            ${caller.body(self + '->values[i]')}
        }
    </%def>
    GLOBAL const ${type} *values;
    int N;
</%rank:ranker_serial>

<%rank:ranker_parallel class_name="ranker_parallel_${type}" serial_class="ranker_serial_${type}" type="${type}" size="${size}">
    <%def name="thread_id(self)">
        get_local_id(0)
    </%def>
</%rank:ranker_parallel>
</%def>

<%self:define_rankers type="float"/>

<%rank:find_min_float ranker_class="ranker_parallel_float"/>
<%rank:find_max_float ranker_class="ranker_parallel_float"/>

<%rank:find_rank_float ranker_class="ranker_parallel_float" uniform="True"/>

/**
 * Computes the percentile an array.
 * Only a single workgroup is used.
 */
KERNEL void test_percentile_float(
    GLOBAL const float * RESTRICT in,
    GLOBAL float * RESTRICT out,
    int N,int pN)
{
    LOCAL_DECL ranker_parallel_float_scratch scratch;
    ranker_parallel_float ranker;
    int lid = get_local_id(0);
    int start = N * lid / ${size};
    int end = N * (lid + 1) / ${size};
    ranker.serial.values = in + start;
    ranker.serial.N = end - start;
    ranker.scratch = &scratch;
    *out = find_rank_float(&ranker, pN, false);
}

/**
 * Computes the [0,100,25,75,50] percentiles of the array.
 *
 * Only a single workgroup is used.
 */
KERNEL void test_percentile5_float(
    GLOBAL const float * RESTRICT in,
    GLOBAL float * RESTRICT out,
    int N)
{
    LOCAL_DECL ranker_parallel_float_scratch scratch;
    ranker_parallel_float ranker;
    int lid = get_local_id(0);
    int start = N * lid / ${size};
    int end = N * (lid + 1) / ${size};
    ranker.serial.values = in + start;
    ranker.serial.N = end - start;
    ranker.scratch = &scratch;
    out[0] = find_min_float(&ranker);
    out[1] = find_max_float(&ranker);
    out[2] = find_rank_float(&ranker, (N-1)/4, false);
    out[3] = find_rank_float(&ranker, ((N-1)*3)/4, false);
    out[4] = find_rank_float(&ranker, (N-1)/2, false);    
}
