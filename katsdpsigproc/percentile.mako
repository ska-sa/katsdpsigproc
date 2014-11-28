/**
 * @file
 *
 * Wrapper around rank-finding functions, to test them from the host.
 */

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>

<%def name="define_rankers(type)">
<%rank:ranker_serial class_name="ranker_serial_${type}" type="${type}">
    <%def name="foreach(self, start=0, stop=None)">
        <% if stop is None: stop = '({})->N'.format(self) %>
        for (int i = ${start}; i < ${stop}; i++)
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
    GLOBAL float * RESTRICT out, int in_stride, int out_stride,
    int N)
{
    LOCAL_DECL ranker_parallel_float_scratch scratch;
    ranker_parallel_float ranker;
    int lid = get_local_id(0);//thread id within processing element
    int row = get_global_id(1);//block id of processing element 
    int start = N * lid / ${size};
    int end = N * (lid + 1) / ${size};
    ranker.serial.values = in + start+row * in_stride;
    ranker.serial.N = end - start;
    ranker.scratch = &scratch;
    float perc0 = find_min_float(&ranker);
    float perc1 = find_max_float(&ranker);
    float perc2 = find_rank_float(&ranker, (N-1)/4, false);
    float perc3 = find_rank_float(&ranker, ((N-1)*3)/4, false);
    float perc4 = find_rank_float(&ranker, (N-1)/2, false);
    
    if (lid == 0)
    {
        out[row+0*out_stride] = perc0;
        out[row+1*out_stride] = perc1;
        out[row+2*out_stride] = perc2;
        out[row+3*out_stride] = perc3;
        out[row+4*out_stride] = perc4;
    }
}
