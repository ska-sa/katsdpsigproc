/**
 * @file
 *
 * Wrapper around rank-finding functions, to test them from the host.
 */

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>
#define VT ${vt}

<%def name="define_rankers(type)">


<%rank:ranker_serial_store class_name="ranker_serial_${type}" type="${type}" size="VT"/>

DEVICE_FN void ranker_serial_${type}_init(
    ranker_serial_${type} *self,
    GLOBAL const ${type} *data, int start, int step, int N)
{
    int p = start;
    for (int i = 0; i < VT; i++)
    {
        self->values[i] = (p < N) ? data[p] : NAN;
        p += step;
    }
}

<%rank:ranker_parallel class_name="ranker_parallel_${type}" serial_class="ranker_serial_${type}" type="${type}" size="${size}">
    <%def name="thread_id(self)">
        get_local_id(0)
    </%def>
</%rank:ranker_parallel>

DEVICE_FN void ranker_parallel_${type}_init(
    ranker_parallel_${type} *self,
    GLOBAL const ${type} * RESTRICT data, int start, int step, int N,
    LOCAL ranker_parallel_${type}_scratch *scratch)
{
    ranker_serial_${type}_init(&self->serial, data, start, step, N);
    self->scratch = scratch;
}
</%def>

<%self:define_rankers type="float"/>

<%rank:find_min_float ranker_class="ranker_parallel_float"/>
<%rank:find_max_float ranker_class="ranker_parallel_float"/>

<%rank:find_rank_float ranker_class="ranker_parallel_float" uniform="True"/>

/**
 * Computes the [0,100,25,75,50] percentiles of the array.
 *
 * Only a single workgroup is used.
 */
KERNEL void percentile5_float(
    GLOBAL const float * RESTRICT in,
    GLOBAL float * RESTRICT out, int in_stride, int out_stride,
    int N)
{
    LOCAL_DECL ranker_parallel_float_scratch scratch;
    ranker_parallel_float ranker;
    int lid = get_local_id(0);//thread id within processing element
    int row = get_global_id(1);//block id of processing element 
    ranker_parallel_float_init(&ranker, in + row * in_stride, get_local_id(0), ${size}, N, &scratch);
        
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
