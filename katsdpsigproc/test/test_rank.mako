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

<%self:define_rankers type="int"/>
<%self:define_rankers type="float"/>

/**
 * Measure the rank of values 0 to @a M - 1 against a given array.
 * It uses only a single work-item.
 */
KERNEL REQD_WORK_GROUP_SIZE(1, 1, 1) void test_rank_serial(
    GLOBAL const int *in,
    GLOBAL int *out,
    int N, int M)
{
    ranker_serial_int ranker;
    ranker.values = in;
    ranker.N = N;
    for (int i = 0; i < M; i++)
        out[i] = ranker_serial_int_rank(&ranker, i);
}

/**
 * Measure the rank of values 0 to @a M - 1 against a given array.
 * It uses only a single work-group.
 */
KERNEL REQD_WORK_GROUP_SIZE(${size}, 1, 1) void test_rank_parallel(
    GLOBAL const int *in,
    GLOBAL int *out,
    int N, int M)
{
    LOCAL_DECL ranker_parallel_int_scratch scratch;

    int lid = get_local_id(0);
    int start = lid * N / ${size};
    int end = (lid + 1) * N / ${size};
    ranker_parallel_int ranker;
    ranker.serial.values = in + start;
    ranker.serial.N = end - start;
    ranker.scratch = &scratch;
    for (int i = 0; i < M; i++)
        out[i] = ranker_parallel_int_rank(&ranker, i);
}

<%rank:median_non_zero_float ranker_class="ranker_parallel_float" uniform="True" prefix="uniform_"/>
<%rank:median_non_zero_float ranker_class="ranker_parallel_float" uniform="False" prefix="nonuniform_"/>

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
    LOCAL_DECL ranker_parallel_float_scratch scratch;
    ranker_parallel_float ranker;
    int lid = get_local_id(0);
    int start = N * lid / ${size};
    int end = N * (lid + 1) / ${size};
    ranker.serial.values = in + start;
    ranker.serial.N = end - start;
    ranker.scratch = &scratch;
    out[0] = uniform_median_non_zero_float(&ranker, N);
    out[1] = nonuniform_median_non_zero_float(&ranker, N);
}
