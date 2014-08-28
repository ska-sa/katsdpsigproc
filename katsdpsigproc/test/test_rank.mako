/**
 * @file
 *
 * Wrapper around rank-finding functions, to test them from the host.
 */

<%include file="/port.mako"/>
<%namespace name="rank" file="/rank.mako"/>

<%rank:ranker_serial class_name="ranker_serial" type="int">
    <%def name="foreach(self)">
        for (int i = 0; i < ${self}->N; i++)
        {
            ${caller.body(self + '->values[i]')}
        }
    </%def>
    GLOBAL const int *values;
    int N;
</%rank:ranker_serial>

<%rank:ranker_parallel class_name="ranker_parallel" serial_class="ranker_serial" type="int" size="${size}"/>

/**
 * Measure the rank of values 0 to @a M - 1 against a given array.
 * It uses only a single work-item.
 */
KERNEL REQD_WORK_GROUP_SIZE(1, 1, 1) void test_rank_serial(
    GLOBAL const int *in,
    GLOBAL int *out,
    int N, int M)
{
    ranker_serial ranker;
    ranker.values = in;
    ranker.N = N;
    for (int i = 0; i < M; i++)
        out[i] = ranker_serial_rank(&ranker, i);
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
    LOCAL_DECL ranker_parallel_scratch scratch;

    int id = get_local_id(0);
    int start = id * N / ${size};
    int end = (id + 1) * N / ${size};
    ranker_parallel ranker;
    ranker.serial.values = in + start;
    ranker.serial.N = end - start;
    ranker.tid = id;
    ranker.scratch = &scratch;
    for (int i = 0; i < M; i++)
        out[i] = ranker_parallel_rank(&ranker, i);
}
