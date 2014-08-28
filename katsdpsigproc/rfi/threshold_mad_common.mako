/**
 * @file
 *
 * Code shared (via mako inclusion) between threshold_mad.mako and threshold_mad_t.mako.
 */

<%def name="median_non_zero(ranker_class)">
/**
 * Return the value which has rank @a rank within the given ranker. The
 * ranker must operate on positive 32-bit floats.
 *
 * If @a halfway is true, then it returns the average of ranks @a rank and @a
 * rank - 1.
 */
__device__ float find_rank(${ranker_class} *ranker, int rank, bool halfway)
{
    int cur = 0;
    for (int i = 30; i >= 0; i--)
    {
        int test = cur | (1 << i);
        int r = ${ranker_class}_rank(ranker, __int_as_float(test));
        if (r <= rank)
            cur = test;
    }

    float result = __int_as_float(cur);
    if (halfway)
    {
        int r = ${ranker_class}_rank(ranker, result);
        if (r == rank)
        {
            float prev = ${ranker_class}_max_below(ranker, result);
            result = (result + prev) * 0.5f;
        }
    }
    return result;
}

/**
 * Finds the median value in the ranker, ignoring zeros. The result is
 * undefined if N is zero or all values are zero. The ranker must operate
 * on positive 32-bit floats.
 */
__device__ float median_non_zero(${ranker_class} *ranker, int N)
{
    int zeros = ${ranker_class}_zeros(ranker);
    int rank2 = N + zeros;
    return find_rank(ranker, rank2 / 2, !(rank2 & 1));
}
</%def>
