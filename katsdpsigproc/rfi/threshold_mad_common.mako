/**
 * @file
 *
 * Code shared (via mako inclusion) between threshold_mad.mako and threshold_mad_t.mako.
 */

<%def name="median_non_zero(ranker_class, uniform, prefix='')">
/**
 * Return the value which has rank @a rank within the given ranker. The
 * ranker must operate on positive 32-bit floats.
 *
 * If @a halfway is true, then it returns the average of ranks @a rank and @a
 * rank - 1.
 *
 * If @a uniform (template parameter is true), then the entire workgroup must
 * be operating on the same set of data and the same parameters.
 */
DEVICE_FN float ${prefix}find_rank(${ranker_class} *ranker, int rank, bool halfway)
{
    int cur = 0;
    for (int i = 30; i >= 0; i--)
    {
        int test = cur | (1 << i);
        int r = ${ranker_class}_rank(ranker, as_float(test));
        if (r <= rank)
            cur = test;
    }

    float result = as_float(cur);
% if uniform:
    if (halfway)
    {
        int r = ${ranker_class}_rank(ranker, result);
        if (r == rank)
        {
            float prev = ${ranker_class}_max_below(ranker, result);
            result = (result + prev) * 0.5f;
        }
    }
% else:
    // These computations use barriers, so they cannot be
    // put inside the conditional
    int r = ${ranker_class}_rank(ranker, result);
    float prev = ${ranker_class}_max_below(ranker, result);
    if (halfway && r == rank)
    {
        result = (result + prev) * 0.5f;
    }
% endif
    return result;
}

/**
 * Finds the median value in the ranker, ignoring zeros. The result is
 * undefined if N is zero or all values are zero. The ranker must operate
 * on positive 32-bit floats.
 */
DEVICE_FN float ${prefix}median_non_zero(${ranker_class} *ranker, int N)
{
    int zeros = ${ranker_class}_zeros(ranker);
    int rank2 = N + zeros;
    return ${prefix}find_rank(ranker, rank2 / 2, !(rank2 & 1));
}
</%def>
