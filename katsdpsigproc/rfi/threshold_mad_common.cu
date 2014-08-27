/**
 * @file
 *
 * Code shared (via mako inclusion) between threshold_mad.cu and threshold_mad_t.cu.
 */

/**
 * Reinterpret a float as integer, and take absolute value.
 */
__device__ static int abs_int(float x)
{
    return __float_as_int(x) & 0x7FFFFFFF;
}

/**
 * Return the element of @a in whose absolute values has rank @a rank.
 *
 * If @a halfway is true, then it returns the average of ranks @a rank and @a
 * rank - 1.
 */
template<typename Ranker>
__device__ float find_rank_abs(Ranker &ranker, int rank, bool halfway)
{
    int cur = 0;
    for (int i = 30; i >= 0; i--)
    {
        int test = cur | (1 << i);
        int r = ranker.rank(__int_as_float(test));
        if (r <= rank)
            cur = test;
    }

    float result = __int_as_float(cur);
    if (halfway)
    {
        int r = ranker.rank(result);
        if (r == rank)
        {
            float prev = ranker.max_below(result);
            result = (result + prev) * 0.5f;
        }
    }
    return result;
}

/**
 * Finds the median absolute value in @a in, ignoring zeros.
 */
template<typename Ranker>
__device__ float median_abs_impl(Ranker &ranker, int N)
{
    int zeros = ranker.zeros();
    int rank2 = N + zeros;
    return find_rank_abs(ranker, rank2 / 2, !(rank2 & 1));
}
