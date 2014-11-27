/**
 * @file
 *
 * Routines for computing rank statistics.
 */

<%namespace name="wg_reduce" file="wg_reduce.mako"/>

/* Defines a class that computes ranks and other convenient statistics.
 * The caller must provide a @a foreach def that generates an iteration
 * over all relevant values. The caller body is pasted into the class.
 */
<%def name="ranker_serial(class_name, type)">
typedef struct ${class_name}
{
    ${caller.body()}
} ${class_name};

/// Count the number of zero elements
DEVICE_FN int ${class_name}_zeros(${class_name} *self)
{
    int c = 0;
    <%call expr="caller.foreach('self')" args="v">
        c += ${v} == 0;
    </%call>
    return c;
}

/**
 * Count the number of elements which strictly less than @a value.
 */
DEVICE_FN int ${class_name}_rank(${class_name} *self, ${type} value)
{
    int r = 0;
    <%call expr="caller.foreach('self')" args="v">
        r += (${v}) < value;
    </%call>
    return r;
}

/**
 * Return the smallest value. The identity value must be at least as big as
 * any array element. The result is undefined if there are larger elements or
 * if there are any NaN values.
 */
DEVICE_FN ${type} ${class_name}_min(${class_name} *self, ${type} identity)
{
    ${type} ans = identity;
    <%call expr="caller.foreach('self')" args="v">
        ans = min(ans, ${v});
    </%call>
    return ans;
}

/**
 * Return the smallest value. The identity value must be at least as small as
 * any array element. The result is undefined if there are smaller elements or
 * if there are any NaN values.
 */
DEVICE_FN ${type} ${class_name}_max(${class_name} *self, ${type} identity)
{
    ${type} ans = identity;
    <%call expr="caller.foreach('self')" args="v">
        ans = max(ans, ${v});
    </%call>
    return ans;
}

/**
 * Return the largest value that is strictly less than @a limit.
 * Returns 0 if there isn't one.
 */
DEVICE_FN ${type} ${class_name}_max_below(${class_name} *self, ${type} limit)
{
    ${type} ans = 0;
    <%call expr="caller.foreach('self')" args="v">
        ${type} cur = ${v};
        if (cur > ans && cur < limit)
            ans = cur;
    </%call>
    return ans;
}
</%def>

<%def name="ranker_parallel(class_name, serial_class, type, size)">
/**
 * Provides the same interface as ranker_serial, but for
 * collective operation between @a size cooperating threads.
 * The caller must provide a constructor that initializes
 * @ref scratch, and it must provide a def called @a thread_id
 * that must range from 0 to @a size - 1.
 */
${wg_reduce.define_scratch('int', size, class_name + '_scratch_sum')}
${wg_reduce.define_scratch(type, size, class_name + '_scratch_minmax')}
${wg_reduce.define_function('int', size, class_name + '_reduce_sum', class_name + '_scratch_sum')}
${wg_reduce.define_function(type, size, class_name + '_reduce_min', class_name + '_scratch_minmax', wg_reduce.op_min)}
${wg_reduce.define_function(type, size, class_name + '_reduce_max', class_name + '_scratch_minmax', wg_reduce.op_max)}

typedef union ${class_name}_scratch
{
    ${class_name}_scratch_sum sum;
    ${class_name}_scratch_minmax minmax;
} ${class_name}_scratch;

typedef struct ${class_name}
{
    /// Underlying implementation
    ${serial_class} serial;
    /// Shared memory scratch space for reducing counts
    LOCAL ${class_name}_scratch *scratch;

    ${caller.body()}
} ${class_name};

DEVICE_FN int ${class_name}_zeros(${class_name} *self)
{
    int c = ${serial_class}_zeros(&self->serial);
    return ${class_name}_reduce_sum(c, (${caller.thread_id('self')}), &self->scratch->sum);
}

DEVICE_FN int ${class_name}_rank(${class_name} *self, ${type} value)
{
    int r = ${serial_class}_rank(&self->serial, value);
    return ${class_name}_reduce_sum(r, (${caller.thread_id('self')}), &self->scratch->sum);
}

DEVICE_FN ${type} ${class_name}_min(${class_name} *self, ${type} identity)
{
    ${type} s = ${serial_class}_min(&self->serial, identity);
    return ${class_name}_reduce_min(s, (${caller.thread_id('self')}), &self->scratch->minmax);
}

DEVICE_FN ${type} ${class_name}_max(${class_name} *self, ${type} identity)
{
    ${type} s = ${serial_class}_max(&self->serial, identity);
    return ${class_name}_reduce_max(s, (${caller.thread_id('self')}), &self->scratch->minmax);
}

DEVICE_FN ${type} ${class_name}_max_below(${class_name} *self, ${type} limit)
{
    ${type} s = ${serial_class}_max_below(&self->serial, limit);
    return ${class_name}_reduce_max(s, (${caller.thread_id('self')}), &self->scratch->minmax);
}
</%def>

<%def name="find_rank_float(ranker_class, uniform, prefix='')">
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
DEVICE_FN float ${prefix}find_rank_float(${ranker_class} *ranker, int rank, bool halfway)
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
</%def>

<%def name="find_min_float(ranker_class, prefix='')">
/**
 * Return the smallest value within the given ranker. The ranker must
 * operate on finite floats (results are undefined on non-finite values).
 */
DEVICE_FN float ${prefix}find_min_float(${ranker_class} *ranker)
{
    return ${ranker_class}_min(ranker, FLT_MAX);
}
</%def>

<%def name="find_max_float(ranker_class, prefix='')">
/**
 * Return the largest value within the given ranker. The ranker must
 * operate on finite floats (results are undefined on non-finite values).
 */
DEVICE_FN float ${prefix}find_max_float(${ranker_class} *ranker)
{
    return ${ranker_class}_max(ranker, -FLT_MAX);
}
</%def>

<%def name="median_non_zero_float(ranker_class, uniform, prefix='')">

<%self:find_rank_float ranker_class="${ranker_class}" uniform="${uniform}" prefix="${prefix}"/>
/**
 * Finds the median value in the ranker, ignoring zeros. The result is
 * undefined if N is zero or all values are zero. The ranker must operate
 * on positive 32-bit floats.
 */
DEVICE_FN float ${prefix}median_non_zero_float(${ranker_class} *ranker, int N)
{
    int zeros = ${ranker_class}_zeros(ranker);
    int rank2 = N + zeros;
    return ${prefix}find_rank_float(ranker, rank2 / 2, !(rank2 & 1));
}
</%def>
