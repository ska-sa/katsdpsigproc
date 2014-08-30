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
        r += ${v} < value;
    </%call>
    return r;
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
${wg_reduce.define_scratch(type, size, class_name + '_scratch_max')}
${wg_reduce.define_function('int', size, class_name + '_reduce_sum', class_name + '_scratch_sum')}
${wg_reduce.define_function(type, size, class_name + '_reduce_max', class_name + '_scratch_max', wg_reduce.op_max)}

typedef union ${class_name}_scratch
{
    ${class_name}_scratch_sum sum;
    ${class_name}_scratch_max max;
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

DEVICE_FN ${type} ${class_name}_max_below(${class_name} *self, ${type} limit)
{
    ${type} s = ${serial_class}_max_below(&self->serial, limit);
    return ${class_name}_reduce_max(s, (${caller.thread_id('self')}), &self->scratch->max);
}
</%def>
