/**
 * @file
 *
 * Routines for computing rank statistics.
 */

<%namespace name="wg_reduce" file="wg_reduce.cu"/>

/* Defines a class that computes ranks and other convenient statistics.
 * The caller must provide a @a foreach def that generates an iteration
 * over all relevant values. The caller body is pasted into the class.
 */
<%def name="ranker_serial(class_name, type)">
class ${class_name}
{
    ${caller.body()}

public:
    /// Count the number of zero elements
    __device__ int zeros() const
    {
        int c = 0;
        <%call expr="caller.foreach()" args="v">
            c += ${v} == 0;
        </%call>
        return c;
    }

    /**
     * Count the number of elements which strictly less than @a value.
     */
    __device__ int rank(${type} value) const
    {
        int r = 0;
        <%call expr="caller.foreach()" args="v">
            r += ${v} < value;
        </%call>
        return r;
    }

    /**
     * Return the largest value that is strictly less than @a limit.
     * Returns 0 if there isn't one.
     */
    __device__ ${type} max_below(${type} limit) const
    {
        ${type} ans = 0;
        <%call expr="caller.foreach()" args="v">
            ${type} cur = ${v};
            if (cur > ans && cur < limit)
                ans = cur;
        </%call>
        return ans;
    }
};
</%def>

<%def name="ranker_parallel(class_name, serial_class, type, size)">
/**
 * Provides the same interface as ranker_serial, but for
 * collective operation between @a size cooperating threads.
 * The caller must provide a constructor that initializes
 * @ref scratch and @ref tid.
 */
class ${class_name}
{
private:
    ${wg_reduce.define_scratch('int', size, 'ScratchSum')}
    ${wg_reduce.define_scratch(type, size, 'ScratchMax')}

public:
    union Scratch
    {
        ScratchSum sum;
        ScratchMax max;
    };

private:
    /// Underlying implementation
    ${serial_class} serial;
    /// Shared memory scratch space for reducing counts
    Scratch *scratch;
    /// Thread ID (must range from 0 to @a size - 1)
    int tid;

    ${wg_reduce.define_function('int', size, 'reduce_sum', 'ScratchSum')}
    ${wg_reduce.define_function(type, size, 'reduce_max', 'ScratchMax', wg_reduce.op_max)}

    ${caller.body()}

public:
    __device__ int zeros()
    {
        int c = serial.zeros();
        return reduce_sum(c, tid, &scratch->sum);
    }

    __device__ int rank(int value)
    {
        int r = serial.rank(value);
        return reduce_sum(r, tid, &scratch->sum);
    }

    __device__ ${type} max_below(int limit)
    {
        ${type} s = serial.max_below(limit);
        return reduce_max(s, tid, &scratch->max);
    }
};
</%def>
