/*******************************************************************************
 * Copyright (c) 2014-2020, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/**
 * @file
 *
 * Routines for computing rank statistics.
 */

<%namespace name="wg_reduce" file="wg_reduce.mako"/>

/* Defines a class that computes ranks and other convenient statistics.
 * The caller must provide a @a foreach def that generates an iteration
 * over all relevant values, and takes optional @a start and @a stop
 * arguments to control the range of iteration.
 * The caller body is pasted into the class.
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
 * Count the number of elements which are strictly less than @a value
 * (and are not NaN). The result is undefined if @a value is NaN.
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
 * Return the smallest non-NaN value, or NaN if there isn't one.
 */
DEVICE_FN ${type} ${class_name}_fmin(${class_name} *self)
{
    ${type} ans;
    <%call expr="caller.foreach('self', start=0, stop=1)" args="v">
        ans = ${v};
    </%call>
    <%call expr="caller.foreach('self', start=1)" args="v">
        ans = ${wg_reduce.op_fmin('ans', v, type)};
    </%call>
    return ans;
}

/**
 * Return the largest non-NaN value, or NaN if there isn't one.
 */
DEVICE_FN ${type} ${class_name}_fmax(${class_name} *self)
{
    ${type} ans;
    <%call expr="caller.foreach('self', start=0, stop=1)" args="v">
        ans = ${v};
    </%call>
    <%call expr="caller.foreach('self', start=1)" args="v">
        ans = ${wg_reduce.op_fmax('ans', v, type)};
    </%call>
    return ans;
}

/**
 * Return the largest non-NaN value that is strictly less than @a limit.
 * Returns 0 if there isn't one. The result is undefined if @a limit is
 * NaN.
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

/* Specialization of ranker_serial that provides the foreach function and
 * struct body. It stores values directly inside the object (which the
 * user must still populate). The number of values is fixed at compile time,
 * so smaller arrays must be handled by padding with NaNs.
 */
<%def name="ranker_serial_store(class_name, type, size)">
    <%self:ranker_serial class_name="${class_name}" type="${type}">
        <%def name="foreach(self, start=0, stop=size)">
            #pragma unroll
            for (int i = ${start}; i < ${stop}; i++)
            {
                ${caller.body('(%s)->values[i]' % (self,))}
            }
        </%def>
        float values[${size}];
    </%self:ranker_serial>
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
${wg_reduce.define_function(type, size, class_name + '_reduce_fmin', class_name + '_scratch_minmax', wg_reduce.op_fmin)}
${wg_reduce.define_function(type, size, class_name + '_reduce_fmax', class_name + '_scratch_minmax', wg_reduce.op_fmax)}

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

DEVICE_FN ${type} ${class_name}_fmin(${class_name} *self)
{
    ${type} s = ${serial_class}_fmin(&self->serial);
    return ${class_name}_reduce_fmin(s, (${caller.thread_id('self')}), &self->scratch->minmax);
}

DEVICE_FN ${type} ${class_name}_fmax(${class_name} *self)
{
    ${type} s = ${serial_class}_fmax(&self->serial);
    return ${class_name}_reduce_fmax(s, (${caller.thread_id('self')}), &self->scratch->minmax);
}

DEVICE_FN ${type} ${class_name}_max_below(${class_name} *self, ${type} limit)
{
    ${type} s = ${serial_class}_max_below(&self->serial, limit);
    return ${class_name}_reduce_fmax(s, (${caller.thread_id('self')}), &self->scratch->minmax);
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
 * If @a uniform (template parameter) is true, then the entire workgroup must
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
 * Return the smallest non-NaN value within the given ranker.
 */
DEVICE_FN float ${prefix}find_min_float(${ranker_class} *ranker)
{
    return ${ranker_class}_fmin(ranker);
}
</%def>

<%def name="find_max_float(ranker_class, prefix='')">
/**
 * Return the largest non-NaN value within the given ranker.
 */
DEVICE_FN float ${prefix}find_max_float(${ranker_class} *ranker)
{
    return ${ranker_class}_fmax(ranker);
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
