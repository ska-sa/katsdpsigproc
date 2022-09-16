/*******************************************************************************
 * Copyright (c) 2014-2019, National Research Foundation (SARAO)
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

<%def name="op_plus(a, b, type)">((${a}) + (${b}))</%def>
<%def name="op_max(a, b, type)">max((${a}), (${b}))</%def>
<%def name="op_min(a, b, type)">min((${a}), (${b}))</%def>
<%def name="op_fmin(a, b, type)">
% if type.startswith('float') or type.startswith('double'):
fmin((${a}), (${b}))
% else:
min((${a}), (${b}))
% endif
</%def>
<%def name="op_fmax(a, b, type)">
% if type.startswith('float') or type.startswith('double'):
fmax((${a}), (${b}))
% else:
max((${a}), (${b}))
% endif
</%def>

## Defines the data structure for a reduction operation. See below for the
## meaning of allow_shuffle.
<%def name="define_scratch(type, size, scratch_type, allow_shuffle=False)">
<%
use_shuffle = int(allow_shuffle and ((size & (size - 1)) == 0 or size % simd_group_size == 0))
small_shuffle = int(use_shuffle and size <= simd_group_size)
%>
typedef struct ${scratch_type}
{
#if !(SHUFFLE_AVAILABLE && ${small_shuffle})
    ${type} data[${size}];
#endif

#if SHUFFLE_AVAILABLE && ${use_shuffle}
    static __device__ ${type} shfl_xor_sync(unsigned mask, ${type} value, int laneMask, int size)
    {
        union
        {
            ${type} v;
            int a[sizeof(${type}) / sizeof(int)];
        } transfer;
        transfer.v = value;
        for (int j = 0; j < sizeof(${type}) / sizeof(int); j++)
        {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
            transfer.a[j] = __shfl_xor_sync(mask, transfer.a[j], laneMask, size);
#else
            transfer.a[j] = __shfl_xor(transfer.a[j], laneMask, size);
#endif
        }
        return transfer.v;
    }
#endif
} ${scratch_type};
</%def>

/**
 * Cooperative reduction operation amongst @a size workitems. Only commutative
 * operations are currently supported. All workitems in the workgroup must
 * call this (non-divergently), but the workitems can be partitioned into
 * separate sets of @a size workitems.
 *
 * If @a allow_shuffle is True, the caller guarantees that
 * - the @a idx passed to the generated function is the linear thread ID modulo
 *   @a size.
 * In this case, the implementation @em may use Kepler shuffle instructions on
 * CUDA for communication, but it will not do so in all cases. The following
 * will prevent this fast path from being used
 * - not compiling for CUDA
 * - not compiling for Compute Capability 3.0 or higher
 * - size is not a power of 2 and is not a multiple of the warp size
 *
 * @param type         The type of items to reduce.
 * @param size         Number of cooperating work-items
 * @param function     Name of the defined function
 * @param scratch_type Type defined with @c define_scratch
 * @param op           Binary operator (defaults to addition)
 * @param rake_width   Number of work-items that perform serial up-sweep.
 *                     Defaults to the SIMD group size.
 * @param allow_shuffle See above.
 * @param broadcast    If False, only the thread with idx 0 has a defined result.
 */
<%def name="define_function(type, size, function, scratch_type, op=None, rake_width=None, allow_shuffle=False, broadcast=True)">
<%
if op is None:
    op = op_plus
if rake_width is not None and allow_shuffle:
    raise ValueError("rake_width may not currently be set when allow_shuffle is used")
if rake_width is None:
    rake_width = simd_group_size
rake_width = min(rake_width, size)
use_shuffle = int(allow_shuffle and ((size & (size - 1)) == 0 or size % simd_group_size == 0))
shuffle_mask = (1 << simd_group_size) - 1
%>

## This is horrifically confusing because it is handling 8 cases:
## - either size <= rake_width or size > rake_width
## - either the rake reduction is done using shuffles or using scratch
## - broadcast is True or False
## It's easiest to understand by picking one of the options at a time and
## following the logic for that case.
DEVICE_FN ${type} ${function}(${type} value, int idx, LOCAL ${scratch_type} *scratch)
{
% if size > rake_width:
    const int rake_width = ${rake_width};
    // Transfer values to shared memory for the raking warp to sum
    const bool first_rake = idx < rake_width;
    if (!first_rake)
        scratch->data[idx] = value;
    BARRIER();
% else:
    const bool first_rake = true;
% endif

    // Raking warp sums values from other warps
    if (first_rake)
    {
% if size > rake_width:
        const int full_chunks = ${size / rake_width};
        for (int i = 1; i < full_chunks; i++)
            value = ${op('value', 'scratch->data[idx + i * rake_width]', type)};
% if size % rake_width != 0:
        if (idx < ${size % rake_width})
            value = ${op('value', 'scratch->data[idx + full_chunks * rake_width]', type)};
% endif
% endif
        // If we're shuffling, we can do the final reduction inside the
        // first_rake test. Otherwise, it has to be outside to allow barriers
        // to be in uniform control flow.
#if SHUFFLE_AVAILABLE && ${use_shuffle}
        for (int i = ${rake_width} / 2; i >= 1; i /= 2)
        {
            ${type} other = ${scratch_type}::shfl_xor_sync(${hex(shuffle_mask)}, value, i, ${rake_width});
            value = ${op('value', 'other', type)};
        }
#else
        scratch->data[idx] = value;
#endif
    }

#if !(SHUFFLE_AVAILABLE && ${use_shuffle})
    /* Rake reduction when shuffle is not available. At this point,
     * scratch->data[0..rake_width-1] is populated with partial reductions,
     * but there has been no barrier since they were written.
     */
<% N = rake_width %>
% while N > 1:
    // N = ${N}
    BARRIER();
    if (idx < ${N // 2})
    {
        value = ${op('value', 'scratch->data[idx + %d]' % ((N + 1) // 2), type)};
        scratch->data[idx] = value;
    }
<% N = (N + 1) // 2 %>
% endwhile
#endif

    /* At this point value in idx==0 contains the final reduction. If not using
     * shuffles, then it is also stored in scratch->data[0], but a barrier is
     * required before it can be accessed by other workitems.
     *
     * If using shuffles and size <= rake_width, we are done, because the
     * shuffles make the result available in all lanes of the warp.
     */

#if !(SHUFFLE_AVAILABLE && ${use_shuffle} && ${int(size <= rake_width)})

% if broadcast:
# if SHUFFLE_AVAILABLE && ${use_shuffle}
    if (idx == 0)
        scratch->data[0] = value;
# endif
    BARRIER();
    value = scratch->data[0];
% endif  ## broadcast
    /* We have accessed scratch since the last barrier. The caller might
     * immediately re-use scratch for something else, so we need a final
     * barrier.
     */
    BARRIER();

#endif  // !(SHUFFLE_AVAILABLE && ${use_shuffle} && ${int(size <= rake_width)})

    return value;
}
</%def>
