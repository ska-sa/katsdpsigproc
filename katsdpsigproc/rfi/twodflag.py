"""Library to contain 2d RFI flagging routines and other RFI related functions."""

from __future__ import division, print_function, absolute_import
import math
import multiprocessing.pool

import numpy as np
import numba
from scipy.ndimage import convolve1d
import six


def running_mean(x, N, axis=None):
    """Fast implementation of a running mean (array `x` with width `N`)
    Stolen from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    and modified to allow axis selection.

    Parameters
    ----------
    x : array, float
        Input array to average
    N : int
        Size of averaging window
    axis : int
        Axis along which to apply

    Returns
    -------
    result: array, float32
        Array with averaging window applied.
    """
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis, dtype=np.float64)
    result = np.apply_along_axis(lambda x: (x[N:] - x[:-N]) / N, axis, cumsum) if axis \
        else (cumsum[N:] - cumsum[:-N]) / N
    return result.astype(x.dtype)


def linearly_interpolate_nans(y):
    """Linearly interpolate across NaNs in y, extrapolate using numpy defaults
    (ie. use constant). If all input data are NaNs, return 0's for all y

    Parameters
    ----------
    y : array, 1d

    Returns
    -------
        array with NaNs removed.

    """

    nan_locs = np.isnan(y)

    if np.all(nan_locs):
        y[:] = 0.
    else:
        X = np.nonzero(~nan_locs)[0]
        Y = y[X]
        y[nan_locs] = np.interp(np.nonzero(nan_locs), X, Y)[0]
    return y


@numba.guvectorize(['f4[:], int_, int_, f4[:]'], '(n),(),()->(n)', nopython=True)
def _box_gaussian_filter1d(data, r, passes, out):
    """Internals of :func:`box_gaussian_filter1d`.

    This is a gufunc that is run along a single row.
    """
    K = passes
    d = 2 * r + 1
    # Pad on left with zeros
    padded = np.empty(data.size + 2 * r * K, data.dtype)
    padded[:2 * r * K] = 0
    padded[2 * r * K:] = data
    for p in range(K):
        # On each pass, padded[i] is replaced by the sum of padded[i : i + d]
        # from the previous pass. The accumulator is kept in double precision
        # to avoid excessive accumulation of errors.
        s = np.float64(0)
        for i in range(0, padded.size - 2 * r):
            s += padded[i + 2 * r]
            prev = padded[i]
            padded[i] = s
            s -= prev
        for i in range(padded.size - 2 * r, padded.size):
            prev = padded[i]
            padded[i] = s
            s -= prev
    out[:] = padded[r * K : -(r * K)] / data.dtype.type(d)**K


def box_gaussian_filter1d(data, sigma, axis=-1, passes=4):
    """Filter `data` with an approximate Gaussian filter, along an axis.

    Refer to :func:`box_gaussian_filter` for details.

    Parameters
    ----------
    data : ndarray
        Input data to filter
    sigma : float
        Standard deviation of the Gaussian filter
    passes : int
        Number of boxcar filters to apply

    Returns
    -------
    ndarray
        Output data, with the same shape as the input
    """
    # Move relevant axis to the end
    data = np.moveaxis(data, axis, -1)
    r = int(0.5 * math.sqrt(12.0 * sigma**2 / passes + 1))
    data = _box_gaussian_filter1d(data, r, passes)
    # Undo the axis reordering
    return np.moveaxis(data, -1, axis)


def box_gaussian_filter(data, sigma, passes=4):
    """Filter `data` with an approximate Gaussian filter.

    The filter is based on repeated filtering with a boxcar function. See
    [Get13]_ for details. It has finite support. Values outside the boundary
    are taken as zero.

    This function is not suitable when the input contains non-finite values,
    or very large variations in magnitude, as it internally computes a rolling
    sum. It also quantizes the requested sigma.

    .. [Get13] Pascal Getreuer, A Survey of Gaussian Convolution Algorithms,
       Image Processing On Line, 3 (2013), pp. 286-310.

    Parameters
    ----------
    data : ndarray
        Input data to filter
    sigma : float or sequence of floats
        Standard deviation of the Gaussian filter, per axis
    passes : int
        Number of boxcar filters to apply

    Returns
    -------
    ndarray
        Output data, with the same shape as the input
    """
    if hasattr(sigma, '__iter__') and not isinstance(sigma, six.string_types):
        sigma = list(sigma)
    else:
        sigma = [sigma] * data.ndim
    if len(sigma) != data.ndim:
        raise ValueError('sigma has wrong number of elements')
    for axis, s in enumerate(sigma):
        if s > 0:
            data = box_gaussian_filter1d(data, s, axis, passes)
    return data


def weighted_gaussian_filter(data, weight, sigma, passes=4):
    """Filter an image using an approximate Gaussian filter, where there are
    additionally weights associated with each input sample. Values outside
    of `data` have zero weight.

    See :func:`box_gaussian_filter` for a number of caveats. The result may
    contain non-finite values where the finite support of the Gaussian
    approximation contains no values with non-zero weight. However, this should
    not be depended on (particularly if arbitrary, rather than zero/one weights
    are used) due to numeric instabilities.

    Parameters
    ----------
    data : ndarray
        Input data to filter
    weight : ndarray, real
        Non-negative weights associated with the corresponding elements of
        `data`
    sigma : float or sequence of floats
        Standard deviation of the Gaussian filter, per axis
    passes : int
        Number of boxcar filters to apply

    Returns
    -------
    ndarray
        Output data, with the same shape as the input
    """
    if data.shape != weight.shape:
        raise ValueError('shape mismatch')
    filtered_weight = box_gaussian_filter(weight, sigma, passes)
    filtered = box_gaussian_filter(data * weight, sigma, passes)
    with np.errstate(invalid='ignore'):
        out = filtered / filtered_weight
    return out


def getbackground_2d(data, in_flags=None, iterations=1, spike_width=(10, 10), reject_threshold=2.0):
    """Determine a smooth background over a 2d data array by
    iteratively convolving the data with elliptical Gaussians with linearly
    decreasing width from `iterations`*`spike_width` down to `spike width`. Outliers
    greater than `reject_threshold`*sigma from the background are masked on each
    iteration.
    Initial weights are set to zero at positions specified in `in_flags` if given.
    After the final iteration a final Gaussian smoothed background is computed
    and any stray NaNs in the background are interpolated in frequency (axis 1)
    for each timestamp (axis 0). The NaNs can appear when the the convolving
    Gaussian is completely covering masked data as the sum of convolved weights
    will be zero.

    Parameters
    ----------
    data : 2D array, float
        The input data array to be smoothed
    in_flags : 2D array, boolean (same shape as `data`)
        The positions in data to have zero weight in initial iteration.
    iterations : int
        The number of iterations of Gaussian smoothing
    spike_width : sequence, float
        The 1 sigma pixel widths of the smoothing gaussian (corresponding
        to the axes of `data`)
    reject_threshold : float
        Multiple of sigma by which to reject outliers on each iteration

    Returns
    -------
    background : 2D array, float
        The smooth background.
    """

    # Make mask array
    mask = np.ones(data.shape, dtype=np.float32)
    # Mask input flags if provided
    if in_flags is not None:
        in_flags = in_flags.astype(np.bool_, copy=False)
        mask[in_flags] = 0.0
    # Convolve with Gaussians of decreasing 1sigma width from iterations*spike_width to spike_width
    for extend_factor in range(iterations, 0, -1):
        sigma = extend_factor*np.array(spike_width, dtype=np.float32)
        # Smooth background
        background = weighted_gaussian_filter(data, mask, sigma)
        residual = data-background
        # Reject outliers using MAD
        abs_residual = np.abs(residual)
        nz_residuals = abs_residual[mask > 0.0]
        if len(nz_residuals) > 0:
            sigma = 1.4826 * np.median(nz_residuals)
            # Some elements of abs_residual are NaN, but these are already masked
            with np.errstate(invalid='ignore'):
                mask = np.where(abs_residual > reject_threshold * sigma, 0.0, mask)
    # Compute final background
    background = weighted_gaussian_filter(data, mask, spike_width)
    # Remove NaNs via linear interpolation
    background = np.apply_along_axis(linearly_interpolate_nans, 1, background)

    return background


def get_baseline_flags(flagger, data, flags):
    """Run flagging method for a single baseline. This is used by multiprocessing
    to avoid problems pickling an object method. It can also be used to run the flagger
    independently of multiprocessing.

    Parameters
    ----------
    flagger : :class:`SumThresholdFlagger`
        A sumthreshold_flagger object
    data : 2D array, float
        data to flag
    flags : 2D array, boolean
        prior flags to ignore

    Returns
    -------
    a 2D array, boolean
        derived flags
    """
    return flagger.get_baseline_flags(data, flags)


@numba.jit(nopython=True, nogil=True)
def _convolve_flags(in_values, scale, threshold, out_flags, window):
    cum_size = in_values.shape[0] + 2 * window - 1
    cum = np.empty(cum_size, np.uint32)
    cum[:window] = 0
    for i in range(in_values.shape[0]):
        cum[window + i] = cum[window + i - 1] + (in_values[i] * scale > threshold)
    # numba doesn't seem to fully support negative indices, hence
    # the addition of cum_size.
    cum[cum_size - (window - 1):] = cum[cum_size - window]
    for i in range(out_flags.shape[0]):
        out_flags[i] |= (cum[i + window] - cum[i] != 0)


@numba.guvectorize(
    ['f4[:], b1[:], i8[:], f4, f4, b1[:]'],
    '(n),(n),(m),(),()->(n)', nopython=True)
def _sumthreshold_1d(input_data, flags, windows, outlier_nsigma, rho,
                     output_flags):
    abs_input = np.empty(input_data.shape, np.float32)
    nvalid = np.int64(0)
    for i in range(input_data.shape[0]):
        if not flags[i] and not np.isnan(input_data[i]):
            abs_input[nvalid] = np.abs(input_data[i])
            nvalid += 1
    # Get standard deviations using MAD
    if nvalid == 0:
        estm_stdev = np.inf
    else:
        estm_stdev = 1.4826 * np.median(abs_input[:nvalid])
    # Set up initial threshold
    threshold = outlier_nsigma * estm_stdev
    output_flags_pos = np.zeros(flags.shape, np.bool_)
    output_flags_neg = np.zeros(flags.shape, np.bool_)
    avgarray_full = np.empty(input_data.shape, np.float32)
    for window in windows:
        # Stop if the window is too large
        if window > input_data.shape[0]:
            break
        # The threshold for this iteration is calculated from the initial threshold
        # using the equation from Offringa (2010).
        tf = pow(rho, np.log2(window))
        # Get the thresholds for each element of the desired axis,
        # with an extra exis for broadcasting.
        thisthreshold = threshold / tf
        # Set already flagged values to be the +/- value of the
        # threshold if they are outside the threshold.
        cum_data = np.empty(input_data.shape[0] + 1, np.float64)
        cum_data[0] = 0
        for i in range(input_data.shape[0]):
            clamped = input_data[i]
            if output_flags_pos[i] and clamped > thisthreshold:
                clamped = thisthreshold
            elif output_flags_neg[i] and clamped < -thisthreshold:
                clamped = -thisthreshold
            cum_data[i + 1] = cum_data[i] + clamped
        # Calculate a rolling sum array from the data with the window for this iteration,
        # which is later scaled by rolliing_scale to give the rolling average.
        avgarray = cum_data[window:] - cum_data[:-window]
        rolling_scale = np.float32(1.0 / window)

        # Work out the flags from the average data above the current threshold,
        # convolve them, and OR with current flags.
        _convolve_flags(avgarray, rolling_scale, thisthreshold, output_flags_pos, window)

        # Work out the flags from the average data below the current threshold,
        # convolve them, and OR with current flags.
        _convolve_flags(avgarray, -rolling_scale, thisthreshold, output_flags_neg, window)

    output_flags[:] = output_flags_pos | output_flags_neg


class SumThresholdFlagger(object):
    """Flagger that uses the SumThreshold method (Offringa, A., MNRAS, 405, 155-167, 2010)
    to detect spikes in both frequency and time axes.
    The full algorithm does the following:

        1. Average the data in the frequency dimension (axis 1) into bins of
           size `self.average_freq`
        2. Divide the data into overlapping sub-chunks in frequency which are
           backgrounded and thresholded independently
        3. Flag a 1d spectrum median filtered in time to get fainter contaminated
           channels.
        4. Derive a smooth 2d background through each chunk
        5. SumThreshold the background subtracted chunks in time and frequency
        6. Extend derived flags in time and frequency, via self.freq_extend and
           self.time_extend
        7. Extend flags to all times and frequencies in cases when more than
           a given fraction of samples are flagged (via `self.flag_all_time_frac` and
           `self.flag_all_freq_frac`)

    Parameters
    ----------

    outlier_nsigma : float
        Number of sigma to reject outliers when thresholding
    windows_time : array, int
        Size of averaging windows to use in the SumThreshold method in time
    windows_freq : array, int
        Size of averaging windows to use in the SumThreshold method in frequency
    background_reject : float
        Number of sigma to reject outliers when backgrounding
    background_iterations : int
        Number of iterations to use when determining a smooth background, after each
        iteration data in excess of `background_reject`*`sigma` are masked
    spike_width_time : float
        Characteristic width in dumps to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 0.
    spike_width_freq : float
        Characteristic width in channels to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 1.
    time_extend : int
        Size of kernel in time to convolve with flags after detection
    freq_extend : int
        Size of kernel in frequency to convolve with flags after detection
    freq_chunks : int
        Number of equal-sized chunks to independently flag in frequency. Smaller
        chunks will be less affected by variations in the band in the frequency domain.
    average_freq : int
        Number of channels to average frequency before flagging. Flags will be extended
        to the frequency shape of the input data before being returned
    flag_all_time_frac : float
        Fraction of data flagged above which to extend flags to all data in time axis.
    flag_all_freq_frac : float
        Fraction of data flagged above which to extend flags to all data in frequency axis.
    rho : float
        Falloff exponent for SumThreshold
    """
    def __init__(self, outlier_nsigma=4.5, windows_time=[1, 2, 4, 8],
                 windows_freq=[1, 2, 4, 8], background_reject=2.0, background_iterations=1,
                 spike_width_time=12.5, spike_width_freq=10.0, time_extend=3, freq_extend=3,
                 freq_chunks=10, average_freq=1, flag_all_time_frac=0.6, flag_all_freq_frac=0.8,
                 rho=0.3):
        self.outlier_nsigma = outlier_nsigma
        self.windows_time = windows_time
        # Scale the frequency windows, and remove possible duplicates
        windows_freq = np.ceil(np.array(windows_freq, dtype=np.float32) / average_freq)
        self.windows_freq = np.unique(windows_freq.astype(np.int))
        self.background_reject = background_reject
        self.background_iterations = background_iterations
        self.spike_width_time = spike_width_time
        # Scale spike_width by average_freq
        self.spike_width_freq = spike_width_freq / average_freq
        self.time_extend = time_extend
        self.freq_extend = freq_extend
        self.freq_chunks = freq_chunks
        self.average_freq = average_freq
        self.flag_all_time_frac = flag_all_time_frac
        self.flag_all_freq_frac = flag_all_freq_frac
        self.rho = 1.3

    def get_flags(self, data, flags, pool=None):
        """Get flags in data array, with optional input flags of same shape
        that denote samples in data to ignore when backgrounding and deriving
        thresholds.

        Parameters
        ----------
        data : 3D array
            The input visibility data, in (time, frequency, baseline) order. It may
            also contain just the magnitudes.
        flags : 3D array, boolean
            Input flags.
        pool : `multiprocessing.Pool` or similar
            Worker pool for parallel computation. If not specified,
            computation will be done serially.

        Returns
        -------
        out_flags : 3D array, boolean, same shape as `data`
            Derived flags (True=flagged)

        """

        out_flags = np.empty(data.shape, dtype=np.bool)
        # This is redundant because get_baseline_flags also does this check,
        # but it avoids having to pickle the full complex values
        if np.iscomplexobj(data) and isinstance(pool, multiprocessing.pool.Pool):
            data = np.abs(data)
        if pool is not None:
            async_results = []
            for i in range(data.shape[-1]):
                async_results.append(pool.apply_async(get_baseline_flags,
                                                      (self, data[..., i], flags[..., i])))
            for i, result in enumerate(async_results):
                out_flags[..., i] = result.get()
        else:
            for i in range(data.shape[-1]):
                out_flags[..., i] = self.get_baseline_flags(data[..., i], flags[..., i])
        return out_flags

    def _average_freq(self, data, flags):
        """Average the frequency axis (axis 1) of data into bins
        of size `self.average_freq`, ignoring flags.
        If all data in a bin is flagged in the input the data are
        averaged and the output is flagged.
        If `self.average_freq` does not divide the frequency axis
        of data, the last channel of `avg_data` will be an average of
        the remaining channels in the last bin.

        Parameters
        ----------
        data : 2D Array, real
            Input data to average.
        flags : 2D Array, boolean
            Input flags of data to ignore when averaging.

        Returns
        -------
        avg_data : 2D Array, real
            The averaged data.

        avg_flags : 2D Array, boolean
            Averaged flags array- only data for which an entire averaging bin
            is flagged in the input will be flagged in avg_flags.

        """

        freq_length = data.shape[1]
        bin_edges = np.arange(0, freq_length, self.average_freq, dtype=np.int)
        data_sum = np.add.reduceat(data * ~flags, bin_edges, axis=1)
        weights = np.add.reduceat(~flags, bin_edges, axis=1, dtype=np.float32)
        avg_flags = (weights == 0.)
        # Fix weights to bin size where all data are flagged
        # This weight will be wrong for the end remainder, but data are flagged so it doesn't matter
        weights = np.where(avg_flags, float(self.average_freq), weights)
        avg_data = data_sum/weights

        return avg_data, avg_flags

    def get_baseline_flags(self, in_data, in_flags):
        """For a time,channel ordered input data find outliers, and extrapolate
        flags in extreme cases.

        Parameters
        ----------
        in_data : 2D Array, float or complex
            Array of input data. Should be in (time, channel) order, and be either
            complex visibilities or their absolute values.
        in_flags : 2D Array, boolean
            Array of input flags. Used to ignore points when backgrounding and
            thresholding.

        Returns
        -------
        out_flags : 2D Array, boolean
            The output flags for the given baseline (same shape as `in_data`)
        """
        # Convert to magnitudes
        if np.iscomplexobj(in_data):
            in_data = np.abs(in_data)
        else:
            in_data = in_data.copy()   # Will be modified
        # Get nonfinite locations
        nf_locs = ~np.isfinite(in_data)
        in_flags = in_flags.copy()
        # Flag nonfinite locations in input
        in_flags[nf_locs] = True
        # Replace nonfinite data with zero
        in_data[nf_locs] = 0.
        # Average `in_data` in frequency if requested
        # (we should have a copy of `in_data` and `in_flags` at this point)
        orig_freq_shape = in_data.shape[1]
        if self.average_freq > 1:
            in_data, in_flags = self._average_freq(in_data, in_flags)
        # Set up output flags
        out_flags = np.empty(in_data.shape, dtype=np.bool)

        # Set up frequency chunks
        freq_chunks = np.linspace(0, in_data.shape[1], self.freq_chunks+1, dtype=np.int)
        # Number of channels to pad start and end of each chunk
        # (factor of 3 is gaussian smoothing box size in getbackground)
        chunk_overlap = int(np.ceil(self.spike_width_freq * self.background_iterations * 3.))
        # Loop over chunks
        for chunk_num in range(len(freq_chunks)-1):
            # Chunk the input data and flags in frequency and create output chunk
            chunk_start, chunk_end = freq_chunks[chunk_num:chunk_num + 2]
            chunk_size = chunk_end - chunk_start
            chunk = slice(max(chunk_start - chunk_overlap, 0),
                          min(chunk_end + chunk_overlap, in_data.shape[1]))
            in_data_chunk = in_data[:, chunk]
            in_flags_chunk = np.copy(in_flags[:, chunk])
            out_flags_chunk = np.zeros_like(in_flags_chunk, dtype=np.bool)

            # Flag a 1d median frequency spectrum
            spec_flags = np.all(in_flags_chunk, axis=0, keepdims=True)
            # Un-flag channels that are completely flagged to avoid nans in the median
            in_flags_chunk[:, spec_flags[0]] = False
            # Get median spectrum
            masked_data = np.where(in_flags_chunk, np.nan, in_data_chunk)
            spec_data = np.nanmedian(masked_data, axis=0, keepdims=True)
            # Re-flag channels that are completely flagged in time.
            in_flags_chunk[:, spec_flags[0]] = True
            spec_background = getbackground_2d(spec_data, in_flags=spec_flags,
                                               iterations=self.background_iterations,
                                               spike_width=(0.0, self.spike_width_freq),
                                               reject_threshold=self.background_reject)
            av_dev = spec_data - spec_background
            spec_flags = self._sumthreshold(av_dev, spec_flags, 1, self.windows_freq)
            # Broadcast spec_flags to timestamps
            out_flags_chunk[:] = spec_flags
            # Bootstrap the spec_flags in the mask from now on
            in_flags_chunk |= out_flags_chunk
            # Flag the data in 2d
            # Get the background in the current chunk
            spike_width = (self.spike_width_time, self.spike_width_freq)
            background_chunk = getbackground_2d(in_data_chunk, in_flags=in_flags_chunk,
                                                iterations=self.background_iterations,
                                                spike_width=spike_width,
                                                reject_threshold=self.background_reject)
            # Subtract background
            av_dev = in_data_chunk - background_chunk
            # SumThreshold along time axis
            time_flags_chunk = self._sumthreshold(av_dev, in_flags_chunk, 0, self.windows_time)
            # SumThreshold along frequency axis- with time flags in the mask
            freq_mask = in_flags_chunk | time_flags_chunk
            freq_flags_chunk = self._sumthreshold(av_dev, freq_mask, 1, self.windows_freq)
            # Combine all the flags
            out_flags_chunk |= time_flags_chunk | freq_flags_chunk

            # Copy output flags from the current chunk into the correct
            # position in out_flags (ignoring overlap)
            chunk_offset = min(chunk_overlap, chunk_start)
            out_chunk = slice(chunk_offset, chunk_size+chunk_offset)
            out_flags[:, chunk_start:chunk_end] = out_flags_chunk[:, out_chunk]

        # Bring flags to correct frequency shape if the input data was averaged
        # Need to chop the end to the right shape if there was a remainder after averaging
        if self.average_freq > 1:
            out_flags = np.repeat(out_flags, self.average_freq, axis=1)[:, :orig_freq_shape]

        # Extend flags in time and frequency
        # TODO: make it more efficient with something similar to _convolve_flags
        if self.freq_extend > 1:
            kern = np.ones((self.freq_extend,), dtype=np.bool_)
            out_flags = convolve1d(out_flags, kern, axis=1, mode='reflect')
        if self.time_extend > 1:
            kern = np.ones((self.time_extend,), dtype=np.bool)
            out_flags = convolve1d(out_flags, kern, axis=0, mode='reflect')

        # Flag all frequencies and times if too much is flagged.
        flag_frac_time = np.sum(out_flags, dtype=np.float32, axis=0) / out_flags.shape[0]
        out_flags[:, flag_frac_time > self.flag_all_time_frac] = True
        flag_frac_freq = np.sum(out_flags, dtype=np.float32, axis=1) / out_flags.shape[1]
        out_flags[flag_frac_freq > self.flag_all_freq_frac] = True

        # Flag nan location in output
        out_flags[nf_locs] = True

        return out_flags

    def _sumthreshold(self, input_data, flags, axis, windows):
        """Apply the SumThreshold method along the given axis of
        `input_data`.

        Parameters
        ----------
        input_data : 2D Array, real
            Deviations from the background, with shape (time,frequency)
        flags : 2D Array, boolean
            Input flags. Used as a mask when computing the initial
            standard deviations of the input data
        axis : int
            The axis to apply the SumThreshold operation
            0=time, 1=frequency
        windows : Array, int
            Window sizes to average data in each SumThreshold step

        Returns
        -------
        output_flags : 2D Array, boolean
            The derived flags
        """
        if axis == 0:
            input_data = input_data.T
            flags = flags.T
        abs_input = np.empty_like(input_data[0])
        windows = np.array(windows, np.int64)
        output_flags = _sumthreshold_1d(input_data, flags, windows,
                                        self.outlier_nsigma, self.rho)
        if axis == 0:
            output_flags = output_flags.T
        return output_flags
