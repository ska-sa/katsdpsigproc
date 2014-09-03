# coding: utf-8
"""RFI flagging algorithms that run on the CPU."""

import numpy as np
import scipy.signal as signal

class BackgroundMedianFilterHost(object):
    """Host backgrounder that applies a median filter to each baseline
    (by amplitude).

    Parameters
    ----------
    width : int
        The kernel width (must be odd)
    """
    def __init__(self, width):
        self.width = width

    def __call__(self, vis):
        amp = np.abs(vis)
        return amp - signal.medfilt2d(amp, [self.width, 1])

def median_abs(deviations):
    """Find the median of absolute deviations (amongst the non-zero
    elements, for each baseline).
    """
    baselines = deviations.shape[1]
    out = np.empty(baselines)
    for i in range(baselines):
        abs_dev = np.abs(deviations[:, i])
        out[i] = np.median(abs_dev[abs_dev > 0])
    return out

class ThresholdMADHost(object):
    """Thresholding on median of absolute deviations.

    Parameters
    ----------
    n_sigma : float
        Number of (estimated) standard deviations for the threshold
    flag_value : int
        Number stored in returned value to indicate RFI
    """
    def __init__(self, n_sigma, flag_value=1):
        self.factor = 1.4826 * n_sigma
        self.flag_value = flag_value

    def __call__(self, deviations):
        """Apply the thresholding

        Parameters
        ----------
        deviations : array-like, real
            Deviations from the background amplitude, indexed by channel
            then baseline.

        Returns
        -------
        array-like
            Array of `np.uint8`, containing the flag value or 0
        """
        medians = median_abs(deviations)
        flags = (deviations > self.factor * medians).astype(np.uint8)
        return flags * self.flag_value

class ThresholdSumHost(object):
    """Thresholding using the Offringa Sum-Threshold algorithm, with
    power-of-two sized windows. The initial (single-pixel) threshold
    is determined by median of absolute deviations.

    At present, auto- and cross-correlations are treated the same.

    Parameters
    ----------
    n_sigma : float
        Number of (estimated) standard deviations for the threshold
    n_windows : int
        Number of window sizes to use
    threshold_falloff : float
        Controls rate at which thresholds decrease (ρ in Offringa 2010)
    flag_value : int
        Number stored in returned value to indicate RFI
    """

    def __init__(self, n_sigma, n_windows=4, threshold_falloff=1.2, flag_value=1):
        self.factor = 1.4826 * n_sigma
        self.windows = [2 ** i for i in range(n_windows)]
        self.threshold_scales = [1.0 / (1.5 ** i) for i in range(n_windows)]
        self.flag_value = flag_value

    def apply_baseline(self, deviations, threshold1):
        """Apply the thresholding to a single baseline.
        The flags are returned as booleans, rather than
        `flag_value`.

        Parameters
        ----------
        deviations : 1D array, float
            Deviations from the background
        threshold1 : float
            Threshold for RFI on individual samples
        """

        # The data are modified, so use a copy
        deviations = deviations.copy()
        flags = np.zeros_like(deviations, dtype=np.bool)
        for window, scale in zip(self.windows, self.threshold_scales):
            threshold = threshold1 * scale
            # Force already identified outliers to the threshold
            deviations[flags] = threshold
            # Compute sums
            weight = np.ones(window)
            sums = np.convolve(deviations, weight, mode='valid')
            # Identify outlier sums
            sum_flags = sums > threshold * window
            # Distribute flags
            weight = np.ones(window, dtype=np.bool)
            flags |= np.convolve(sum_flags, weight)
        return flags

    def __call__(self, deviations):
        """Apply the thresholding

        Parameters
        ----------
        deviations : array-like, real
            Deviations from the background amplitude, indexed by channel
            then baseline.

        Returns
        -------
        array-like
            Array of `np.uint8`, containing the flag value or 0
        """
        flags = np.empty_like(deviations, dtype=np.uint8)
        baselines = deviations.shape[1]
        medians = median_abs(deviations)
        for i in range(baselines):
            bl_flags = self.apply_baseline(deviations[:, i], self.factor * medians[i])
            flags[:, i] = bl_flags * np.uint8(self.flag_value)
        return flags

class FlaggerHost(object):
    """Combine host background and thresholding implementations
    to make a flagger.
    """

    def __init__(self, background, threshold):
        self.background = background
        self.threshold = threshold

    def __call__(self, vis):
        """Perform the flagging.

        Parameters
        ----------
        vis : array_like
            The input visibilities as a 2D array of complex64, indexed
            by channel and baseline.

        Returns
        -------
        :class:`numpy.ndarray`
            Flags of the same shape as `vis`
        """

        deviations = self.background(vis)
        flags = self.threshold(deviations)
        return flags
