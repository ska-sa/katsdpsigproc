"""RFI flagging algorithms that run on the CPU."""

import numpy as np
import scipy.signal as signal

class BackgroundMedianFilterHost(object):
    """Host backgrounder that applies a median filter to each baseline
    (by amplitude).
    """
    def __init__(self, width):
        """Constructor.

        Parameters
        ----------
        width : int
            The kernel width (must be odd)
        """
        self.width = width

    def __call__(self, vis):
        amp = np.abs(vis)
        return amp - signal.medfilt2d(amp, [self.width, 1])

class ThresholdMADHost(object):
    """Thresholding on median of absolute deviations."""
    def __init__(self, n_sigma, flag_value=1):
        """Constructor.

        Parameters
        ----------
        n_sigma : float
            Number of (estimated) standard deviations for the threshold
        flag_value : int
            Number stored in returned value to indicate RFI
        """
        self.factor = 1.4826 * n_sigma
        self.flag_value = flag_value

    def median_abs(self, deviations):
        """Find the median of absolute deviations (amongst the non-zero
        elements).
        """
        (channels, baselines) = deviations.shape
        out = np.empty(baselines)
        for i in range(baselines):
            abs_dev = np.abs(deviations[:, i])
            out[i] = np.median(abs_dev[abs_dev > 0])
        return out

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
        medians = self.median_abs(deviations)
        flags = (deviations > self.factor * medians).astype(np.uint8)
        return flags * self.flag_value

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
        numpy.ndarray
            Flags of the same shape as `vis`
        """

        deviations = self.background(vis)
        flags = self.threshold(deviations)
        return flags
