################################################################################
# Copyright (c) 2014-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""RFI flagging algorithms that run on the CPU."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from . import MAD_NORMAL


class AbstractBackgroundHost(ABC):
    @abstractmethod
    def __init__(self, width: int, amplitudes: bool = False) -> None:
        pass    # pragma: nocover

    @abstractmethod
    def __call__(self, vis: np.ndarray, flags: Optional[np.ndarray] = None) -> np.ndarray:
        """Subtract an estimate of background signal (without RFI).

        Parameters
        ----------
        vis : array-like, float or complex
            Visibilities, either as complex values or amplitudes, depending on
            how the class was constructed.
        flags : array-like, int
            Initial flags, indicating data known to be bad prior to RFI flagging.

        Returns
        -------
        deviations
            Amplitude of signal after subtracting background.
        """


class AbstractNoiseEstHost(ABC):
    @abstractmethod
    def __call__(self, deviations: np.ndarray) -> np.ndarray:
        """Compute the estimated standard deviation of noise per baseline.

        Parameters
        ----------
        deviations
            Deviations from the background amplitude, indexed by channel
            then baseline.

        Returns
        -------
        noise
            1D array of `np.float32`, containing the noise estimate for each channel
        """


class AbstractThresholdHost(ABC):
    @abstractmethod
    def __init__(self, n_sigma: float) -> None:
        pass        # pragma: nocover

    @abstractmethod
    def __call__(self, deviations: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Apply the thresholding.

        Parameters
        ----------
        deviations : array-like, real
            Deviations from the background amplitude, indexed by channel
            then baseline.
        noise : array-like, real
            Per-baseline estimate of the standard deviation for noise

        Returns
        -------
        array-like
            Array of `np.uint8`, containing the flag value or 0
        """


class AbstractFlaggerHost(ABC):
    @abstractmethod
    def __call__(self, vis: np.ndarray, input_flags: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform the flagging.

        Parameters
        ----------
        vis
            The input visibilities as a 2D array of complex64 (or float, if
            that's what the backgrounder expects), indexed by channel and
            baseline.
        input_flags
            Predefined flags as an array of uint8. These can be either
            a 1D array of per-channel flags or a 2D array with the same
            shape as `vis`.

        Returns
        -------
        flags
            Flags of the same shape as `vis`. Note that `input_flags`
            are not copied into the output.
        """


class BackgroundMedianFilterHost(AbstractBackgroundHost):
    """Host backgrounder that applies a median filter to each baseline (by amplitude).

    Parameters
    ----------
    width
        The kernel width (must be odd)
    amplitudes
        If `True`, the inputs are amplitudes rather than complex visibilities
    """

    def __init__(self, width: int, amplitudes: bool = False) -> None:
        self.width = width
        self.amplitudes = amplitudes

    def __call__(self, vis: np.ndarray, flags: Optional[np.ndarray] = None) -> np.ndarray:
        if self.amplitudes:
            amp = pd.DataFrame(vis)
        else:
            amp = pd.DataFrame(np.abs(vis))
        if flags is not None:
            # Convert the flags to bool, and add in a baseline axis if not
            # already present. The mask function in Pandas doesn't
            # automatically broadcast, so we have to do so explicitly with
            # np.broadcast_to.
            flags = flags.astype(np.bool_)
            if flags.ndim < 2:
                flags = flags[:, np.newaxis]
            flags_2d = np.broadcast_to(flags, vis.shape)
            amp = amp.mask(flags_2d)
        med = amp.rolling(self.width, center=True, min_periods=1).median()
        deviation = amp - med
        deviation.fillna(0, inplace=True)
        return deviation.values


class NoiseEstMADHost(AbstractNoiseEstHost):
    """Estimate noise using the median of non-zero absolute deviations."""

    def __call__(self, deviations: np.ndarray) -> np.ndarray:
        baselines = deviations.shape[1]
        out = np.empty(baselines)
        for i in range(baselines):
            abs_dev = np.abs(deviations[:, i])
            out[i] = np.median(abs_dev[abs_dev > 0])
        return out * MAD_NORMAL


class ThresholdSimpleHost(AbstractThresholdHost):
    """Threshold each element independently.

    Parameters
    ----------
    n_sigma
        Number of (estimated) standard deviations for the threshold
    flag_value
        Number stored in returned value to indicate RFI
    """

    def __init__(self, n_sigma: float, flag_value: int = 1) -> None:
        self.n_sigma = n_sigma
        self.flag_value = flag_value

    def __call__(self, deviations: np.ndarray, noise: np.ndarray) -> np.ndarray:
        flags = (deviations > self.n_sigma * noise).astype(np.uint8)
        return flags * self.flag_value


class ThresholdSumHost(AbstractThresholdHost):
    """Thresholding using the Offringa Sum-Threshold algorithm, with power-of-two sized windows.

    The initial (single-pixel) threshold is determined by median of absolute
    deviations.

    At present, auto- and cross-correlations are treated the same.

    Parameters
    ----------
    n_sigma
        Number of (estimated) standard deviations for the threshold
    n_windows
        Number of window sizes to use
    threshold_falloff
        Controls rate at which thresholds decrease (ρ in Offringa 2010)
    flag_value
        Number stored in returned value to indicate RFI
    """

    def __init__(self, n_sigma: float, n_windows: int = 4,
                 threshold_falloff: float = 1.2, flag_value: int = 1) -> None:
        self.n_sigma = n_sigma
        self.windows = [2 ** i for i in range(n_windows)]
        self.threshold_scales = [pow(threshold_falloff, -i) for i in range(n_windows)]
        self.flag_value = flag_value

    def apply_baseline(self, deviations: np.ndarray, threshold1: float) -> np.ndarray:
        """Apply the thresholding to a single baseline.

        The flags are returned as booleans, rather than
        `flag_value`.

        Parameters
        ----------
        deviations : 1D array, float
            Deviations from the background
        threshold1
            Threshold for RFI on individual samples
        """
        # The data are modified, so use a copy
        deviations = deviations.copy()
        flags = np.zeros_like(deviations, dtype=np.bool_)
        for window, scale in zip(self.windows, self.threshold_scales):
            threshold = np.float32(threshold1 * scale)
            # Force already identified outliers to the threshold
            deviations[flags] = threshold
            # Compute sums
            weight = np.ones(window)
            sums = np.convolve(deviations, weight, mode='valid')
            # Identify outlier sums
            sum_flags = sums > threshold * window
            # Distribute flags
            weight = np.ones(window, dtype=np.bool_)
            flags |= np.convolve(sum_flags, weight)
        return flags

    def __call__(self, deviations: np.ndarray, noise: np.ndarray) -> np.ndarray:
        flags = np.empty_like(deviations, dtype=np.uint8)
        baselines = deviations.shape[1]
        for i in range(baselines):
            bl_flags = self.apply_baseline(deviations[:, i], self.n_sigma * noise[i])
            flags[:, i] = bl_flags * np.uint8(self.flag_value)
        return flags


class FlaggerHost(AbstractFlaggerHost):
    """Combine host background and thresholding implementations to make a flagger."""

    def __init__(self, background: AbstractBackgroundHost, noise_est: AbstractNoiseEstHost,
                 threshold: AbstractThresholdHost):
        self.background = background
        self.noise_est = noise_est
        self.threshold = threshold

    def __call__(self, vis: np.ndarray, input_flags: Optional[np.ndarray] = None) -> np.ndarray:
        deviations = self.background(vis, input_flags)
        noise = self.noise_est(deviations)
        return self.threshold(deviations, noise)
