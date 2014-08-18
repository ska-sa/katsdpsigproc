"""RFI thresholding algorithms. Each algorithm takes as input the
deviations in amplitude from the background, and emits a set of
flags.
"""

import numpy as np
from ..accel import DeviceArray, LinenoLexer, push_context
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from mako.lookup import TemplateLookup
import os.path
import pkg_resources

_lookup = TemplateLookup(pkg_resources.resource_filename(__name__, ''))

class ThresholdHostFromDevice(object):
    """Wraps a device-side thresholder to present the host interface"""
    def __init__(self, real_threshold):
        self.real_threshold = real_threshold

    def __call__(self, deviations):
        padded_shape = self.real_threshold.min_padded_shape(deviations.shape)
        device_deviations = DeviceArray(self.real_threshold.ctx,
                shape=deviations.shape, dtype=np.float32, padded_shape=padded_shape)
        device_deviations.set(deviations)
        device_flags = DeviceArray(self.real_threshold.ctx,
                shape=deviations.shape, dtype=np.uint8, padded_shape=padded_shape)
        self.real_threshold(device_deviations, device_flags)
        return device_flags.get()

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

class ThresholdMADDevice(object):
    """Device-side thresholding by median of absolute deviations. It
    should give the same results as :class:`ThresholdMADHost`, up to
    floating-point accuracy.
    """
    host_class = ThresholdMADHost

    def __init__(self, ctx, n_sigma, wgsx=32, wgsy=8, flag_value=1):
        """Constructor.

        Parameters
        ----------
        ctx : pycuda.driver.Context
            CUDA context
        n_sigma : float
            Number of (estimated) standard deviations for the threshold
        wgsx : int
            Number of baselines per workgroup
        wgsy : int
            Number of channels per workgroup
        flag_value : int
            Number stored in returned value to indicate RFI
        """
        self.ctx = ctx
        self.factor = 1.4826 * n_sigma
        self.wgsx = wgsx
        self.wgsy = wgsy
        self.flag_value = flag_value
        source = _lookup.get_template('threshold_mad.cu').render(
                wgsx=wgsx, wgsy=wgsy,
                flag_value=flag_value)
        with push_context(self.ctx):
            module = SourceModule(source, no_extern_c=True)
            self.kernel = module.get_function('threshold_mad')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (channels, baselines) = shape
        padded_baselines = (baselines + self.wgsx - 1) // self.wgsx * self.wgsx
        return (channels, padded_baselines)

    def _blocks(self, deviations):
        baselines = deviations.shape[1]
        blocks = (baselines + self.wgsx - 1) // self.wgsx
        assert blocks * self.wgsx <= deviations.padded_shape[1]
        return blocks

    def _vt(self, deviations):
        channels = deviations.shape[0]
        vt = (channels + self.wgsy - 1) // self.wgsy
        return vt

    def __call__(self, deviations, flags, stream=None):
        """Apply the thresholding

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude, indexed by channel
            then baseline. It must have a padded size at least that
            given by :meth:`min_padded_shape`.
        flags : :class:`katsdpsigproc.accel.DeviceArray`, uint8
            Output flags
        stream : pycuda.driver.Stream or None
            CUDA stream for enqueueing the work
        """
        assert deviations.shape == flags.shape
        assert deviations.padded_shape == flags.padded_shape
        (channels, baselines) = deviations.shape
        blocks = self._blocks(deviations)
        vt = self._vt(deviations)
        with push_context(self.ctx):
            self.kernel(
                    deviations.buffer, flags.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1]),
                    np.float32(self.factor), np.int32(vt),
                    block=(self.wgsx, self.wgsy, 1), grid=(blocks, 1), stream=stream)
