"""Backgrounding algorithms. Each algorithm takes a set of complex
visibilities as input (indexed by channel then baseline), and returns a
deviation in amplitude from the background, of the same shape.
"""

from ..accel import DeviceArray, LinenoLexer, push_context
import numpy as np
import scipy.signal as signal
from mako.lookup import TemplateLookup
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import pycuda.driver as cuda
import pkg_resources

_lookup = TemplateLookup(pkg_resources.resource_filename(__name__, ''))

class BackgroundHostFromDevice(object):
    """Wraps a device-side backgrounder to present the host interface"""
    def __init__(self, real_background):
        self.real_background = real_background

    def __call__(self, vis):
        padded_shape = self.real_background.min_padded_shape(vis.shape)
        device_vis = DeviceArray(
                self.real_background.ctx, vis.shape, np.complex64, padded_shape)
        device_vis.set(vis)
        device_deviations = DeviceArray(
                self.real_background.ctx, vis.shape, np.float32, padded_shape)
        self.real_background(device_vis, device_deviations)
        return device_deviations.get()

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

class BackgroundMedianFilterDevice(object):
    """Device algorithm that applies a median filter to each baseline
    (in amplitude). It is the same algorithm as
    :class:`BackgroundMedianFilterHost`, but may give slightly different
    results due to rounding errors when computing complex magnitude.
    """

    host_class = BackgroundMedianFilterHost

    def __init__(self, ctx, width, wgs=128, csplit=8):
        """Constructor.

        Parameters
        ----------
        ctx : pycuda.driver.Context
            CUDA context
        width : int
            The kernel width (must be odd)
        wgs : int
            Number of baselines per workgroup
        csplit : int
            Approximate number of thread cooperating on each channel
        """
        self.ctx = ctx
        self.width = width
        self.wgs = wgs
        self.csplit = csplit
        source = _lookup.get_template('background_median_filter.cu').render(
                width=width, wgs=wgs)
        with push_context(self.ctx):
            module = SourceModule(source, no_extern_c=True)
            self.kernel = module.get_function('background_median_filter')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (channels, baselines) = shape
        padded_baselines = (baselines + self.wgs - 1) // self.wgs * self.wgs
        return (channels, padded_baselines)

    def __call__(self, vis, deviations, stream=None):
        """Perform the backgrounding.

        Parameters
        ----------
        vis : :class:`katsdpsigproc.accel.DeviceArray`, complex64
            Input visibilities: 2D complex64 array indexed by channel
            then baseline. The padded size must be at least
            :meth:`min_padded_shape`.
        deviations : :class:`katsdpsigproc.accel.DeviceArray`
            Output deviations, of the same shape and padding as `vis`.
        stream : pycuda.driver.Stream or None
            CUDA stream to enqueue work to
        """
        assert vis.shape == deviations.shape
        assert vis.padded_shape == deviations.padded_shape
        (channels, baselines) = vis.shape
        N = baselines * channels
        H = self.width // 2
        VT = max(channels // self.csplit, 1)
        xblocks = (baselines + self.wgs - 1) // self.wgs
        yblocks = (channels + VT - 1) // VT
        assert xblocks * self.wgs <= vis.padded_shape[1]

        with push_context(self.ctx):
            self.kernel(
                    vis.buffer, deviations.buffer,
                    np.int32(channels), np.int32(vis.padded_shape[1]), np.int32(VT),
                    block=(self.wgs, 1, 1), grid=(xblocks, yblocks), stream=stream)
