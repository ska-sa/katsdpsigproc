"""RFI flagging algorithms that run on an accelerator (currently only
CUDA).
"""

from ..accel import DeviceArray, LinenoLexer, push_context, Transpose
import numpy as np
from mako.lookup import TemplateLookup
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import pkg_resources
from . import host

_lookup = TemplateLookup(
        pkg_resources.resource_filename(__name__, ''), lexer_cls=LinenoLexer)
_nvcc_flags = DEFAULT_NVCC_FLAGS + ['-lineinfo']

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

class BackgroundMedianFilterDevice(object):
    """Device algorithm that applies a median filter to each baseline
    (in amplitude). It is the same algorithm as
    :class:`BackgroundMedianFilterHost`, but may give slightly different
    results due to rounding errors when computing complex magnitude.
    """

    host_class = host.BackgroundMedianFilterHost

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
            module = SourceModule(source, options=_nvcc_flags, no_extern_c=True)
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
        VT = max(channels // self.csplit, 1)
        xblocks = (baselines + self.wgs - 1) // self.wgs
        yblocks = (channels + VT - 1) // VT
        assert xblocks * self.wgs <= vis.padded_shape[1]

        with push_context(self.ctx):
            self.kernel(
                    vis.buffer,
                    deviations.buffer,
                    np.int32(channels),
                    np.int32(vis.padded_shape[1]),
                    np.int32(VT),
                    block=(self.wgs, 1, 1), grid=(xblocks, yblocks),
                    stream=stream)

class ThresholdHostFromDevice(object):
    """Wraps a device-side thresholder to present the host interface"""
    def __init__(self, real_threshold):
        self.real_threshold = real_threshold

    def __call__(self, deviations):
        if self.real_threshold.transposed:
            deviations = deviations.T
        padded_shape = self.real_threshold.min_padded_shape(deviations.shape)
        device_deviations = DeviceArray(self.real_threshold.ctx,
                shape=deviations.shape, dtype=np.float32, padded_shape=padded_shape)
        device_deviations.set(deviations)
        device_flags = DeviceArray(self.real_threshold.ctx,
                shape=deviations.shape, dtype=np.uint8, padded_shape=padded_shape)
        self.real_threshold(device_deviations, device_flags)
        flags = device_flags.get()
        if self.real_threshold.transposed:
            flags = flags.T
        return flags

class ThresholdMADDevice(object):
    """Device-side thresholding by median of absolute deviations. It
    should give the same results as :class:`ThresholdMADHost`, up to
    floating-point accuracy.
    """
    host_class = host.ThresholdMADHost
    transposed = False

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
            module = SourceModule(source, options=_nvcc_flags, no_extern_c=True)
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
        channels = deviations.shape[0]
        blocks = self._blocks(deviations)
        vt = self._vt(deviations)
        with push_context(self.ctx):
            self.kernel(
                    deviations.buffer, flags.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1]),
                    np.float32(self.factor), np.int32(vt),
                    block=(self.wgsx, self.wgsy, 1), grid=(blocks, 1), stream=stream)

class ThresholdMADTDevice(object):
    """Device-side thresholding by median of absolute deviations. It
    should give the same results as :class:`ThresholdMADHost`, up to
    floating-point accuracy.
    """
    host_class = host.ThresholdMADHost
    transposed = True
    _vt = 16
    _wgsy = 1

    def __init__(self, ctx, n_sigma, max_channels, flag_value=1):
        """Constructor.

        Parameters
        ----------
        ctx : pycuda.driver.Context
            CUDA context
        n_sigma : float
            Number of (estimated) standard deviations for the threshold
        max_channels : int
            Maximum number of channels. Choosing too large a value will
            reduce performance.
        flag_value : int
            Number stored in returned value to indicate RFI
        """
        self.ctx = ctx
        self.factor = 1.4826 * n_sigma
        self.flag_value = flag_value
        source = _lookup.get_template('threshold_mad_t.cu').render(
                vt=self._vt,
                wgsy=self._wgsy,
                flag_value=flag_value)
        with push_context(self.ctx):
            module = SourceModule(source, options=_nvcc_flags, no_extern_c=True)
            self.kernel = module.get_function('threshold_mad_t')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (baselines, channels) = shape
        padded_channels = (channels + 31) // 32 * 32
        return (baselines, padded_channels)

    def __call__(self, deviations, flags, stream=None):
        """Apply the thresholding

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude, indexed by baseline
            then channel. It must have a padded size at least that
            given by :meth:`min_padded_shape`.
        flags : :class:`katsdpsigproc.accel.DeviceArray`, uint8
            Output flags
        stream : pycuda.driver.Stream or None
            CUDA stream for enqueueing the work
        """
        assert deviations.shape == flags.shape
        assert deviations.padded_shape == flags.padded_shape
        (baselines, channels) = deviations.shape
        block_width = (channels + self._vt - 1) // self._vt
        with push_context(self.ctx):
            self.kernel(
                    deviations.buffer, flags.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1]),
                    np.float32(self.factor),
                    block=(block_width, 1, 1), grid=(1, baselines), stream=stream)

class FlaggerDevice(object):
    """Combine device backgrounder and thresholder implementations to
    create a flagger.
    """
    def __init__(self, background, threshold):
        self.background = background
        self.threshold = threshold
        self.ctx = self.background.ctx
        self._deviations = None
        if threshold.transposed:
            self._deviations_t = None
            self._flags_t = None
            self._transpose_deviations = Transpose(self.ctx, 'float')
            self._transpose_flags = Transpose(self.ctx, 'unsigned char')

    def min_padded_shape(self, shape):
        """The minimum padding needed for the inputs and outputs of the
        flagger."""
        if self.threshold.transposed:
            return self.background.min_padded_shape(shape)
        else:
            return map(max,
                    self.background.min_padded_shape(shape),
                    self.threshold.min_padded_shape(shape))

    def __call__(self, vis, flags, stream=None):
        """Perform the flagging.

        Note
        ----
        Temporary intermediate data is stored in the class and used
        asynchronously by the device. This means that for concurrent
        flagging of multiple visibility sets, one must use multiple
        instances of this class.

        Parameters
        ----------
        vis : :class:`katsdpsigproc.accel.DeviceArray`
            The input visibilities as a 2D array of complex64, indexed
            by channel and baseline, and with padded size given by
            :meth:`min_padded_shape`.
        flags : :class:katsdpsigproc.accel.DeviceArray`
            The output flags, with the same shape and padding as `vis`.
        stream : `pycuda.driver.Stream`
            CUDA stream for enqueuing the operations
        """
        assert vis.shape == flags.shape
        assert vis.padded_shape == flags.padded_shape
        assert np.all(np.greater_equal(
                self.min_padded_shape(vis.shape), vis.padded_shape))

        if (self._deviations is None or
                self._deviations.shape != vis.shape or
                self._deviations.padded_shape != vis.padded_shape):
            self._deviations = DeviceArray(
                    self.ctx, vis.shape, np.float32, vis.padded_shape)
        if self.threshold.transposed:
            shape_t = (vis.shape[1], vis.shape[0])
            padded_shape_t = self.threshold.min_padded_shape(shape_t)
            if (self._deviations_t is None or
                    self._deviations_t.shape != shape_t or
                    self._deviations_t.padded_shape != padded_shape_t):
                self._deviations_t = DeviceArray(
                        self.ctx, shape_t, np.float32, padded_shape_t)
                self._flags_t = DeviceArray(
                        self.ctx, shape_t, np.uint8, padded_shape_t)

        self.background(vis, self._deviations, stream)
        if self.threshold.transposed:
            self._transpose_deviations(self._deviations_t, self._deviations, stream)
            self.threshold(self._deviations_t, self._flags_t, stream)
            self._transpose_flags(flags, self._flags_t, stream)
        else:
            self.threshold(self._deviations, flags, stream)

class FlaggerHostFromDevice(object):
    """Wrapper that makes a :class:`FlaggerDeviceFromHost` present the
    interface of :class:`FlaggerHost`. This is intended only for ease of
    use. It is not efficient, because it allocates and frees memory on
    every call.
    """
    def __init__(self, real_flagger):
        self.real_flagger = real_flagger

    def __call__(self, vis):
        padded_shape = self.real_flagger.min_padded_shape(vis.shape)
        device_vis = DeviceArray(self.real_flagger.ctx, vis.shape,
                np.complex64, padded_shape)
        device_vis.set(vis)
        device_flags = DeviceArray(self.real_flagger.ctx, vis.shape,
                np.uint8, padded_shape)
        self.real_flagger(device_vis, device_flags)
        return device_flags.get()
