"""RFI flagging algorithms that run on an accelerator (currently only
CUDA).

The thresholding functions may take data in either channel-major or
baseline-major order (the flags are emitted in the same order). In the former
case, the `transposed` member is `False`, otherwise it is `True`. The flagger
classes automatically detect this and apply a transposition kernel.
"""

from .. import accel
from ..accel import DeviceArray, LinenoLexer, Transpose
import numpy as np
from . import host

class BackgroundHostFromDevice(object):
    """Wraps a device-side backgrounder to present the host interface"""
    def __init__(self, real_background):
        self.real_background = real_background

    def __call__(self, vis):
        padded_shape = self.real_background.min_padded_shape(vis.shape)
        device_vis = DeviceArray(
                self.real_background.command_queue.context, vis.shape, np.complex64, padded_shape)
        device_vis.set(self.real_background.command_queue, vis)
        device_deviations = DeviceArray(
                self.real_background.command_queue.context, vis.shape, np.float32, padded_shape)
        self.real_background(device_vis, device_deviations)
        return device_deviations.get(self.real_background.command_queue)

class BackgroundMedianFilterDevice(object):
    """Device algorithm that applies a median filter to each baseline
    (in amplitude). It is the same algorithm as
    :class:`BackgroundMedianFilterHost`, but may give slightly different
    results due to rounding errors when computing complex magnitude.
    """

    host_class = host.BackgroundMedianFilterHost

    def __init__(self, command_queue, width, wgs=128, csplit=8):
        """Constructor.

        Parameters
        ----------
        context : pycuda.driver.Context
            CUDA context
        width : int
            The kernel width (must be odd)
        wgs : int
            Number of baselines per workgroup
        csplit : int
            Approximate number of thread cooperating on each channel
        """
        self.command_queue = command_queue
        self.width = width
        self.wgs = wgs
        self.csplit = csplit
        program = accel.build(command_queue.context, 'rfi/background_median_filter.mako',
                {'width': width, 'wgs': wgs})
        self.kernel = program.get_kernel('background_median_filter')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (channels, baselines) = shape
        padded_baselines = (baselines + self.wgs - 1) // self.wgs * self.wgs
        return (channels, padded_baselines)

    def __call__(self, vis, deviations):
        """Perform the backgrounding.

        Parameters
        ----------
        vis : :class:`katsdpsigproc.accel.DeviceArray`, complex64
            Input visibilities: 2D complex64 array indexed by channel
            then baseline. The padded size must be at least
            :meth:`min_padded_shape`.
        deviations : :class:`katsdpsigproc.accel.DeviceArray`
            Output deviations, of the same shape and padding as `vis`.
        """
        assert vis.shape == deviations.shape
        assert vis.padded_shape == deviations.padded_shape
        (channels, baselines) = vis.shape
        VT = max(channels // self.csplit, 1)
        xblocks = (baselines + self.wgs - 1) // self.wgs
        yblocks = (channels + VT - 1) // VT
        assert xblocks * self.wgs <= vis.padded_shape[1]

        self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    vis.buffer,
                    deviations.buffer,
                    np.int32(channels),
                    np.int32(vis.padded_shape[1]),
                    np.int32(VT)
                ],
                global_size=(xblocks * self.wgs, yblocks),
                local_size=(self.wgs, 1))

class ThresholdHostFromDevice(object):
    """Wraps a device-side thresholder to present the host interface"""
    def __init__(self, real_threshold):
        self.real_threshold = real_threshold

    def __call__(self, deviations):
        if self.real_threshold.transposed:
            deviations = deviations.T
        padded_shape = self.real_threshold.min_padded_shape(deviations.shape)
        device_deviations = DeviceArray(self.real_threshold.command_queue.context,
                shape=deviations.shape, dtype=np.float32, padded_shape=padded_shape)
        device_deviations.set(self.real_threshold.command_queue, deviations)
        device_flags = DeviceArray(self.real_threshold.command_queue.context,
                shape=deviations.shape, dtype=np.uint8, padded_shape=padded_shape)
        self.real_threshold(device_deviations, device_flags)
        flags = device_flags.get(self.real_threshold.command_queue)
        if self.real_threshold.transposed:
            flags = flags.T
        return flags

class ThresholdMADDevice(object):
    """Device-side thresholding by median of absolute deviations. It
    should give the same results as :class:`ThresholdMADHost`, up to
    floating-point accuracy.

    See :class:`ThresholdMADTDevice` for a more efficient implementation.
    """
    host_class = host.ThresholdMADHost
    transposed = False

    def __init__(self, command_queue, n_sigma, wgsx=32, wgsy=8, flag_value=1):
        """Constructor.

        Parameters
        ----------
        context : pycuda.driver.Context
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
        self.command_queue = command_queue
        self.factor = 1.4826 * n_sigma
        self.wgsx = wgsx
        self.wgsy = wgsy
        self.flag_value = flag_value
        program = accel.build(command_queue.context, 'rfi/threshold_mad.mako',
                {'wgsx': wgsx, 'wgsy': wgsy, 'flag_value': flag_value})
        self.kernel = program.get_kernel('threshold_mad')

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

    def __call__(self, deviations, flags):
        """Apply the thresholding

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude, indexed by channel
            then baseline. It must have a padded size at least that
            given by :meth:`min_padded_shape`.
        flags : :class:`katsdpsigproc.accel.DeviceArray`, uint8
            Output flags
        """
        assert deviations.shape == flags.shape
        assert deviations.padded_shape == flags.padded_shape
        channels = deviations.shape[0]
        blocks = self._blocks(deviations)
        vt = self._vt(deviations)
        self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    deviations.buffer, flags.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1]),
                    np.float32(self.factor), np.int32(vt)
                ],
                global_size=(blocks * self.wgsx, self.wgsy),
                local_size=(self.wgsx, self.wgsy))

class ThresholdMADTDevice(object):
    """Device-side thresholding by median of absolute deviations. It
    should give the same results as :class:`ThresholdMADHost`, up to
    floating-point accuracy. It uses transposed (baseline-major) memory
    order, which allows an entire baseline to be efficiently loaded into
    registers.

    Attributes
    ----------
    factor : float
        Multiple of median absolute deviation at which detection occurs
    max_channels : int
        Maximum number of channels that can be supported
    _vt : int
        Number of elements handled by each thread
    _wgsx : int
        Number of work-items per baseline

    Notes
    -----
    There is a tradeoff in selecting the workgroup size: a large value gives
    more parallelism and reduces the register pressure, but increases the
    overhead of reduction operations.

    This class may fail for very large numbers of channels (10k can
    definitely be supported), in which case ThresholdMADDevice may be used.
    """
    host_class = host.ThresholdMADHost
    transposed = True

    def __init__(self, command_queue, n_sigma, max_channels, flag_value=1):
        """Constructor.

        Parameters
        ----------
        context : pycuda.driver.Context
            CUDA context
        n_sigma : float
            Number of (estimated) standard deviations for the threshold
        max_channels : int
            Maximum number of channels. Choosing too large a value will
            reduce performance.
        flag_value : int
            Number stored in returned value to indicate RFI
        """
        self.command_queue = command_queue
        self.factor = 1.4826 * n_sigma
        self.flag_value = flag_value
        self.max_channels = max_channels
        self._vt = 16
        self._wgsx = 512
        self._vt = (max_channels + self._wgsx - 1) // self._wgsx
        program = accel.build(command_queue.context, 'rfi/threshold_mad_t.mako',
                {'vt': self._vt, 'wgsx': self._wgsx, 'flag_value': flag_value})
        self.kernel = program.get_kernel('threshold_mad_t')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (baselines, channels) = shape
        padded_channels = (channels + 31) // 32 * 32
        return (baselines, padded_channels)

    def __call__(self, deviations, flags):
        """Apply the thresholding

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude, indexed by baseline
            then channel. It must have a padded size at least that
            given by :meth:`min_padded_shape`.
        flags : :class:`katsdpsigproc.accel.DeviceArray`, uint8
            Output flags
        """
        assert deviations.shape == flags.shape
        assert deviations.padded_shape == flags.padded_shape
        (baselines, channels) = deviations.shape
        assert channels <= self.max_channels
        self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    deviations.buffer, flags.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1]),
                    np.float32(self.factor)
                ],
                global_size=(self._wgsx, baselines),
                local_size=(self._wgsx, 1))

class FlaggerDevice(object):
    """Combine device backgrounder and thresholder implementations to
    create a flagger.
    """
    def __init__(self, background, threshold):
        self.background = background
        self.threshold = threshold
        assert self.background.command_queue is self.threshold.command_queue
        self.command_queue = self.background.command_queue
        self._deviations = None
        if threshold.transposed:
            self._deviations_t = None
            self._flags_t = None
            self._transpose_deviations = Transpose(self.command_queue, 'float')
            self._transpose_flags = Transpose(self.command_queue, 'unsigned char')

    def min_padded_shape(self, shape):
        """The minimum padding needed for the inputs and outputs of the
        flagger."""
        if self.threshold.transposed:
            return self.background.min_padded_shape(shape)
        else:
            return map(max,
                    self.background.min_padded_shape(shape),
                    self.threshold.min_padded_shape(shape))

    def __call__(self, vis, flags):
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
        """
        assert vis.shape == flags.shape
        assert vis.padded_shape == flags.padded_shape
        assert np.all(np.greater_equal(
                self.min_padded_shape(vis.shape), vis.padded_shape))

        if (self._deviations is None or
                self._deviations.shape != vis.shape or
                self._deviations.padded_shape != vis.padded_shape):
            self._deviations = DeviceArray(
                    self.command_queue.context, vis.shape, np.float32, vis.padded_shape)
        if self.threshold.transposed:
            shape_t = (vis.shape[1], vis.shape[0])
            padded_shape_t = self.threshold.min_padded_shape(shape_t)
            if (self._deviations_t is None or
                    self._deviations_t.shape != shape_t or
                    self._deviations_t.padded_shape != padded_shape_t):
                self._deviations_t = DeviceArray(
                        self.command_queue.context, shape_t, np.float32, padded_shape_t)
                self._flags_t = DeviceArray(
                        self.command_queue.context, shape_t, np.uint8, padded_shape_t)

        self.background(vis, self._deviations)
        if self.threshold.transposed:
            self._transpose_deviations(self._deviations_t, self._deviations)
            self.threshold(self._deviations_t, self._flags_t)
            self._transpose_flags(flags, self._flags_t)
        else:
            self.threshold(self._deviations, flags)

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
        device_vis = DeviceArray(self.real_flagger.command_queue.context, vis.shape,
                np.complex64, padded_shape)
        device_vis.set(self.real_flagger.command_queue, vis)
        device_flags = DeviceArray(self.real_flagger.command_queue.context, vis.shape,
                np.uint8, padded_shape)
        self.real_flagger(device_vis, device_flags)
        return device_flags.get(self.real_flagger.command_queue)
