"""RFI flagging algorithms that run on an accelerator (currently only
CUDA).

The noise estimation and thresholding functions may take data in either
channel-major or baseline-major order (the flags are emitted in the same
order). In the former case, the `transposed` member is `False`, otherwise it is
`True`. The flagger classes automatically detect this and apply a transposition
kernel at the appropriate point.
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
    :class:`host.BackgroundMedianFilterHost`, but may give slightly different
    results due to rounding errors when computing complex magnitude.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    width : int
        The kernel width (must be odd)
    wgs : int
        Number of baselines per workgroup
    csplit : int
        Approximate number of workitems cooperating on each channel
    """

    host_class = host.BackgroundMedianFilterHost

    def __init__(self, command_queue, width, wgs=128, csplit=8):
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
        padded_baselines = accel.roundup(baselines, self.wgs)
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
        xblocks = accel.divup(baselines, self.wgs)
        yblocks = accel.divup(channels, VT)
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

class NoiseEstHostFromDevice(object):
    """Wraps a device-side noise estimator to present the host interface"""
    def __init__(self, real_noise_est):
        self.real_noise_est = real_noise_est

    def __call__(self, deviations):
        baselines = deviations.shape[1]
        transposed = self.real_noise_est.transposed
        if transposed:
            deviations = deviations.T
        padded_shape = self.real_noise_est.min_padded_shape(deviations.shape)
        padded_noise_shape = self.real_noise_est.min_padded_noise_shape(baselines)
        # Allocate memory and copy data
        device_deviations = DeviceArray(self.real_noise_est.command_queue.context,
                shape=deviations.shape, dtype=np.float32, padded_shape=padded_shape)
        device_deviations.set(self.real_noise_est.command_queue, deviations)
        device_noise = DeviceArray(self.real_noise_est.command_queue.context,
                shape=(baselines,), dtype=np.float32,
                padded_shape=padded_noise_shape)
        # Perform computation
        self.real_noise_est(device_deviations, device_noise)
        # Copy back results
        noise = device_noise.get(self.real_noise_est.command_queue)
        return noise

class NoiseEstMADDevice(object):
    """Estimate noise using the median of non-zero absolute deviations.

    In most cases NoiseEstMADTDevice is more efficient.

    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    wgsx : int, optional
        Number of baselines per workgroup
    wgsy : int, optional
        Number of channels per workgroup
    """

    host_class = host.NoiseEstMADHost
    transposed = False

    def __init__(self, command_queue, wgsx=32, wgsy=8):
        self.command_queue = command_queue
        self.wgsx = wgsx
        self.wgsy = wgsy
        program = accel.build(command_queue.context, 'rfi/madnz.mako',
                {'wgsx': wgsx, 'wgsy': wgsy})
        self.kernel = program.get_kernel('madnz')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (channels, baselines) = shape
        padded_baselines = accel.roundup(baselines, self.wgsx)
        return (channels, padded_baselines)

    def min_padded_noise_shape(self, baselines):
        """Minimum padded shape for noise"""
        return (accel.roundup(baselines, self.wgsx),)

    def _blocks(self, deviations):
        """Number of baseline-axis workgroups"""
        baselines = deviations.shape[1]
        blocks = accel.divup(baselines, self.wgsx)
        assert blocks * self.wgsx <= deviations.padded_shape[1]
        return blocks

    def _vt(self, deviations):
        """Number of channels processed by each thread"""
        channels = deviations.shape[0]
        vt = accel.divup(channels, self.wgsy)
        return vt

    def __call__(self, deviations, noise):
        """Perform the noise estimation

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude, indexed by channel
            then baseline. It must have a padded size at least that
            given by :meth:`min_padded_shape`.
        noise : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Output estimates, with the same shape and padding as baseline axis
            of `deviations`
        """
        (channels, baselines) = deviations.shape
        assert noise.shape[0] == baselines
        assert noise.padded_shape[0] >= self.min_padded_noise_shape(baselines)[0]
        blocks = self._blocks(deviations)
        vt = self._vt(deviations)
        self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    deviations.buffer, noise.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1]),
                    np.int32(vt)
                ],
                global_size=(blocks * self.wgsx, self.wgsy),
                local_size=(self.wgsx, self.wgsy))

class NoiseEstMADTDevice(object):
    """Device-side thresholding by median of absolute deviations. It
    should give the same results as :class:`ThresholdMADHost`, up to
    floating-point accuracy. It uses transposed (baseline-major) memory
    order, which allows an entire baseline to be efficiently loaded into
    registers.

    .. note:: There is a tradeoff in selecting the workgroup size: a large
        value gives more parallelism and reduces the register pressure, but
        increases the overhead of reduction operations.

    .. note:: This class may fail for very large numbers of channels (10k can
        definitely be supported), in which case :class:`ThresholdMADDevice` may be
        used.

    Parameters
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    max_channels : int
        Maximum number of channels. Choosing too large a value will
        reduce performance.

    Attributes
    ----------
    max_channels : int
        Maximum number of channels that can be supported
    _vt : int
        Number of elements handled by each thread
    _wgsx : int
        Number of work-items per baseline
    """
    host_class = host.NoiseEstMADHost
    transposed = True

    def __init__(self, command_queue, max_channels):
        self.command_queue = command_queue
        self.max_channels = max_channels
        self._wgsx = 256   # TODO: tune based on hardware
        self._vt = accel.divup(max_channels, self._wgsx)
        program = accel.build(command_queue.context, 'rfi/madnz_t.mako',
                {'vt': self._vt, 'wgsx': self._wgsx})
        self.kernel = program.get_kernel('madnz_t')

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        (baselines, channels) = shape
        # TODO: this is just for alignment, and should move to accel.py
        padded_channels = accel.roundup(channels, 32)
        return (baselines, padded_channels)

    def min_padded_noise_shape(self, baselines):
        """Minimum padded shape for noise"""
        return (baselines,)

    def __call__(self, deviations, noise):
        """Perform the noise estimation

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude, indexed by baseline
            then channel. It must have a padded size at least that
            given by :meth:`min_padded_shape`.
        noise : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Output estimates, with the same shape and padding as baseline axis
            of `deviations`
        """
        (baselines, channels) = deviations.shape
        assert noise.shape[0] == baselines
        assert channels <= self.max_channels
        self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    deviations.buffer, noise.buffer,
                    np.int32(channels), np.int32(deviations.padded_shape[1])
                ],
                global_size=(self._wgsx, baselines),
                local_size=(self._wgsx, 1))

class ThresholdHostFromDevice(object):
    """Wraps a device-side thresholder to present the host interface"""
    def __init__(self, real_threshold):
        self.real_threshold = real_threshold

    def __call__(self, deviations, noise):
        (channels, baselines) = deviations.shape
        transposed = self.real_threshold.transposed
        if transposed:
            deviations = deviations.T
        padded_shape = self.real_threshold.min_padded_shape(deviations.shape)
        padded_noise_shape = self.real_threshold.min_padded_noise_shape(baselines)
        # Allocate memory and copy data
        device_deviations = DeviceArray(self.real_threshold.command_queue.context,
                shape=deviations.shape, dtype=np.float32, padded_shape=padded_shape)
        device_deviations.set(self.real_threshold.command_queue, deviations)
        device_noise = DeviceArray(self.real_threshold.command_queue.context,
                shape=(baselines,), dtype=np.float32,
                padded_shape=padded_noise_shape)
        device_noise.set(self.real_threshold.command_queue, noise)
        device_flags = DeviceArray(self.real_threshold.command_queue.context,
                shape=deviations.shape, dtype=np.uint8, padded_shape=padded_shape)
        # Do computation
        self.real_threshold(device_deviations, device_noise, device_flags)
        # Copy back results
        flags = device_flags.get(self.real_threshold.command_queue)
        if transposed:
            flags = flags.T
        return flags

class ThresholdSimpleDevice(object):
    """Device-side thresholding, operating independently on each sample. It
    should give the same results as :class:`ThresholdSimpleHost`, up to
    floating-point accuracy.

    This class can operate on either transposed or non-transposed inputs,
    depending on a constructor argument.

    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    n_sigma : float
        Number of (estimated) standard deviations for the threshold
    transposed : boolean
        Whether inputs and outputs are transposed
    wgsx : int
        Number of baselines per workgroup
    wgsy : int
        Number of channels per workgroup
    flag_value : int
        Number stored in returned value to indicate RFI
    """
    host_class = host.ThresholdSimpleHost

    def __init__(self, command_queue, n_sigma, transposed, wgsx=32, wgsy=8, flag_value=1):
        self.command_queue = command_queue
        self.n_sigma = n_sigma
        self.transposed = transposed
        self.wgsx = wgsx
        self.wgsy = wgsy
        self.flag_value = flag_value
        if transposed:
            source_name = 'rfi/threshold_simple_t.mako'
            kernel_name = 'threshold_simple_t'
        else:
            source_name = 'rfi/threshold_simple.mako'
            kernel_name = 'threshold_simple'
        program = accel.build(command_queue.context, source_name,
                {'wgsx': wgsx, 'wgsy': wgsy, 'flag_value': flag_value})
        self.kernel = program.get_kernel(kernel_name)

    def min_padded_shape(self, shape):
        """Minimum padded size for inputs and outputs"""
        return (
                accel.roundup(shape[0], self.wgsy),
                accel.roundup(shape[1], self.wgsx))

    def min_padded_noise_shape(self, baselines):
        if self.transposed:
            wgs = self.wgsy
        else:
            wgs = self.wgsx
        return (accel.roundup(baselines, wgs),)

    def __call__(self, deviations, noise, flags):
        """Apply the thresholding

        Parameters
        ----------
        deviations : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Deviations from the background amplitude. It must have a padded
            size at least that given by :meth:`min_padded_shape`.
        noise : :class:`katsdpsigproc.accel.DeviceArray`, float32
            Noise estimates, with the same shape as the baseline axis of
            `deviations`, and padded shape of at least
            :meth:`min_padded_noise_shape`
        flags : :class:`katsdpsigproc.accel.DeviceArray`, uint8
            Output flags, with the same shape and padding as `deviations`
        """
        assert deviations.shape == flags.shape
        assert deviations.padded_shape == flags.padded_shape
        bl_axis = 1 - int(self.transposed)
        baselines = deviations.shape[bl_axis]
        assert noise.shape[0] == baselines
        assert noise.padded_shape[0] >= self.min_padded_noise_shape(baselines)[0]

        self.command_queue.enqueue_kernel(
                self.kernel,
                [
                    deviations.buffer, noise.buffer, flags.buffer,
                    np.int32(deviations.padded_shape[1]),
                    np.float32(self.n_sigma)
                ],
                global_size=tuple(reversed(self.min_padded_shape(deviations.shape))),
                local_size=(self.wgsx, self.wgsy))

class FlaggerDevice(object):
    """Combine device backgrounder and thresholder implementations to
    create a flagger. The thresholder may take transposed input, in which
    case this object will manage temporary buffers and the transposition
    automatically.

    Intermediate buffers are allocated when first required or when the
    sizes of inputs change. Thus, the first call may be slower than subsequent
    calls. It is also a good idea to make subsequent calls with the same size
    input data. For example, if using two subarrays (of different sizes), it
    will be more efficient to use a separate flagger object for each rather
    than multiplexing them through one flagger.

    Attributes
    ----------
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    background
        Backgrounder object
    noise_est
        Noise estimator object
    threshold
        Thresholder object
    _deviations : :class`:katsdpsigproc.accel.DeviceArray`, float32
        Deviations of the amplitude from the background
    _deviations_t : :class`:katsdpsigproc.accel.DeviceArray`, float32
        Transposed deviations
    _noise : :class:`katsdpsigproc.accel.DeviceArray`, float32
        Noise estimate for each baseline
    _flags_t : :class`:katsdpsigproc.accel.DeviceArray`, uint8
        Transposed flags
    _transpose_deviations : :class:`katsdpsigproc.accel.Transpose`
        Kernel for transposing deviations
    _transpose_flags : :class:`katsdpsigproc.accel.Transpose`
        Kernel for transposing flags
    """
    def __init__(self, background, noise_est, threshold):
        self.background = background
        self.noise_est = noise_est
        self.threshold = threshold
        assert self.background.command_queue is self.threshold.command_queue
        self.command_queue = self.background.command_queue
        self._deviations = None
        self._noise = None
        self._deviations_t = None
        self._flags_t = None
        if noise_est.transposed or threshold.transposed:
            self._transpose_deviations = Transpose(self.command_queue, 'float')
        if threshold.transposed:
            self._transpose_flags = Transpose(self.command_queue, 'unsigned char')

    def _min_padded_shape_t(self, shape_t):
        padded_shape = (0, 0)
        if self.noise_est.transposed:
            padded_shape = map(max,
                    padded_shape,
                    self.noise_est.min_padded_shape(shape_t))
        if self.threshold.transposed:
            padded_shape = map(max,
                    padded_shape,
                    self.threshold.min_padded_shape(shape_t))
        return tuple(padded_shape)

    def _min_padded_noise_shape(self, baselines):
        return tuple(map(max,
                self.noise_est.min_padded_noise_shape(baselines),
                self.threshold.min_padded_noise_shape(baselines)))

    def min_padded_shape(self, shape):
        """The minimum padding needed for the inputs and outputs of the
        flagger."""
        padded_shape = self.background.min_padded_shape(shape)
        if not self.noise_est.transposed:
            padded_shape = map(max,
                    padded_shape,
                    self.noise_est.min_padded_shape(shape))
        if not self.threshold.transposed:
            padded_shape = map(max,
                    padded_shape,
                    self.threshold.min_padded_shape(shape))
        return padded_shape

    def _ensure_array(self, ary, shape, padded_shape, dtype):
        """Allocates array if needed, returns the new value"""
        if ary is None or ary.shape != shape or ary.padded_shape != padded_shape:
            ary = DeviceArray(
                    self.command_queue.context, shape, dtype, padded_shape)
        return ary

    def __call__(self, vis, flags):
        """Perform the flagging.

        Parameters
        ----------
        vis : :class:`katsdpsigproc.accel.DeviceArray`, complex64
            The input visibilities as a 2D array, indexed
            by channel and baseline, and with padded size given by
            :meth:`min_padded_shape`.
        flags : :class:katsdpsigproc.accel.DeviceArray`, uint8
            The output flags, with the same shape and padding as `vis`.
        """
        (channels, baselines) = vis.shape
        assert vis.shape == flags.shape
        assert vis.padded_shape == flags.padded_shape
        assert np.all(np.greater_equal(
                self.min_padded_shape(vis.shape), vis.padded_shape))

        # Allocate or reallocate internal buffers if necessary
        shape_t = (vis.shape[1], vis.shape[0])
        padded_shape_t = self._min_padded_shape_t(shape_t)
        padded_noise_shape = self._min_padded_noise_shape(baselines)
        self._deviations = self._ensure_array(
                self._deviations, vis.shape, vis.padded_shape, np.float32)
        self._noise = self._ensure_array(
                self._noise, (baselines,), padded_noise_shape, np.float32)
        if self.noise_est.transposed or self.threshold.transposed:
            self._deviations_t = self._ensure_array(
                    self._deviations_t, shape_t, padded_shape_t, np.float32)
        if self.threshold.transposed:
            self._flags_t = self._ensure_array(
                    self._flags_t, shape_t, padded_shape_t, np.uint8)

        # Do computations
        self.background(vis, self._deviations)
        if self.noise_est.transposed or self.threshold.transposed:
            self._transpose_deviations(self._deviations_t, self._deviations)

        if self.noise_est.transposed:
            self.noise_est(self._deviations_t, self._noise)
        else:
            self.noise_est(self._deviations, self._noise)

        if self.threshold.transposed:
            self.threshold(self._deviations_t, self._noise, self._flags_t)
            self._transpose_flags(flags, self._flags_t)
        else:
            self.threshold(self._deviations, self._noise, flags)

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
