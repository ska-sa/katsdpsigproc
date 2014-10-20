# coding: utf-8
"""RFI flagging algorithms that run on an accelerator, using OpenCL or CUDA.

The noise estimation and thresholding functions may take data in either
channel-major or baseline-major order (the flags are emitted in the same
order). In the former case, the `transposed` member is `False`, otherwise it is
`True`. The flagger classes automatically detect this and apply a transposition
kernel at the appropriate point.
"""

from __future__ import division
from .. import accel
from .. import tune
from .. import transpose
from ..accel import DeviceArray, LinenoLexer
import numpy as np
from . import host

class BackgroundHostFromDevice(object):
    """Wraps a device-side background template to present the host interface."""
    def __init__(self, template, command_queue):
        self.template = template
        self.command_queue = command_queue

    def __call__(self, vis):
        (channels, baselines) = vis.shape
        fn = self.template.instantiate(self.command_queue, channels, baselines)
        # Trigger allocations
        fn.check_all_bound()
        fn.slots['vis'].buffer.set(self.command_queue, vis)
        # Do the computation
        fn()
        return fn.slots['deviations'].buffer.get(self.command_queue)

class BackgroundMedianFilterDeviceTemplate(object):
    """Device algorithm that applies a median filter to each baseline
    (in amplitude). It is the same algorithm as
    :class:`host.BackgroundMedianFilterHost`, but may give slightly different
    results due to rounding errors when computing complex magnitude.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    width : int
        The kernel width (must be odd)
    tune : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are
        - wgs: number of work-items per baseline
        - csplit: approximate number of workitems for each channel
    """

    host_class = host.BackgroundMedianFilterHost

    def __init__(self, context, width, tune=None):
        if tune is None:
            tune = self.autotune(context, width)
        self.context = context
        self.width = width
        self.wgs = tune['wgs']
        self.csplit = tune['csplit']
        program = accel.build(context, 'rfi/background_median_filter.mako',
                {'width': width, 'wgs': self.wgs})
        self.kernel = program.get_kernel('background_median_filter')

    @classmethod
    @tune.autotuner
    def autotune(cls, context, width):
        queue = context.create_tuning_command_queue()
        # Note: baselines must be a multiple of any tested workgroup size
        channels = 4096
        baselines = 8192
        shape = (channels, baselines)
        vis = DeviceArray(context, shape, np.complex64)
        deviations = DeviceArray(context, shape, np.float32)
        # Initialize with Gaussian random values
        rs = np.random.RandomState(seed=1)
        vis_host = (rs.standard_normal(shape) + rs.standard_normal(shape) * 1j).astype(np.complex64)
        vis.set(queue, vis_host)
        def generate(**tune):
            fn = cls(context, width, tune).instantiate(queue, channels, baselines)
            fn.bind(vis=vis, deviations=deviations)
            def measure(iters):
                queue.start_tuning()
                for i in range(iters):
                    fn()
                return queue.stop_tuning() / iters
            return measure
        return tune.autotune(generate, wgs=[32, 64, 128, 256, 512], csplit=[1, 2, 4, 8, 16])

    def instantiate(self, command_queue, channels, baselines):
        """Create an instance. See :class:`BackgroundMedianFilterDevice`."""
        return BackgroundMedianFilterDevice(self, command_queue, channels, baselines)

class BackgroundMedianFilterDevice(accel.Operation):
    """Concrete instance of :class:`BackgroundMedianFilterDeviceTemplate`.

    Parameters
    ----------
    template : :class:`BackgroundMedianFilterDevice`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    channels, baselines : int
        Shape of the visibilities array

    Slots
    -----
    vis : channels x baselines, complex64
        Input visibilities
    deviations : channels x baselines, float32
        Output deviations from the background
    """
    def __init__(self, template, command_queue, channels, baselines):
        super(BackgroundMedianFilterDevice, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.slots['vis'] = accel.IOSlot(
                (channels, baselines), np.complex64,
                (1, self.template.wgs))
        self.slots['deviations'] = accel.IOSlot(
                (channels, baselines), np.float32,
                (1, self.template.wgs))

    def __call__(self, **kwargs):
        """Perform the backgrounding.

        Parameters
        ----------
        **kwargs : :class:`katsdpsigproc.accel.DeviceArray`
            Passed to :meth:`bind`
        """
        self.bind(**kwargs)
        self.check_all_bound()
        VT = accel.divup(self.channels, self.template.csplit)
        xblocks = accel.divup(self.baselines, self.template.wgs)
        yblocks = accel.divup(self.channels, VT)

        vis = self.slots['vis'].buffer
        deviations = self.slots['deviations'].buffer
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    vis.buffer,
                    deviations.buffer,
                    np.int32(self.channels),
                    np.int32(vis.padded_shape[1]),
                    np.int32(deviations.padded_shape[1]),
                    np.int32(VT)
                ],
                global_size=(xblocks * self.template.wgs, yblocks),
                local_size=(self.template.wgs, 1))

class NoiseEstHostFromDevice(object):
    """Wraps a device-side noise estimator template to present the host interface"""
    def __init__(self, template, command_queue):
        self.template = template
        self.command_queue = command_queue

    def __call__(self, deviations):
        (channels, baselines) = deviations.shape
        transposed = self.template.transposed
        if transposed:
            deviations = deviations.T
        fn = self.template.instantiate(self.command_queue, channels, baselines)
        # Allocate and populate memory
        fn.check_all_bound()
        fn.slots['deviations'].buffer.set(self.command_queue, deviations)
        # Perform computation
        fn()
        # Copy back results
        noise = fn.slots['noise'].buffer.get(self.command_queue)
        return noise

class NoiseEstMADDeviceTemplate(object):
    """Estimate noise using the median of non-zero absolute deviations.

    In most cases NoiseEstMADTDeviceTemplate is more efficient.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    wgsx : int, optional
        Number of baselines per workgroup
    wgsy : int, optional
        Number of channels per workgroup
    """

    host_class = host.NoiseEstMADHost
    transposed = False

    def __init__(self, context, wgsx=32, wgsy=8):
        self.context = context
        self.wgsx = wgsx
        self.wgsy = wgsy
        program = accel.build(context, 'rfi/madnz.mako',
                {'wgsx': wgsx, 'wgsy': wgsy})
        self.kernel = program.get_kernel('madnz')

    def instantiate(self, command_queue, channels, baselines):
        """Create an instance. See :class:`NoiseEstMADDevice`."""
        return NoiseEstMADDevice(self, command_queue, channels, baselines)

class NoiseEstMADDevice(accel.Operation):
    """Concrete instantiation of :class:`NoiseEstMADDeviceTemplate`.

    Parameters
    ----------
    template : :class:`NoiseEstMADDeviceTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array

    Slots
    -----
    deviations : channels x baselines, float32
        Input deviations from the background, computed by a backgrounder
    noise : baselines, float32
        Output per-baseline noise estimate
    """

    transposed = False

    def __init__(self, template, command_queue, channels, baselines):
        super(NoiseEstMADDevice, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.slots['noise'] = accel.IOSlot((baselines,), np.float32, (self.template.wgsx,))
        self.slots['deviations'] = accel.IOSlot(
                (channels, baselines), np.float32, (1, self.template.wgsx))

    def __call__(self, **kwargs):
        """Perform the noise estimation.

        Parameters
        ----------
        **kwargs : :class:`katsdpsigproc.accel.DeviceArray`
            Passed to :meth:`bind`
        """
        self.bind(**kwargs)
        self.check_all_bound()
        blocks = accel.divup(self.baselines, self.template.wgsx)
        vt = accel.divup(self.channels, self.template.wgsy)
        deviations = self.slots['deviations'].buffer
        noise = self.slots['noise'].buffer
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    deviations.buffer, noise.buffer,
                    np.int32(self.channels), np.int32(deviations.padded_shape[1]),
                    np.int32(vt)
                ],
                global_size=(blocks * self.template.wgsx, self.template.wgsy),
                local_size=(self.template.wgsx, self.template.wgsy))

class NoiseEstMADTDeviceTemplate(object):
    """Device-side noise estimation by median of absolute deviations. It
    should give the same results as :class:`NoiseEstMADHost`, up to
    floating-point accuracy. It uses transposed (baseline-major) memory
    order, which allows an entire baseline to be efficiently loaded into
    registers.

    .. note:: There is a tradeoff in selecting the workgroup size: a large
        value gives more parallelism and reduces the register pressure, but
        increases the overhead of reduction operations.

    .. note:: This class may fail for very large numbers of channels (10k can
        definitely be supported), in which case :class:`NoiseEstMADDevice` may be
        used.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    max_channels : int
        Maximum number of channels. Choosing too large a value will
        reduce performance.
    tune : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are
        - wgsx: number of work-items per baseline
    """

    host_class = host.NoiseEstMADHost
    transposed = True

    def __init__(self, context, max_channels, tune=None):
        self.context = context
        self.max_channels = max_channels
        if tune is None:
            tune = self.autotune(context, max_channels)
        self.wgsx = tune['wgsx']
        vt = accel.divup(max_channels, self.wgsx)
        program = accel.build(context, 'rfi/madnz_t.mako',
                {'vt': vt, 'wgsx': self.wgsx})
        self.kernel = program.get_kernel('madnz_t')

    @classmethod
    @tune.autotuner
    def autotune(cls, context, max_channels):
        queue = context.create_tuning_command_queue()
        baselines = 128
        rs = np.random.RandomState(seed=1)
        host_deviations = rs.uniform(size=(baselines, max_channels)).astype(np.float32)
        def generate(**tune):
            # Very large values of VT cause the AMD compiler to choke and segfault
            if max_channels > 256 * tune['wgsx']:
                raise ValueError('wgsx is too small')
            fn = cls(context, max_channels, tune).instantiate(queue, max_channels, baselines)
            noise = fn.slots['noise'].allocate(context)
            deviations = fn.slots['deviations'].allocate(context)
            deviations.set(queue, host_deviations)
            def measure(iters):
                queue.start_tuning()
                for i in range(iters):
                    fn()
                return queue.stop_tuning() / iters
            return measure
        return tune.autotune(generate, wgsx=[32, 64, 128, 256, 512, 1024])

    def instantiate(self, command_queue, channels, baselines):
        """Create an instance. See :class:`NoiseEstMADTDevice`."""
        return NoiseEstMADTDevice(self, command_queue, channels, baselines)

class NoiseEstMADTDevice(accel.Operation):
    """Concrete instance of :class:`NoiseEstMADTDeviceTemplate`.

    Parameters
    ----------
    template : :class:`NoiseEstMADTDeviceTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array

    Slots
    ----------
    deviations : baselines x channels, float32
        Input deviations from the background, computed by a backgrounder
    noise : baselines, float32
        Output per-baseline noise estimate
    """
    transposed = True

    def __init__(self, template, command_queue, channels, baselines):
        super(NoiseEstMADTDevice, self).__init__(command_queue)
        self.template = template
        if channels > self.template.max_channels:
            raise ValueError('channels exceeds max_channels')
        self.channels = channels
        self.baselines = baselines
        self.slots['noise'] = accel.IOSlot((baselines,), np.float32)
        self.slots['deviations'] = accel.IOSlot((baselines, channels), np.float32)

    def __call__(self, **kwargs):
        """Perform the noise estimation.

        Parameters
        ----------
        **kwargs : :class:`katsdpsigproc.accel.DeviceArray`
            Passed to :meth:`bind`
        """
        self.bind(**kwargs)
        self.check_all_bound()
        deviations = self.slots['deviations'].buffer
        noise = self.slots['noise'].buffer
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    deviations.buffer, noise.buffer,
                    np.int32(self.channels), np.int32(deviations.padded_shape[1])
                ],
                global_size=(self.template.wgsx, self.baselines),
                local_size=(self.template.wgsx, 1))

class ThresholdHostFromDevice(object):
    """Wraps a device-side thresholder template to present the host interface"""
    def __init__(self, template, command_queue):
        self.template = template
        self.command_queue = command_queue

    def __call__(self, deviations, noise):
        (channels, baselines) = deviations.shape
        transposed = self.template.transposed
        if transposed:
            deviations = deviations.T

        fn = self.template.instantiate(self.command_queue, channels, baselines)
        # Allocate memory and copy data
        fn.check_all_bound()
        fn.slots['deviations'].buffer.set(self.command_queue, deviations)
        fn.slots['noise'].buffer.set(self.command_queue, noise)
        # Do computation
        fn()
        # Copy back results
        flags = fn.slots['flags'].buffer.get(self.command_queue)
        if transposed:
            flags = flags.T
        return flags

class ThresholdSimpleDeviceTemplate(object):
    """Device-side thresholding, operating independently on each sample. It
    should give the same results as :class:`ThresholdSimpleHost`, up to
    floating-point accuracy.

    This class can operate on either transposed or non-transposed inputs,
    depending on a constructor argument.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
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

    def __init__(self, context, n_sigma, transposed, wgsx=32, wgsy=8, flag_value=1):
        self.context = context
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
        program = accel.build(context, source_name,
                {'wgsx': wgsx, 'wgsy': wgsy, 'flag_value': flag_value})
        self.kernel = program.get_kernel(kernel_name)

    def instantiate(self, command_queue, channels, baselines):
        """Create an instance. See :class:`ThresholdSimpleDevice`."""
        return ThresholdSimpleDevice(self, command_queue, channels, baselines)

class ThresholdSimpleDevice(accel.Operation):
    """Concrete instance of :class:`ThresholdSimpleDeviceTemplate`.

    Parameters
    ----------
    template : :class:`ThresholdSimpleDeviceTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array

    Slots
    -----
    deviations : channels x baselines (or transposed), float32
        Input deviations from the background
    noise : baselines, float32
        Noise estimates per baseline
    flags : channels x baselines (or transposed), uint8
        Output flags
    """

    def __init__(self, template, command_queue, channels, baselines):
        super(ThresholdSimpleDevice, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.transposed = template.transposed

        shape = (baselines, channels) if self.transposed else (channels, baselines)
        alignment = (self.template.wgsy, self.template.wgsx)
        noise_alignment = (self.template.wgsy if self.transposed else self.template.wgsx,)

        self.slots['deviations'] = accel.IOSlot(shape, np.float32, alignment)
        self.slots['noise'] = accel.IOSlot((baselines,), np.float32, noise_alignment)
        self.slots['flags'] = accel.IOSlot(shape, np.uint8, alignment)

    def __call__(self, **kwargs):
        """Apply the thresholding

        Parameters
        ----------
        **kwargs : :class:`katsdpsigproc.accel.DeviceArray`
            Passed to :meth:`bind`
        """
        self.bind(**kwargs)
        self.check_all_bound()
        deviations = self.slots['deviations'].buffer
        noise = self.slots['noise'].buffer
        flags = self.slots['flags'].buffer

        global_x = accel.roundup(deviations.shape[1], self.template.wgsx)
        global_y = accel.roundup(deviations.shape[0], self.template.wgsy)
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    deviations.buffer, noise.buffer, flags.buffer,
                    np.int32(deviations.padded_shape[1]),
                    np.int32(flags.padded_shape[1]),
                    np.float32(self.template.n_sigma)
                ],
                global_size=(global_x, global_y),
                local_size=(self.template.wgsx, self.template.wgsy))

class ThresholdSumDeviceTemplate(object):
    """A device version of :class:`katsdpsigproc.rfi.host.ThresholdSumHost.
    It uses transposed data. Performance will be best with a large work
    group size, because of the stencil-like nature of the computation.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    n_sigma : float
        Number of (estimated) standard deviations for the threshold
    n_windows : int
        Number of window sizes to use
    threshold_falloff : float
        Controls rate at which thresholds decrease (ρ in Offringa 2010)
    wgs : int
        Number of work items to use per work group
    vt : int
        Number of elements to process in each work item
    flag_value : int
        Number stored in returned value to indicate RFI
    """

    host_class = host.ThresholdSumHost
    transposed = True

    def __init__(self, context, n_sigma, n_windows=4, threshold_falloff=1.2,
            flag_value=1, tune=None):
        if tune is None:
            tune = self.autotune(context, n_windows)
        wgs = tune['wgs']
        vt = tune['vt']
        edge_size = 2 ** n_windows - n_windows - 1
        self.chunk = wgs * vt - 2 * edge_size
        assert self.chunk > 0
        self.context = context
        self.n_windows = n_windows
        self.n_sigma = [np.float32(n_sigma * pow(threshold_falloff, -i)) for i in range(n_windows)]
        self.wgs = wgs
        self.vt = vt
        self.flag_value = flag_value
        program = accel.build(context, 'rfi/threshold_sum.mako',
                {'wgs': self.wgs,
                 'vt': self.vt,
                 'windows' : self.n_windows,
                 'flag_value': self.flag_value})
        self.kernel = program.get_kernel('threshold_sum')

    @classmethod
    @tune.autotuner
    def autotune(cls, context, n_windows):
        queue = context.create_tuning_command_queue()
        channels = 4096
        baselines = 128
        shape = (baselines, channels)
        rs = np.random.RandomState(seed=1)
        deviations = DeviceArray(context, shape, dtype=np.float32)
        deviations.set(queue, rs.uniform(size=deviations.shape).astype(np.float32))
        noise = DeviceArray(context, (baselines,), dtype=np.float32)
        noise.set(queue, rs.uniform(high=0.1, size=noise.shape).astype(np.float32))
        flags = DeviceArray(context, shape, dtype=np.uint8)
        def generate(**tune):
            template = cls(context, 11.0, n_windows=n_windows, tune=tune)
            fn = template.instantiate(queue, channels, baselines)
            def measure(iters):
                queue.start_tuning()
                for i in range(iters):
                    fn(deviations=deviations, noise=noise, flags=flags)
                return queue.stop_tuning() / iters
            return measure
        return tune.autotune(generate,
                wgs=[32, 64, 128, 256, 512],
                vt=[1, 2, 3, 4, 8, 16])

    def instantiate(self, command_queue, channels, baselines):
        """Create an instance. See :class:`ThresholdSumDevice`."""
        return ThresholdSumDevice(self, command_queue, channels, baselines)

class ThresholdSumDevice(accel.Operation):
    """Concrete instance of :class:`ThresholdSumDeviceTemplate`.

    Parameters
    ----------
    template : :class:`ThresholSumDeviceTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array

    Slots
    -----
    deviations : baselines x channels, float32
        Input deviations from the background
    noise : baselines, float32
        Noise estimates per baseline
    flags : baselines x channels, uint8
        Output flags
    """
    host_class = host.ThresholdSumHost
    transposed = True

    def __init__(self, template, command_queue, channels, baselines):
        super(ThresholdSumDevice, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.slots['deviations'] = accel.IOSlot((baselines, channels), np.float32)
        self.slots['noise'] = accel.IOSlot((baselines,), np.float32)
        self.slots['flags'] = accel.IOSlot((baselines, channels), np.uint8)

    def __call__(self, **kwargs):
        """Apply the thresholding

        Parameters
        ----------
        **kwargs : :class:`katsdpsigproc.accel.DeviceArray`
            Passed to :meth:`bind`
        """
        self.bind(**kwargs)
        self.check_all_bound()
        deviations = self.slots['deviations'].buffer
        noise = self.slots['noise'].buffer
        flags = self.slots['flags'].buffer

        blocks = accel.divup(self.channels, self.template.chunk)
        args = [deviations.buffer, noise.buffer, flags.buffer,
                np.int32(self.channels), np.int32(deviations.padded_shape[1])]
        args.extend(self.template.n_sigma)
        self.command_queue.enqueue_kernel(
                self.template.kernel, args,
                global_size = (blocks * self.template.wgs, self.baselines),
                local_size = (self.template.wgs, 1))

class FlaggerDeviceTemplate(object):
    """Combine device backgrounder, noise estimation and thresholder
    implementations to create a flagger. The thresholder and/or noise
    estimation may take transposed input, in which case this object will manage
    temporary buffers and the transposition automatically.

    Parameters
    ----------
    background, noise_est, threshold : template types
        The templates for the individual steps
    """
    def __init__(self, background, noise_est, threshold):
        self.background = background
        self.noise_est = noise_est
        self.threshold = threshold
        context = self.background.context
        # Check that all operations work in the same context
        assert self.noise_est.context is context
        assert self.threshold.context is context

        # Create transposition operations if needed
        if noise_est.transposed or threshold.transposed:
            self.transpose_deviations = transpose.TransposeTemplate(context, np.float32, 'float')
        else:
            self.transpose_deviations = None
        if threshold.transposed:
            self.transpose_flags = transpose.TransposeTemplate(context, np.uint8, 'unsigned char')
        else:
            self.transpose_flags = None

    def instantiate(self, command_queue, channels, baselines):
        """Create an instance. See :class:`FlaggerDevice`."""
        return FlaggerDevice(self, command_queue, channels, baselines)

class FlaggerDevice(accel.Operation):
    """Concrete instance of :class:`FlaggerDeviceTemplate`.

    Temporary buffers are presented as slots, which allows them to either
    be set by the user or allocated automatically on first use.

    Parameters
    ----------
    template : :class:`BackgroundMedianFilterDevice`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    channels, baselines : int
        Shape of the visibilities array

    Slots
    -----
    vis : channels x baselines, complex64
        Input visibilities
    deviations : channels x baselines, float32
        Temporary, deviations from the background
    deviations_t : baselines x channels, float32, optional
        Transpose of `deviations`
    noise : baselines, float32
        Estimate of per-baseline noise
    flags_t : baselines x channels, uint8, optional
        Transpose of `flags`
    flags : channels x baselines, uint8
        Output flags
    """
    def __init__(self, template, command_queue, channels, baselines):
        super(FlaggerDevice, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.background = self.template.background.instantiate(command_queue, channels, baselines)
        self.noise_est = self.template.noise_est.instantiate(command_queue, channels, baselines)
        self.threshold = self.template.threshold.instantiate(command_queue, channels, baselines)

        self.slots['vis'] = accel.CompoundIOSlot([self.background.slots['vis']])
        # Slots contributing to the non-transposed and transposed versions
        deviations_slots = ([], [])
        deviations_slots[0].append(self.background.slots['deviations'])
        deviations_slots[self.noise_est.transposed].append(self.noise_est.slots['deviations'])
        deviations_slots[self.threshold.transposed].append(self.threshold.slots['deviations'])
        if deviations_slots[1]:
            # Transposition is needed
            self.transpose_deviations = self.template.transpose_deviations.instantiate(
                    command_queue, (channels, baselines))
            deviations_slots[0].append(self.transpose_deviations.slots['src'])
            deviations_slots[1].append(self.transpose_deviations.slots['dest'])
            self.slots['deviations_t'] = accel.CompoundIOSlot(deviations_slots[1])
        else:
            self.transpose_deviations = None
        self.slots['deviations'] = accel.CompoundIOSlot(deviations_slots[0])

        self.slots['noise'] = accel.CompoundIOSlot([
            self.noise_est.slots['noise'],
            self.threshold.slots['noise']])

        # Similar to deviations, but for flags
        flags_slots = ([], [])
        flags_slots[self.threshold.transposed].append(self.threshold.slots['flags'])
        if flags_slots[1]:
            # Transposition is needed
            self.transpose_flags = self.template.transpose_flags.instantiate(
                    command_queue, (baselines, channels))
            flags_slots[1].append(self.transpose_flags.slots['src'])
            flags_slots[0].append(self.transpose_flags.slots['dest'])
            self.slots['flags_t'] = accel.CompoundIOSlot(flags_slots[1])
        else:
            self.transpose_flags = None
        self.slots['flags'] = accel.CompoundIOSlot(flags_slots[0])

    def __call__(self, **kwargs):
        """Perform the flagging

        Parameters
        ----------
        **kwargs : :class:`katsdpsigproc.accel.DeviceArray`
            Passed to :meth:`bind`
        """
        self.bind(**kwargs)
        self.check_all_bound()
        # Do computations
        self.background()
        if self.transpose_deviations is not None:
            self.transpose_deviations()
        self.noise_est()
        self.threshold()
        if self.transpose_flags is not None:
            self.transpose_flags()

class FlaggerHostFromDevice(object):
    """Wrapper that makes a :class:`FlaggerDeviceTemplate` present the
    interface of :class:`FlaggerHost`. This is intended only for ease of
    use. It is not efficient, because it allocates and frees memory on
    every call.

    Parameters
    ----------
    template : :class:`FlaggerDeviceTemplate`
        Operation template
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    """
    def __init__(self, template, command_queue):
        self.template = template
        self.command_queue = command_queue

    def __call__(self, vis):
        (channels, baselines) = vis.shape
        fn = self.template.instantiate(self.command_queue, channels, baselines)
        fn.check_all_bound()
        fn.slots['vis'].buffer.set(self.command_queue, vis)
        fn()
        return fn.slots['flags'].buffer.get(self.command_queue)
