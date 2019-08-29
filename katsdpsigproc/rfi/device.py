# coding: utf-8
"""RFI flagging algorithms that run on an accelerator, using OpenCL or CUDA.

The noise estimation and thresholding functions may take data in either
channel-major or baseline-major order (the flags are emitted in the same
order). In the former case, the `transposed` member is `False`, otherwise it is
`True`. The flagger classes automatically detect this and apply a transposition
kernel at the appropriate point.
"""

import enum

import numpy as np

from .. import accel
from .. import tune
from .. import transpose
from ..accel import DeviceArray
from . import host


class BackgroundFlags(enum.Enum):
    NONE = 0
    CHANNEL = 1
    FULL = 2

    def __bool__(self):
        return self != BackgroundFlags.NONE


class BackgroundHostFromDevice(object):
    """Wraps a device-side background template to present the host interface."""
    def __init__(self, template, command_queue):
        self.template = template
        self.command_queue = command_queue

    def __call__(self, vis, flags=None):
        if flags is not None and not self.template.use_flags:
            raise TypeError("flags were provided but not included in the template")
        if flags is None and self.template.use_flags:
            raise TypeError("flags were expected but not provided")
        (channels, baselines) = vis.shape
        fn = self.template.instantiate(self.command_queue, channels, baselines)
        # Trigger allocations
        fn.ensure_all_bound()
        fn.buffer('vis').set(self.command_queue, vis)
        if flags is not None:
            fn.buffer('flags').set(self.command_queue, flags)
        # Do the computation
        fn()
        return fn.buffer('deviations').get(self.command_queue)


class BackgroundMedianFilterDeviceTemplate(object):
    """Device algorithm that applies a median filter to each baseline
    (in amplitude). It is the same algorithm as
    :class:`host.BackgroundMedianFilterHost`, but may give slightly different
    results due to rounding errors when computing complex magnitude.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    width : int
        The kernel width (must be odd)
    is_amplitude : boolean
        If ``True``, the inputs are amplitudes rather than complex visibilities
    use_flags : :class:`BackgroundFlags` or bool
        Specify that flags are taken as input to indicate bad data that should
        not contribute to the backgrounding. The legal values are

        - ``FULL``: takes flags with same shape as visibilities
        - ``CHANNEL``: takes a single flag per channel
        - ``NONE``: do not take flags as input

        For backwards compatibility, ``True`` is an alias for ``CHANNEL`` and
        ``False`` is an alias for ``NONE``.
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgs: number of work-items per baseline
        - csplit: approximate number of workitems for each channel
    """

    host_class = host.BackgroundMedianFilterHost
    autotune_version = 4

    def __init__(self, context, width, is_amplitude=False,
                 use_flags=BackgroundFlags.NONE, tuning=None):
        if tuning is None:
            tuning = self.autotune(context, width, is_amplitude, use_flags)
        self.context = context
        self.width = width
        self.is_amplitude = is_amplitude
        if use_flags is True:
            use_flags = BackgroundFlags.CHANNEL
        elif use_flags is False:
            use_flags = BackgroundFlags.NONE
        if not isinstance(use_flags, BackgroundFlags):
            raise TypeError('use_flags must be an instance of BackgroundFlags or bool')
        self.use_flags = use_flags
        self.wgs = tuning['wgs']
        self.csplit = tuning['csplit']
        self.program = accel.build(
            context, 'rfi/background_median_filter.mako',
            {
                'width': width, 'wgs': self.wgs,
                'is_amplitude': is_amplitude,
                'BackgroundFlags': BackgroundFlags,
                'use_flags': use_flags
            })

    @classmethod
    @tune.autotuner(test={'wgs': 128, 'csplit': 4})
    def autotune(cls, context, width, is_amplitude, use_flags):
        queue = context.create_tuning_command_queue()
        # Note: baselines must be a multiple of any tested workgroup size
        channels = 4096
        baselines = 8192
        shape = (channels, baselines)
        vis_type = np.float32 if is_amplitude else np.complex64
        vis = DeviceArray(context, shape, vis_type)
        deviations = DeviceArray(context, shape, np.float32)
        if use_flags == BackgroundFlags.CHANNEL:
            flags = DeviceArray(context, (channels,), np.uint8)
        elif use_flags == BackgroundFlags.FULL:
            flags = DeviceArray(context, shape, np.uint8)
        # Initialize with Gaussian random values
        rs = np.random.RandomState(seed=1)
        if is_amplitude:
            vis_host = rs.rayleigh(size=shape).astype(np.float32)
        else:
            vis_host = rs.standard_normal(shape) + rs.standard_normal(shape) * 1j
            vis_host = vis_host.astype(np.complex64)
        vis.set(queue, vis_host)
        if use_flags:
            flags.set(queue, (rs.uniform(size=flags.shape) < 0.0001).astype(np.uint8))

        def generate(**tuning):
            fn = cls(context, width, is_amplitude, use_flags, tuning).instantiate(
                queue, channels, baselines)
            fn.bind(vis=vis, deviations=deviations)
            if use_flags:
                fn.bind(flags=flags)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate, wgs=[32, 64, 128, 256, 512], csplit=[1, 2, 4, 8, 16])

    def instantiate(self, *args, **kwargs):
        """Create an instance. See :class:`BackgroundMedianFilterDevice`."""
        return BackgroundMedianFilterDevice(self, *args, **kwargs)


class BackgroundMedianFilterDevice(accel.Operation):
    """Concrete instance of :class:`BackgroundMedianFilterDeviceTemplate`.

    .. rubric:: Slots

    **vis** : channels × baselines, float32 or complex64
        Input visibilities, or their amplitudes if `template.is_amplitude` is true
    **flags** : channels × baselines or channels, float32
        Input flags (only present if `template.use_flags` is used)
    **deviations** : channels × baselines, float32
        Output deviations from the background

    Parameters
    ----------
    template : :class:`BackgroundMedianFilterDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    channels, baselines : int
        Shape of the visibilities array
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, channels, baselines, allocator=None):
        super(BackgroundMedianFilterDevice, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('background_median_filter')
        self.channels = channels
        self.baselines = baselines
        vis_type = np.float32 if template.is_amplitude else np.complex64
        dims = (channels,
                accel.Dimension(baselines, self.template.wgs))
        self.slots['vis'] = accel.IOSlot(dims, vis_type)
        self.slots['deviations'] = accel.IOSlot(dims, np.float32)
        if template.use_flags == BackgroundFlags.FULL:
            self.slots['flags'] = accel.IOSlot(dims, np.uint8)
        elif template.use_flags == BackgroundFlags.CHANNEL:
            self.slots['flags'] = accel.IOSlot((channels,), np.uint8)

    def _run(self):
        VT = accel.divup(self.channels, self.template.csplit)
        xblocks = accel.divup(self.baselines, self.template.wgs)
        yblocks = accel.divup(self.channels, VT)

        buffers = [self.buffer('vis'), self.buffer('deviations')]
        if self.template.use_flags:
            buffers.append(self.buffer('flags'))
        stride = buffers[0].padded_shape[1]
        self.command_queue.enqueue_kernel(
            self.kernel,
            [b.buffer for b in buffers] + [
                np.int32(self.channels),
                np.int32(stride),
                np.int32(VT)
            ],
            global_size=(xblocks * self.template.wgs, yblocks),
            local_size=(self.template.wgs, 1))

    def parameters(self):
        return {
            'width': self.template.width,
            'use_flags': self.template.use_flags.name,
            'channels': self.channels,
            'baselines': self.baselines
        }


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
        fn.ensure_all_bound()
        fn.buffer('deviations').set(self.command_queue, deviations)
        # Perform computation
        fn()
        # Copy back results
        noise = fn.buffer('noise').get(self.command_queue)
        return noise


class NoiseEstMADDeviceTemplate(object):
    """Estimate noise using the median of non-zero absolute deviations.

    In most cases NoiseEstMADTDeviceTemplate is more efficient.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx: number of baselines per workgroup
        - wgsy: number of channels per workgroup
    """

    host_class = host.NoiseEstMADHost
    transposed = False

    def __init__(self, context, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.context = context
        self.wgsx = tuning['wgsx']
        self.wgsy = tuning['wgsy']
        self.program = accel.build(context, 'rfi/madnz.mako',
                                   {'wgsx': self.wgsx, 'wgsy': self.wgsy})

    @classmethod
    @tune.autotuner(test={'wgsx': 32, 'wgsy': 8})
    def autotune(cls, context):
        # TODO: do real autotuning
        return {'wgsx': 32, 'wgsy': 8}

    def instantiate(self, *args, **kwargs):
        """Create an instance. See :class:`NoiseEstMADDevice`."""
        return NoiseEstMADDevice(self, *args, **kwargs)


class NoiseEstMADDevice(accel.Operation):
    """Concrete instantiation of :class:`NoiseEstMADDeviceTemplate`.

    .. rubric:: Slots

    **deviations** : channels × baselines, float32
        Input deviations from the background, computed by a backgrounder
    **noise** : baselines, float32
        Output per-baseline noise estimate

    Parameters
    ----------
    template : :class:`NoiseEstMADDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    transposed = False

    def __init__(self, template, command_queue, channels, baselines, allocator=None):
        super(NoiseEstMADDevice, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('madnz')
        self.channels = channels
        self.baselines = baselines
        baselines_dim = accel.Dimension(baselines, self.template.wgsx)
        self.slots['noise'] = accel.IOSlot((baselines_dim,), np.float32)
        self.slots['deviations'] = accel.IOSlot((channels, baselines_dim), np.float32)

    def _run(self):
        blocks = accel.divup(self.baselines, self.template.wgsx)
        vt = accel.divup(self.channels, self.template.wgsy)
        deviations = self.buffer('deviations')
        noise = self.buffer('noise')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                deviations.buffer, noise.buffer,
                np.int32(self.channels), np.int32(deviations.padded_shape[1]),
                np.int32(vt)
            ],
            global_size=(blocks * self.template.wgsx, self.template.wgsy),
            local_size=(self.template.wgsx, self.template.wgsy))

    def parameters(self):
        return {
            'channels': self.channels,
            'baselines': self.baselines
        }


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
    context : |Context|
        Context for which kernels will be compiled
    max_channels : int
        Maximum number of channels. Choosing too large a value will
        reduce performance.
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx: number of work-items per baseline
    """

    host_class = host.NoiseEstMADHost
    transposed = True

    def __init__(self, context, max_channels, tuning=None):
        self.context = context
        self.max_channels = max_channels
        if tuning is None:
            tuning = self.autotune(context, max_channels)
        self.wgsx = tuning['wgsx']
        vt = accel.divup(max_channels, self.wgsx)
        self.program = accel.build(context, 'rfi/madnz_t.mako', {'vt': vt, 'wgsx': self.wgsx})

    @classmethod
    @tune.autotuner(test={'wgsx': 128})
    def autotune(cls, context, max_channels):
        queue = context.create_tuning_command_queue()
        baselines = 128
        rs = np.random.RandomState(seed=1)
        host_deviations = rs.uniform(size=(baselines, max_channels)).astype(np.float32)

        def generate(**tuning):
            # Very large values of VT cause the AMD compiler to choke and segfault
            if max_channels > 256 * tuning['wgsx']:
                raise ValueError('wgsx is too small')
            fn = cls(context, max_channels, tuning).instantiate(queue, max_channels, baselines)
            fn.slots['noise'].allocate(fn.allocator)
            deviations = fn.slots['deviations'].allocate(fn.allocator)
            deviations.set(queue, host_deviations)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate, wgsx=[32, 64, 128, 256, 512, 1024])

    def instantiate(self, *args, **kwargs):
        """Create an instance. See :class:`NoiseEstMADTDevice`."""
        return NoiseEstMADTDevice(self, *args, **kwargs)


class NoiseEstMADTDevice(accel.Operation):
    """Concrete instance of :class:`NoiseEstMADTDeviceTemplate`.

    .. rubric:: Slots

    **deviations** : baselines × channels, float32
        Input deviations from the background, computed by a backgrounder
    **noise** : baselines, float32
        Output per-baseline noise estimate

    Parameters
    ----------
    template : :class:`NoiseEstMADTDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    transposed = True

    def __init__(self, template, command_queue, channels, baselines, allocator=None):
        super(NoiseEstMADTDevice, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('madnz_t')
        if channels > self.template.max_channels:
            raise ValueError('channels exceeds max_channels')
        self.channels = channels
        self.baselines = baselines
        self.slots['noise'] = accel.IOSlot((baselines,), np.float32)
        self.slots['deviations'] = accel.IOSlot((baselines, channels), np.float32)

    def _run(self):
        deviations = self.buffer('deviations')
        noise = self.buffer('noise')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                deviations.buffer, noise.buffer,
                np.int32(self.channels), np.int32(deviations.padded_shape[1])
            ],
            global_size=(self.template.wgsx, self.baselines),
            local_size=(self.template.wgsx, 1))

    def parameters(self):
        return {
            'max_channels': self.template.max_channels,
            'baselines': self.baselines,
            'channels': self.channels
        }


class ThresholdHostFromDevice(object):
    """Wraps a device-side thresholder template to present the host interface"""
    def __init__(self, template, command_queue, *args, **kwargs):
        self.template = template
        self.command_queue = command_queue
        self.args = args
        self.kwargs = kwargs

    def __call__(self, deviations, noise):
        (channels, baselines) = deviations.shape
        transposed = self.template.transposed
        if transposed:
            deviations = deviations.T

        fn = self.template.instantiate(self.command_queue, channels, baselines,
                                       *self.args, **self.kwargs)
        # Allocate memory and copy data
        fn.ensure_all_bound()
        fn.buffer('deviations').set(self.command_queue, deviations)
        fn.buffer('noise').set(self.command_queue, noise)
        # Do computation
        fn()
        # Copy back results
        flags = fn.buffer('flags').get(self.command_queue)
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
    context : |Context|
        Context for which kernels will be compiled
    transposed : boolean
        Whether inputs and outputs are transposed
    flag_value : int
        Number stored in returned value to indicate RFI
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx: number of baselines (channels if `transposed`) per workgroup
        - wgsy: number of channels (baselines if `transposed`) per workgroup
    """

    host_class = host.ThresholdSimpleHost

    def __init__(self, context, transposed, flag_value=1, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.context = context
        self.transposed = transposed
        self.wgsx = tuning['wgsx']
        self.wgsy = tuning['wgsy']
        self.flag_value = flag_value
        if transposed:
            source_name = 'rfi/threshold_simple_t.mako'
        else:
            source_name = 'rfi/threshold_simple.mako'
        self.program = accel.build(
            context, source_name,
            {'wgsx': self.wgsx, 'wgsy': self.wgsy,
             'flag_value': flag_value})

    @classmethod
    @tune.autotuner(test={'wgsx': 32, 'wgsy': 4})
    def autotune(cls, context):
        # TODO: do real autotuning
        return {'wgsx': 32, 'wgsy': 4}

    def instantiate(self, *args, **kwargs):
        """Create an instance. See :class:`ThresholdSimpleDevice`."""
        return ThresholdSimpleDevice(self, *args, **kwargs)


class ThresholdSimpleDevice(accel.Operation):
    """Concrete instance of :class:`ThresholdSimpleDeviceTemplate`.

    .. rubric:: Slots

    **deviations** : channels × baselines (or transposed), float32
        Input deviations from the background
    **noise** : baselines, float32
        Noise estimates per baseline
    **flags** : channels × baselines (or transposed), uint8
        Output flags

    Parameters
    ----------
    template : :class:`ThresholdSimpleDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command-queue in which work will be enqueued
    n_sigma : float
        Number of (estimated) standard deviations for the threshold
    channels, baselines : int
        Shape of the visibility array
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """

    def __init__(self, template, command_queue, channels, baselines, n_sigma, allocator=None):
        super(ThresholdSimpleDevice, self).__init__(command_queue, allocator)
        self.template = template
        if template.transposed:
            kernel_name = 'threshold_simple_t'
        else:
            kernel_name = 'threshold_simple'
        self.kernel = template.program.get_kernel(kernel_name)
        self.n_sigma = n_sigma
        self.channels = channels
        self.baselines = baselines
        self.transposed = template.transposed

        shape = (baselines, channels) if self.transposed else (channels, baselines)
        dims = (accel.Dimension(shape[0], self.template.wgsy),
                accel.Dimension(shape[1], self.template.wgsx))
        noise_dim = dims[0] if self.transposed else dims[1]

        self.slots['deviations'] = accel.IOSlot(dims, np.float32)
        self.slots['noise'] = accel.IOSlot((noise_dim,), np.float32)
        self.slots['flags'] = accel.IOSlot(dims, np.uint8)

    def _run(self):
        deviations = self.buffer('deviations')
        noise = self.buffer('noise')
        flags = self.buffer('flags')
        stride = deviations.padded_shape[1]

        global_x = accel.roundup(deviations.shape[1], self.template.wgsx)
        global_y = accel.roundup(deviations.shape[0], self.template.wgsy)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                deviations.buffer, noise.buffer, flags.buffer,
                np.int32(stride),
                np.float32(self.n_sigma)
            ],
            global_size=(global_x, global_y),
            local_size=(self.template.wgsx, self.template.wgsy))

    def parameters(self):
        return {
            'n_sigma': self.n_sigma,
            'flag_value': self.template.flag_value,
            'transposed': self.transposed,
            'channels': self.channels,
            'baselines': self.baselines
        }


class ThresholdSumDeviceTemplate(object):
    """A device version of :class:`katsdpsigproc.rfi.host.ThresholdSumHost`.
    It uses transposed data. Performance will be best with a large work
    group size, because of the stencil-like nature of the computation.

    Parameters
    ----------
    context : |Context|
        Context for which kernels will be compiled
    n_windows : int
        Number of window sizes to use
    flag_value : int
        Number stored in returned value to indicate RFI
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgs: Number of work items to use per work group
        - vt: Number of elements to process in each work item
    """

    host_class = host.ThresholdSumHost
    transposed = True

    def __init__(self, context, n_windows=4, flag_value=1, tuning=None):
        if tuning is None:
            tuning = self.autotune(context, n_windows)
        wgs = tuning['wgs']
        vt = tuning['vt']
        edge_size = 2 ** n_windows - n_windows - 1
        self.chunk = wgs * vt - 2 * edge_size
        assert self.chunk > 0
        self.context = context
        self.n_windows = n_windows
        self.wgs = wgs
        self.vt = vt
        self.flag_value = flag_value
        self.program = accel.build(
            context, 'rfi/threshold_sum.mako',
            {'wgs': self.wgs,
             'vt': self.vt,
             'windows' : self.n_windows,
             'flag_value': self.flag_value})

    @classmethod
    @tune.autotuner(test={'wgs': 128, 'vt': 3})
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

        def generate(**tuning):
            template = cls(context, n_windows=n_windows, tuning=tuning)
            fn = template.instantiate(queue, channels, baselines, 11.0)
            fn.bind(deviations=deviations, noise=noise, flags=flags)
            return tune.make_measure(queue, fn)

        return tune.autotune(generate,
                             wgs=[32, 64, 128, 256, 512],
                             vt=[1, 2, 3, 4, 8, 16])

    def instantiate(self, *args, **kwargs):
        """Create an instance. See :class:`ThresholdSumDevice`."""
        return ThresholdSumDevice(self, *args, **kwargs)


class ThresholdSumDevice(accel.Operation):
    """Concrete instance of :class:`ThresholdSumDeviceTemplate`.

    .. rubric:: Slots

    **deviations** : baselines × channels, float32
        Input deviations from the background
    **noise** : baselines, float32
        Noise estimates per baseline
    **flags** : baselines × channels, uint8
        Output flags

    Parameters
    ----------
    template : :class:`ThresholSumDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command-queue in which work will be enqueued
    channels, baselines : int
        Shape of the visibility array
    n_sigma : float
        Number of (estimated) standard deviations for the threshold
    threshold_falloff : float, optional
        Controls rate at which thresholds decrease (ρ in Offringa 2010)
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`, optional
        Allocator used to allocate unbound slots
    """
    host_class = host.ThresholdSumHost
    transposed = True

    def __init__(self, template, command_queue, channels, baselines, n_sigma, threshold_falloff=1.2,
                 allocator=None):
        super(ThresholdSumDevice, self).__init__(command_queue, allocator)
        self.template = template
        self.kernel = template.program.get_kernel('threshold_sum')
        self.channels = channels
        self.baselines = baselines
        self.n_sigma = [np.float32(n_sigma * pow(threshold_falloff, -i))
                        for i in range(template.n_windows)]
        # For channels we must construct the Dimension here rather than in
        # the IOSlot constructor, so that the deviations and flags share
        # the same object and hence have the same stride.
        dims = (baselines, accel.Dimension(channels))
        self.slots['deviations'] = accel.IOSlot(dims, np.float32)
        self.slots['noise'] = accel.IOSlot((baselines,), np.float32)
        self.slots['flags'] = accel.IOSlot(dims, np.uint8)

    def _run(self):
        deviations = self.buffer('deviations')
        noise = self.buffer('noise')
        flags = self.buffer('flags')

        blocks = accel.divup(self.channels, self.template.chunk)
        args = [deviations.buffer, noise.buffer, flags.buffer,
                np.int32(self.channels), np.int32(deviations.padded_shape[1])]
        args.extend(self.n_sigma)
        self.command_queue.enqueue_kernel(
            self.kernel, args,
            global_size=(blocks * self.template.wgs, self.baselines),
            local_size=(self.template.wgs, 1))

    def parameters(self):
        return {
            'n_sigma': self.n_sigma,
            'flag_value': self.template.flag_value,
            'channels': self.channels,
            'baselines': self.baselines
        }


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

    def instantiate(self, *args, **kwargs):
        """Create an instance. See :class:`FlaggerDevice`."""
        return FlaggerDevice(self, *args, **kwargs)


class FlaggerDevice(accel.OperationSequence):
    """Concrete instance of :class:`FlaggerDeviceTemplate`.

    .. rubric:: Slots

    **vis** : channels × baselines, float32 or complex64
        Input visibilities (or amplitudes, if the backgrounder takes amplitudes)
    **noise** : baselines, float32
        Estimate of per-baseline noise
    **flags** : channels × baselines, uint8
        Output flags
    **channel_flags** : channels, uint8
        Input flag per channel. These are only used for estimating the
        background, and are *not* automatically copied into the output. Any
        visibilities marked as flagged here will *not* be flagged as RFI.

        This slot is only present if the backgrounder template has
        `use_flags` set to true.

    .. rubric:: Temporary slots

    Temporary buffers are presented as slots, which allows them to either
    be set by the user or allocated automatically on first use.

    **deviations** : channels × baselines, float32
        Deviations from the background
    **deviations_t** : baselines × channels, float32, optional
        Transpose of `deviations`
    **flags_t** : baselines × channels, uint8, optional
        Transpose of `flags`

    Parameters
    ----------
    template : :class:`FlaggerDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    channels, baselines : int
        Shape of the visibilities array
    background_args : dict, optional
        Extra keyword arguments to pass to the background instantiation
    noise_est_args : dict, optional
        Extra keyword arguments to pass to the noise estimation instantiation
    threshold_args : dict, optional
        Extra keyword arguments to pass to the threshold instantiation
    allocator : :class:`DeviceAllocator` or :class:`SVMAllocator`
        Allocator used to allocate unbound slots
    """
    def __init__(self, template, command_queue, channels, baselines,
                 background_args={}, noise_est_args={}, threshold_args={},
                 allocator=None):
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.background = self.template.background.instantiate(
            command_queue, channels, baselines, allocator=allocator, **background_args)
        self.noise_est = self.template.noise_est.instantiate(
            command_queue, channels, baselines, allocator=allocator, **noise_est_args)
        self.threshold = self.template.threshold.instantiate(
            command_queue, channels, baselines, allocator=allocator, **threshold_args)

        noise_est_suffix = '_t' if self.noise_est.transposed else ''
        threshold_suffix = '_t' if self.threshold.transposed else ''

        operations = []
        compounds = {
            'vis': ['background:vis'],
            'channel_flags': ['background:flags'],
            'deviations': ['background:deviations', 'transpose_deviations:src'],
            'deviations_t': ['transpose_deviations:dest'],
            'noise': ['noise_est:noise', 'threshold:noise'],
            'flags_t': ['transpose_flags:src'],
            'flags': ['transpose_flags:dest'],
        }
        compounds['deviations' + noise_est_suffix].append('noise_est:deviations')
        compounds['deviations' + threshold_suffix].append('threshold:deviations')
        compounds['flags' + threshold_suffix].append('threshold:flags')

        operations.append(('background', self.background))
        if self.template.transpose_deviations:
            self.transpose_deviations = self.template.transpose_deviations.instantiate(
                command_queue, (channels, baselines))
            operations.append(('transpose_deviations', self.transpose_deviations))
        operations.append(('noise_est', self.noise_est))
        operations.append(('threshold', self.threshold))
        if self.template.transpose_flags:
            self.transpose_flags = self.template.transpose_flags.instantiate(
                command_queue, (baselines, channels))
            operations.append(('transpose_flags', self.transpose_flags))

        super(FlaggerDevice, self).__init__(
            command_queue, operations, compounds, allocator=allocator)


class FlaggerHostFromDevice(object):
    """Wrapper that makes a :class:`FlaggerDeviceTemplate` present the
    interface of :class:`FlaggerHost`. This is intended only for ease of
    use. It is not efficient, because it allocates and frees memory on
    every call.

    Parameters
    ----------
    template : :class:`FlaggerDeviceTemplate`
        Operation template
    command_queue : |CommandQueue|
        Command queue for the operation
    background_args : dict, optional
        Extra keyword arguments to pass to the background instantiation
    noise_est_args : dict, optional
        Extra keyword arguments to pass to the noise estimation instantiation
    threshold_args : dict, optional
        Extra keyword arguments to pass to the threshold instantiation
    """
    def __init__(self, template, command_queue,
                 background_args={}, noise_est_args={}, threshold_args={}):
        self.template = template
        self.command_queue = command_queue
        self.background_args = dict(background_args)
        self.noise_est_args = dict(noise_est_args)
        self.threshold_args = dict(threshold_args)

    def __call__(self, vis, channel_flags=None):
        if channel_flags is not None and not self.template.background.use_flags:
            raise TypeError("channel flags were provided but not included in the template")
        if channel_flags is None and self.template.background.use_flags:
            raise TypeError("channel flags were expected but not provided")
        (channels, baselines) = vis.shape
        fn = self.template.instantiate(
            self.command_queue, channels, baselines,
            self.background_args, self.noise_est_args, self.threshold_args)
        fn.ensure_all_bound()
        fn.buffer('vis').set(self.command_queue, vis)
        if channel_flags is not None:
            fn.buffer('channel_flags').set(self.command_queue, channel_flags)
        fn()
        return fn.buffer('flags').get(self.command_queue)
