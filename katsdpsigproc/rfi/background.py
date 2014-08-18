from ..accel import DeviceArray, LinenoLexer, push_context
import numpy as np
import scipy.signal as signal
from mako.lookup import TemplateLookup
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import pycuda.driver as cuda
import os.path

_lookup = TemplateLookup(
        os.path.abspath(os.path.dirname(__file__)),
        lexer_cls = LinenoLexer)

class BackgroundHostFromDevice(object):
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
    def __init__(self, width):
        self.width = width

    def __call__(self, vis):
        amp = np.abs(vis)
        return amp - signal.medfilt2d(amp, [self.width, 1]).astype(np.float32)

class BackgroundMedianFilterDevice(object):
    host_class = BackgroundMedianFilterHost

    def __init__(self, ctx, width, wgs, csplit):
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
        (channels, baselines) = shape
        padded_baselines = (baselines + self.wgs - 1) // self.wgs * self.wgs
        return (channels, padded_baselines)

    def __call__(self, vis, deviations, stream=None):
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
