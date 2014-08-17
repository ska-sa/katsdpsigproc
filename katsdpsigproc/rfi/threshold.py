import numpy as np
from ..accel import DeviceArray
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from mako.lookup import TemplateLookup
import os.path

_lookup = TemplateLookup(
        os.path.abspath(os.path.dirname(__file__)),
        lexer_cls = LinenoLexer)

class ThresholdHostFromDevice(object):
    def __init__(self, real_threshold):
        self.real_threshold = real_threshold

    def __call__(self, deviations):
        padded_shape = self.real_threshold.min_padded_shape(deviations.shape)
        device_deviations = DeviceArray(
                shape=deviations.shape, dtype=np.float32, padded_shape=padded_shape)
        device_deviations.set(deviations)
        device_flags = DeviceArray(
                shape=deviations.shape, dtype=np.uint8, padded_shape=padded_shape)
        self.real_threshold(device_deviations, device_flags)
        return device_flags.get()

class ThresholdMADHost(object):
    def __init__(self, n_sigma, flag_value=1):
        self.factor = 1.4826 * n_sigma
        self.flag_value = flag_value

    def median_abs(self, deviations):
        (channels, baselines) = deviations.shape
        out = np.empty(baselines)
        for i in range(baselines):
            abs_dev = np.abs(deviations[:, i])
            out[i] = np.median(abs_dev[abs_dev > 0])
        return out

    def __call__(self, deviations):
        medians = self.median_abs(deviations)
        flags = (deviations > self.factor * medians).astype(np.uint8)
        return flags * self.flag_value

class ThresholdMADDevice(object):
    host_class = ThresholdMADHost

    def __init__(self, n_sigma, wgsx, wgsy, flag_value=1):
        self.factor = 1.4826 * n_sigma
        self.wgsx = wgsx
        self.wgsy = wgsy
        self.flag_value = flag_value
        source = _lookup.get_template('threshold_mad.cu').render(
                wgsx=wgsx, wgsy=wgsy, wgs=wgsx * wgsy,
                flag_value=flag_value)
        module = SourceModule(source, no_extern_c=True)
        self.kernel = module.get_function('threshold_mad')

    def min_padded_shape(self, shape):
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
        assert deviations.shape == flags.shape
        assert deviations.padded_shape == flags.padded_shape
        (channels, baselines) = deviations.shape
        blocks = self._blocks(deviations)
        vt = self._vt(deviations)
        self.kernel(
                deviations.buffer, flags.buffer,
                np.int32(channels), np.int32(deviations.padded_shape[1]),
                np.float32(self.factor), np.int32(vt),
                block=(self.wgsx, self.wgsy, 1), grid=(blocks, 1), stream=stream)
