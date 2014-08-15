import numpy as np
from ..accel import DeviceArray

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
