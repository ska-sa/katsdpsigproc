from ..accel.array import DeviceArray
import numpy as np
import scipy.signal as signal

class BackgroundHostFromDevice(object):
    def __init__(self, real_background):
        self.api = api
        self.real_background = real_background

    def __call__(self, vis):
        padded_shape = self.real_background.min_padded_shape(self, vis.shape)
        device_vis = DeviceArray(self.real_background.api,
                vis.shape, np.complex64, padded_shape)
        device_vis.set(vis)
        device_deviations = DeviceArray(self.real_background.api,
                vis.shape, np.float32, padded_shape)
        self.real_background(device_vis, device_deviations)

class BackgroundMedianFilterHost(object):
    def __init__(self, width):
        self.width = width

    def __call__(self, vis):
        amp = np.abs(vis)
        return amp - signal.medfilt2d(amp, [self.width, 1]).astype(np.float32)
