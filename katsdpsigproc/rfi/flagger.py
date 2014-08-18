from ..accel import DeviceArray
import numpy as np

class FlaggerHost(object):
    """Combine host background and thresholding implementations
    to make a flagger.
    """

    def __init__(self, background, threshold):
        self.background = background
        self.threshold = threshold

    def __call__(self, vis):
        """Perform the flagging.

        Parameters
        ----------
        vis : array_like
            The input visibilities as a 2D array of complex64, indexed
            by channel and baseline.

        Returns
        -------
        numpy.ndarray
            Flags of the same shape as `vis`
        """

        deviations = self.background(vis)
        flags = self.threshold(deviations)
        return flags

class FlaggerDevice(object):
    def __init__(self, background, threshold):
        self.background = background
        self.threshold = threshold
        self.ctx = self.background.ctx
        self.deviations = None

    def min_padded_shape(self, shape):
        return map(max,
                self.background.min_padded_shape(shape),
                self.threshold.min_padded_shape(shape))

    def __call__(self, vis, flags, stream=None):
        assert vis.shape == flags.shape
        assert vis.padded_shape == flags.padded_shape
        assert np.all(np.greater_equal(self.min_padded_shape(vis.shape), vis.padded_shape))

        if (self.deviations is None or
                self.deviations.shape != vis.shape or
                self.deviations.padded_shape != vis.padded_shape):
            self.deviations = DeviceArray(self.ctx, vis.shape, np.float32, vis.padded_shape)

        self.background(vis, self.deviations, stream)
        self.threshold(self.deviations, flags, stream)

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
        device_vis = DeviceArray(self.real_flagger.ctx, vis.shape, np.complex64, padded_shape)
        device_vis.set(vis)
        device_flags = DeviceArray(self.real_flagger.ctx, vis.shape, np.uint8, padded_shape)
        self.real_flagger(device_vis, device_flags)
        return device_flags.get()
