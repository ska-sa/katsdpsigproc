"""Tests for :mod:`katsdpsigproc.rfi.twodflag`."""

from __future__ import division, print_function, absolute_import

import numpy as np
from nose.tools import assert_equal, assert_less

from .. import twodflag


class TestRunningMean(object):
    """Tests for :func:`katsdpsigproc.rfi.twodflag.running_mean`."""

    def setup(self):
        self.a = np.array([[2.0, 5.0, 4.0, 4.5],
                           [3.2, 3.0, 2.0, 1.5],
                           [-1.0, 0.0, 5.0, 5.0]])

    def test_flat(self):
        expected = np.array([11, 13.5, 11.7, 10.7, 8.2, 6.5, 2.5, 0.5, 4, 10]) / 3
        out = twodflag.running_mean(self.a, 3)
        np.testing.assert_allclose(expected, out, rtol=1e-9)

    def test_axis0(self):
        expected = np.array([[5.2, 8, 6, 6], [2.2, 3, 7, 6.5]]) / 2
        out = twodflag.running_mean(self.a, 2, axis=0)
        np.testing.assert_allclose(expected, out, rtol=1e-9)

    def test_axis1(self):
        expected = np.array([[11, 13.5], [8.2, 6.5], [4, 10]]) / 3
        out = twodflag.running_mean(self.a, 3, axis=1)
        np.testing.assert_allclose(expected, out, rtol=1e-9)

    def test_float32(self):
        expected = (np.array([[11, 13.5], [8.2, 6.5], [4, 10]]) / 3).astype(np.float32)
        out = twodflag.running_mean(self.a.astype(np.float32), 3, axis=1)
        assert_equal(np.float32, out.dtype)
        np.testing.assert_allclose(expected, out, rtol=1e-6)

    def test_precision(self):
        """Must avoid loss of precision due to cumulative sum"""
        x = np.array([1e9, 4, 7, 8], np.float32)
        expected = np.array([0.5e9 + 2, 5.5, 7.5])
        out = twodflag.running_mean(x, 2)
        np.testing.assert_allclose(expected, out, rtol=1e-6)

    def test_too_wide(self):
        """Return empty array if window size larger than array dim"""
        expected = np.empty((0, 4))
        out = twodflag.running_mean(self.a, 5, axis=0)
        np.testing.assert_array_equal(expected, out)


class TestLinearlyInterpolateNans(object):
    """Tests for :func:`katsdpsigproc.rfi.twodflag.linearly_interpolate_nans`."""

    def setup(self):
        self.y = np.array([np.nan, np.nan, 4.0, np.nan, np.nan, 10.0, np.nan, -2.0, np.nan, np.nan])
        self.expected = np.array([4.0, 4.0, 4.0, 6.0, 8.0, 10.0, 4.0, -2.0, -2.0, -2.0])

    def test_basic(self):
        orig = self.y[:]
        out = twodflag.linearly_interpolate_nans(self.y)
        np.testing.assert_allclose(self.expected, out)
        # Check that the input isn't being overwritten
        np.testing.assert_array_equal(orig, self.y)

    def test_no_nans(self):
        out = twodflag.linearly_interpolate_nans(self.expected)
        np.testing.assert_allclose(self.expected, out)

    def test_all_nans(self):
        orig = self.y[:]
        expected = np.zeros_like(self.y)
        self.y[:] = np.nan
        out = twodflag.linearly_interpolate_nans(self.expected)
        np.testing.assert_allclose(self.expected, out)
        # Check that the input isn't being overwritten
        np.testing.assert_array_equal(orig, self.y)

    def test_float32(self):
        expected = self.expected.astype(np.float32)
        out = twodflag.linearly_interpolate_nans(self.y.astype(np.float32))
        assert_equal(np.float32, out.dtype)
        np.testing.assert_allclose(expected, out, rtol=1e-6)


class TestGetbackground2D(object):
    """Tests for :func:`katsdpsigproc.rfi.twodflag.getbackground_2d`.

    This is a difficult function to test, because it's not really practical to
    determine expected results by hand. The tests mainly check corner cases
    where large regions are flagged.
    """

    def setup(self):
        self.shape = (95, 86)
        self.data = np.ones(self.shape, np.float32) * 7.5
        self.flags = np.zeros(self.shape, np.bool_)

    def test_no_flags(self):
        background = twodflag.getbackground_2d(self.data)
        assert_equal(np.float32, background.dtype)
        # It's all constant, so background and output should match.
        # It won't be exact though, because the Gaussian filter accumulates
        # errors as it sums.
        np.testing.assert_allclose(self.data, background, rtol=1e-5)

    def test_all_masked(self):
        self.flags.fill(True)
        background = twodflag.getbackground_2d(self.data, self.flags)
        assert_equal(np.float32, background.dtype)
        np.testing.assert_array_equal(np.zeros(self.shape, np.float32), background)

    def test_in_flags(self):
        # This needs to be done carefully, because getbackground_2d does
        # internal masking on outliers too. We give every 3rd time a higher
        # power and flag it.
        self.data[::3] = 20.0
        self.flags[::3] = 1
        background = twodflag.getbackground_2d(self.data, self.flags)
        expected = np.ones_like(self.data) * 7.5
        np.testing.assert_allclose(expected, background, rtol=1e-5)

    def test_interpolate(self):
        """Linear interpolation across completely flagged data"""
        # Block of channels is 7.5, then a block is flagged (still 7.5), then
        # a block is 3.0.
        self.data[:, 70:] = 3.0
        self.flags[:, 30:70] = 1
        # The setup above has no deviation from the background, which makes the
        # outlier rejection unstable, so we add noise to half the timesteps, and test
        # them at lower precision. We use uniform noise to guarantee no outliers.
        rs = np.random.RandomState(seed=1)
        self.data[:50, :] += rs.uniform(-0.001, 0.001, self.data[0:50, :].shape)

        # The rejection threshold is adjusted, because the default doesn't do
        # well when only about half the data is noisy.
        background = twodflag.getbackground_2d(self.data, self.flags, spike_width=(1.99, 1.99),
                                               reject_threshold=5.0)
        expected = np.zeros_like(self.data)
        expected[:, :35] = 7.5
        expected[:, 65:] = 3.0
        expected[:, 35:65] = np.linspace(7.5, 3.0, 30)
        np.testing.assert_allclose(expected[56:], background[56:], rtol=1e-5)
        np.testing.assert_allclose(expected[:56], background[:56], rtol=1e-2)

    def test_iterations(self):
        expected = self.data.copy()
        # Add some noise
        rs = np.random.RandomState(seed=1)
        self.data += rs.standard_normal(self.data.shape) * 0.1
        # Add a "spike" that's larger than the initial spike_width to check
        # that it gets masked out.
        self.data[20:50, 30:80] += 15

        background = twodflag.getbackground_2d(self.data, iterations=3)
        import matplotlib.pyplot as plt; plt.imshow(background - expected); plt.show()
        np.testing.assert_allclose(expected, background, rtol=1e-2)


class TestSumThresholdFlagger(object):
    """Tests for :class:`katsdpsigproc.rfi.twodflag.SumThresholdFlagger`."""

    def setup(self):
        self.flagger = twodflag.SumThresholdFlagger()
        self.small_data = np.arange(30, dtype=np.float32).reshape(5, 6)
        self.small_flags = np.zeros(self.small_data.shape, np.bool_)
        self.small_flags[3, :] = 1
        self.small_flags[:, 4] = 1
        self.small_flags[2, 0] = 1
        self.small_flags[2, 5] = 1

    def test_average_freq_one(self):
        """_average_freq with 1 channel must have no effect on unflagged data"""
        avg_data, avg_flags = self.flagger._average_freq(self.small_data, self.small_flags)
        expected = self.small_data.copy()
        expected[self.small_flags] = 0
        assert_equal(np.float32, avg_data.dtype)
        assert_equal(np.bool_, avg_flags.dtype)
        np.testing.assert_array_equal(expected, avg_data)
        np.testing.assert_array_equal(self.small_flags, avg_flags)

    def test_average_freq_divides(self):
        """Test _average_freq when averaging factor divides in exactly"""
        expected_data = np.array([
            [0.5, 2.5, 5.0],
            [6.5, 8.5, 11.0],
            [13.0, 14.5, 0.0],
            [0.0, 0.0, 0.0],
            [24.5, 26.5, 29.0]], np.float32)
        expected_flags = np.array([
            [False, False, False],
            [False, False, False],
            [False, False, True],
            [True, True, True],
            [False, False, False]])
        flagger = twodflag.SumThresholdFlagger(average_freq=2)
        avg_data, avg_flags = flagger._average_freq(self.small_data, self.small_flags)
        assert_equal(np.float32, avg_data.dtype)
        assert_equal(np.bool_, avg_flags.dtype)
        np.testing.assert_array_equal(expected_data, avg_data)
        np.testing.assert_array_equal(expected_flags, avg_flags)

    def test_average_freq_uneven(self):
        """Test _average_freq when averaging factor does not divide number of channels"""
        expected_data = np.array([
            [1.5, 5.0],
            [7.5, 11.0],
            [14.0, 0.0],
            [0.0, 0.0],
            [25.5, 29.0]], np.float32)
        expected_flags = np.array([
            [False, False],
            [False, False],
            [False, True],
            [True, True],
            [False, False]], np.bool_)
        flagger = twodflag.SumThresholdFlagger(average_freq=4)
        avg_data, avg_flags = flagger._average_freq(self.small_data, self.small_flags)
        assert_equal(np.float32, avg_data.dtype)
        assert_equal(np.bool_, avg_flags.dtype)
        np.testing.assert_array_equal(expected_data, avg_data)
        np.testing.assert_array_equal(expected_flags, avg_flags)

    def test_sumthreshold_all_flagged(self):
        out_flags = self.flagger._sumthreshold(self.small_data, self.small_flags, 0, [1, 2, 4])
        np.testing.assert_equal(np.zeros_like(self.small_flags), out_flags)

    def _test_sumthreshold_basic(self, axis):
        rs = np.random.RandomState(seed=1)
        data = rs.standard_normal((100, 90)).astype(np.float32) * 3.0
        rfi = np.zeros_like(data)
        # Add some individual spikes and some bad channels
        rfi[10, 20] = 100.0
        rfi[80, 80] = -100.0
        rfi[:, 40] = rs.uniform(80.0, 120.0, size=(100,))
        rfi[:, 2] = -rfi[:, 40]
        # Smaller but wider spike
        rfi[:, 60:67] = rs.uniform(15.0, 20.0, size=(100, 7))
        rfi[:, 10:17] = -rfi[:, 60:67]
        in_flags = np.zeros(data.shape, np.bool_)
        expected_flags = rfi != 0
        data += rfi
        if axis == 0:
            # Swap axes around so that we're doing essentially the same test
            rfi = rfi.T.copy()
            data = data.T.copy()
            in_flags = in_flags.T.copy()
        out_flags = self.flagger._sumthreshold(data, in_flags, axis, self.flagger.windows_freq)
        if axis == 0:
            out_flags = out_flags.T
        # Due to random data, won't get perfect agreement, but should get close
        errors = np.sum(expected_flags != out_flags)
        assert_less(errors / data.size, 0.01)
        # Check for exact match on the individual spikes
        for region in (np.s_[8:13, 18:23], np.s_[78:83, 78:83]):
            np.testing.assert_equal(expected_flags[region], out_flags[region])

    def test_sumthreshold_time(self):
        self._test_sumthreshold_basic(axis=0)

    def test_sumthreshold_frequency(self):
        self._test_sumthreshold_basic(axis=1)

    def test_sumthreshold_existing(self):
        rs = np.random.RandomState(seed=1)
        flagger = twodflag.SumThresholdFlagger(outlier_nsigma=5)
        data = rs.standard_normal((100, 90)).astype(np.float32) * 3.0
        in_flags = np.zeros(data.shape, np.bool_)
        # Corrupt but pre-flag just under half the data, which will skew the
        # noise estimate if not taken into account.
        data[:48] += 1000.0
        in_flags[:48] = True
        # Add some spikes that should be just under the detection limit.
        data[70, 0] = 12.5
        data[70, 1] = -12.5
        # Add some spikes that should still be detected.
        data[70, 2] = 20.0
        data[70, 3] = -20.0
        # Test it
        out_flags = flagger._sumthreshold(data, in_flags, 0, flagger.windows_freq)
        np.testing.assert_array_equal([False, False, True, True], out_flags[70, :4])
