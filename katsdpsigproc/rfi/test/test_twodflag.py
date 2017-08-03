"""Tests for :mod:`katsdpsigproc.rfi.twodflag`."""

from __future__ import division, print_function, absolute_import

import numpy as np
from nose.tools import assert_equal

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
        np.testing.assert_equal(expected, out)


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
        np.testing.assert_equal(orig, self.y)

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
        np.testing.assert_equal(orig, self.y)

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
        np.testing.assert_equal(np.zeros(self.shape, np.float32), background)

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
