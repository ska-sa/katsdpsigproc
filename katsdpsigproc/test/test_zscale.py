"""Tests for :mod:`katsdpsigproc.zscale."""

import numpy as np
from nose.tools import assert_equal, assert_less

from ..zscale import sample_image, zscale


class TestSampleImage:
    def setup(self):
        self.image = np.arange(30.0).reshape(5, 6)

    def test_undersample(self):
        sample = sample_image(self.image, 1)
        np.testing.assert_array_equal(sample, np.array([0]))
        sample = sample_image(self.image, 26)
        np.testing.assert_array_equal(sample, np.arange(26.0))

    def test_oversample(self):
        sample = sample_image(self.image, 10000)
        np.testing.assert_array_equal(sample, np.arange(30.0))

    def test_nan(self):
        self.image[0, 0] = np.nan
        sample = sample_image(self.image, 10000)
        np.testing.assert_array_equal(sample, np.arange(1.0, 30.0))

    def test_random_offsets(self):
        image = np.arange(10000.0).reshape(100, 100)
        rs = np.random.RandomState(seed=1)
        passes = 1000
        n = 100
        # The algorithm is still biased, but check that it is not *too*
        # biased.
        s = 0
        for i in range(passes):
            sample = sample_image(image, n, random_offsets=rs)
            s += np.sum(sample)
        mean = s / n / passes
        assert_less(4800, mean)
        assert_less(mean, 5200)


class TestZscale:
    def setup(self):
        rs = np.random.RandomState(seed=1)
        linear = np.linspace(4.0, 7.0, 1000)
        noise = rs.normal(scale=1e-3, size=linear.shape)
        outliers = rs.uniform(-50.0, 100.0, 20)
        self.samples = np.concatenate([linear + noise, outliers])
        rs.shuffle(self.samples)

    def test_simple(self):
        z1, z2 = zscale(self.samples, contrast=0.0, stretch=1.0)
        np.testing.assert_allclose(z1, 4.0, rtol=1e-2)
        np.testing.assert_allclose(z2, 7.0, rtol=1e-2)

    def test_contrast(self):
        z1, z2 = zscale(self.samples, contrast=0.2, stretch=1.0)
        np.testing.assert_allclose(z1, 5.5 - 1.5 / 0.2, rtol=1e-1)
        np.testing.assert_allclose(z2, 5.5 + 1.5 / 0.2, rtol=1e-1)

    def test_contrast_clip(self):
        z1, z2 = zscale(self.samples, contrast=0.002, stretch=1.0)
        assert_equal(z1, min(self.samples))
        assert_equal(z2, max(self.samples))

    def test_stretch(self):
        z1, z2 = zscale(self.samples, contrast=0.2, stretch=2.0)
        np.testing.assert_allclose(z1, 5.5 - 1.5 / 0.2 / 2, rtol=1e-1)
        np.testing.assert_allclose(z2, 5.5 + 1.5 / 0.2 * 2, rtol=1e-1)
