#!/usr/bin/env python

"""RFI flagging package.

The package consists of a number of algorithms for *backgrounding*,
*noise estimation* and *thresholding*. Backgrounding finds a smooth version of
the spectrum, with RFI removed. Noise estimation computes statistics on the
Thresholding applies to the difference between the original spectrum and the
background, and identifies the RFI as deviations that are too large to be
noise.

A backgrounding, a noise estimation and a thresholding algorithm are combined
in either :class:`host.FlaggerHost`, :class:`device.FlaggerDevice`, or
:class:`device.FlaggerHostFromDevice`.
"""

MAD_NORMAL = 1.4826
"""Ratio between `median absolute deviation`_ and standard deviation of a Gaussian distribution.

.. _median absolute deviation: https://en.wikipedia.org/wiki/Median_absolute_deviation
"""
