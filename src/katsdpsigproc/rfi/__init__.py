################################################################################
# Copyright (c) 2014, 2019, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

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
