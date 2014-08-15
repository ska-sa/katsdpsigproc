#!/usr/bin/env python

"""RFI flagging package

The package consists of a number of algorithms for *backgrounding* and
for *thresholding*. Backgrounding finds a smooth version of the
spectrum, with RFI removed. Thresholding applies to the difference
between the original spectrum and the background, and identifies the
RFI.

A backgrounding and a thresholding algorithm are combined in
either :class:`FlaggerHost`, :class:`FlaggerDevice`, or
`FlaggerHostFromDevice`.
"""
