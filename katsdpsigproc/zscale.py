"""A Python implementation of the IRAF/sa9 zscale algorithm."""

import random
from typing import Tuple

import numpy as np


def sample_image(image: np.ndarray, max_samples: int = 100000,
                 random_offsets: bool = False) -> np.ndarray:
    """Take regularly-spaced samples from a 2D image.

    This provides suitable input for :func:`zscale`. The samples are taken from
    a regular grid.

    Parameters
    ----------
    image
        2D array of float
    max_samples
        Maximum number of samples to return. The actual number returned may be
        less if the image is smaller or if there are non-finite samples (which
        are discarded)
    random_offsets
        If true, the placement of the grid is chosen randomly; otherwise, it
        start from the pixel at (0, 0). Enabling this is useful when combining
        samples from multiple images.
    """
    nx, ny = image.shape
    stride = int(max(1.0, np.sqrt(nx * ny / max_samples)))
    if random_offsets:
        offset_x = random.randrange(0, stride)
        offset_y = random.randrange(0, stride)
    else:
        offset_x = 0
        offset_y = 0
    samples = image[offset_x::stride, offset_y::stride].flatten()
    # Remove blanked pixels
    samples = samples[np.isfinite(samples)]
    # Force the maximum number of samples
    samples = samples[:max_samples]
    return samples


def zscale(samples: np.ndarray, contrast: float = 0.02, stretch: float = 5.0,
           sigma_rej: float = 3.0, max_iter: int = 500, max_reject: float = 0.1,
           min_npix: int = 5) -> Tuple[float, float]:
    """A Python implementation of the IRAF/ds9 zscale algorithm.

    This is used for determining maximum and minumum image values for a given
    contrast. This is used to scale a colourmap applied to FITS images when
    saving to alternative image formats or for on screen display.

    A description of the zscale algorithm can be found here:
    https://iraf.net/forum/viewtopic.php?showtopic=134139

    This implementation has a few minor changes from IRAF:

      1. Don't default to the full data range if iterations
         reject `max_reject` samples—use the range determined
         on the final iteration instead.
      2. Use the masked set of samples to work out the image
         median. The masked data should give an answer without
         including source/artefact contributions to the pixel
         distribution.
      3. Addition of a stretch factor to scale the derived
         minimum and maximum. This is applicable to radio
         images, where negative pixel values can cause the
         vanilla IRAF algorithm to display too much of the
         background rumble.

    Parameters
    ----------
    samples
        1D array of pixel values sampled from the full image.
        See :func:`sample_image`.
    contrast
        The scaling factor (between 0 and 1) for determining
        the returned minimum and maximum value. Larger values
        decrease the difference between them.
    stretch
        Scale factor by which to scale the contrasted gradient.
        The minimum value is derived from gradient / stretch
        and the maximum value from gradient * stretch.
    sigma_rej
        Multiple of standard deviation to reject outliers
        while iterating.
    max_iter
        Maximum number of iterations.
    max_reject
        Stop iterating if number of pixels left is less than
        `max_reject`*npix.
    min_npix: int
        Hard limit on the minimum number of pixels allowed in
        the sample and after rejection.

    Returns
    -------
    z1, z2 : float
        Minimum and maximum sample values to be mapped to the extremes
        of a colour map.
    """
    samples = np.sort(samples)
    npix = len(samples)
    zmin = samples[0]
    zmax = samples[-1]

    # Minimum number of pixels allowed
    minpix = max(min_npix, int(npix * max_reject))

    # Grow rejected pixels in each iteration
    ngrow = max(1, int(npix * 0.01))
    # Ensure ngrow is odd, so that growth is symmetric
    if ngrow % 2 == 0:
        ngrow -= 1

    if npix <= minpix:
        return zmin, zmax

    # Re-map indices from -1.0 to 1.0
    # to improve fitting.
    xscale = 2.0 / (npix - 1)
    xnorm = np.arange(npix)
    xnorm = xnorm * xscale - 1.0

    # Set up for iteration
    ngoodpix = npix
    last_ngoodpix = npix + 1

    # Mask used in k-sigma clipping.
    badpix = np.zeros(npix, dtype=np.bool)

    # Iterate, until maximum iteration, too many pixels
    # rejected or until no change in number of good pixels
    niter = 0
    while niter < max_iter and ngoodpix > minpix and ngoodpix < last_ngoodpix:
        # Fit a line to the remaining pixels
        good_xnorm = xnorm[~badpix]
        A = np.vstack((good_xnorm, np.ones(len(good_xnorm)))).T
        slope, intercept = np.linalg.lstsq(A, samples[~badpix], rcond=None)[0]

        # Subtract fitted line from the full data array
        fitted = xnorm * slope + intercept
        flat = samples - fitted

        # Compute the k-sigma rejection threshold
        sigma = np.std(flat[~badpix])
        threshold = sigma * sigma_rej

        # Detect and reject pixels further than k*sigma from the fitted line
        badpix = np.abs(flat) > threshold

        # Compute number of remaining pixels before convolution
        last_ngoodpix = ngoodpix
        ngoodpix = np.sum(~badpix)

        # Convolve with a kernel of length ngrow
        cum = np.cumsum(np.pad(badpix, (ngrow // 2 + 1, ngrow // 2)))
        badpix[:] = cum[ngrow:] - cum[:-ngrow]

        niter += 1

    # Transform the slope back to the X range [0:npix-1]
    slope = slope * xscale

    # Apply contrast scaling
    if contrast > 0:
        slope = slope / contrast

    # Stretch the slope
    slope_hi = slope * stretch
    slope_low = slope / stretch

    # Median of the remaining samples is close to true
    # pixel median sans sources/artefacts
    good_samples = samples[~badpix]
    median = np.median(good_samples)

    # Find the indices of the median, low and high pixels
    median_pixel = np.searchsorted(samples, median)
    low_pixel = np.searchsorted(samples, good_samples[0])
    hi_pixel = np.searchsorted(samples, good_samples[-1])

    # Derive scale limits from slope_low and slope_hi
    z1 = max(zmin, median - (median_pixel - low_pixel) * slope_low)
    z2 = min(zmax, median + (hi_pixel - median_pixel) * slope_hi)

    return z1, z2
