"""Wrapper to perform photometry without knowing PhotUtils. Like IRAF.

This function is a wrapper around iraf_style_photometry, and removes
the need to know how to construct PhotUtils apertures.  This was
created to make using these photometry tools easier, as only the
photometric parameters need to be known.  The interface is similar
to IRAF's DAOPHOT.  See the docstring of photometry() and
photometry_tools.py for info.

Authors
-------
    - Varun Bajaj, January 2018
Use
---
    from phot_wrapper import photometry

    Example call
    photometry_tbl = photometry(
        data, coords=xys, radius=10., annulus=15., dannulus=3.,
        salgorithm='median',origin=1.0)
"""

import numpy as np

from .photometry_with_errors import iraf_style_photometry
from photutils import CircularAnnulus, CircularAperture

def photometry(
        data,
        coords=None,
        coord_file=None,
        radius=10.,
        annulus=10.,
        dannulus=5.,
        salgorithm='mode',
        error_array=None,
        epadu=1.0,
        origin=0.0):
    """Computes photometry with PhotUtils apertures, with IRAF formulae

    Parameters
    ----------
    data : array
        The data for the image to be measured.
    coords : array_like or `~astropy.units.Quantity`, optional
        Pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` tuple
            * list of ``(x, y)`` tuples
            * ``Nx2`` or ``2xN`` `~numpy.ndarray`
            * ``Nx2`` or ``2xN`` `~astropy.units.Quantity` in pixels

        Note that a ``2x2`` `~numpy.ndarray` or
        `~astropy.units.Quantity` is interpreted as ``Nx2``, i.e. two
        rows of (x, y) coordinates.

        NOTE: Either coords or  coord_file must be defined
    coord_file : str, optional
        The name of the file containing the coordinates.  The file
        should should have 2 columns (X and Y), and a row for each
        source.  Lines starting with '#' are skipped.  Do not
        include column names unless those lines start with '#'.
    radius : float, optional
        The radius of the photometric aperture in pixels. Default 10.
    annulus: float, optional
        The inner radius of the background annulus in pixels.
        Default 10.
    dannulus: float, optional
        The width of the background annulus in pixels.
        Default 5.
    salgorithm: {'mean', 'median', 'mode'}, optional
        The statistic used to calculate the background.
        All measurements are sigma clipped.
        NOTE: From DAOPHOT, mode = 3 * median - 2 * mean.
    error_array: array, optional
        The array of pixelwise error of the data.  If none, the
        Poisson noise term in the error computation will just be the
        square root of the flux/epadu. If not none, the
        aperture_sum_err column output by aperture_photometry
        (divided by epadu) will be used as the Poisson noise term.
    epadu : float, optional
        Gain in electrons per adu (only use if image units aren't e-).
    origin : float, optional
        The position of the center of the bottom-left pixel of data.
        The coordinate convention of the package used to find the
        sources dictates what this value should be. If coordinates are
        from DS9, IRAF, or SExtractor, set to 1. If coordinates are
        from PhotUtils (or another 0 indexed coordinate system), leave
        as the default.  Default 0.


    Returns
    -------
    photometry_tbl : astropy.table.Table
        An astropy Table with the colums X, Y, flux, flux_error, mag,
        and mag_err measurements for each of the sources.

    """
    if coords is not None:
        xy = coords
    elif coord_file:
        xy = np.genfromtxt(coord_file)
    else:
        raise RuntimeError('Either coords or coordfile must be defined\
                            see docstring for details.')

    xy = list(zip(*(np.array(xy).T - origin))) # Normalize positions to phoutils coordinates

    phot_apertures = CircularAperture(xy, radius)
    bg_apertures = CircularAnnulus(xy, annulus, annulus+dannulus)

    photometry_results = iraf_style_photometry(phot_apertures,
                                               bg_apertures, data,
                                               error_array, salgorithm,
                                               epadu)

    # Add back the origin to get the coordinates to match the input
    photometry_results['X'] += origin
    photometry_results['Y'] += origin
    return photometry_results
