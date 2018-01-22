"""Tools for aperture photometry with non native bg/error methods

This function serves to ease the computation of photometric magnitudes
and errors using PhotUtils by replicating DAOPHOT's photometry and
error methods.  The formula for DAOPHOT's error is:

err = sqrt (Poisson_noise / epadu
            + area * stdev**2
            + area**2 * stdev**2 / nsky)

Which gives a magnitude error:

mag_err = 1.0857 * err / flux

Where epadu is electrons per ADU (gain), area is the photometric
aperture area, stdev is the uncertainty in the sky measurement
and nsky is the sky annulus area.  To get the uncertainty in the sky
we must use a custom background tool, which also enables computation of
the mean and median of the sky as well (more robust statistics).
All the stats are sigma clipped.  These are calculated by the
functions in aperture_stats_tbl.


NOTE: Currently, the background computations will fully include a
pixel that has ANY overlap with the background aperture (the annulus).
This is to simplify the computation of the median, as a
weighted median is nontrivial, and slower.

Authors
-------
    - Varun Bajaj, January 2018
Use
---
    from photometry_tools import iraf_style_photometry
    phot_aps = CircularAperture(
        (sources['xcentroid'], sources['ycentroid']),
        r=10.)
    bg_aps = CircularAnnulus(
        (sources['xcentroid'], sources['ycentroid']),
        r_in=13., r_out=15.)

    photometry_tbl = iraf_style_photometry(
        phot_aps, bg_aps, data, error_array, bg_method='mode')
"""

import numpy as np
from astropy.table import Table
from background_median import aperture_stats_tbl
from photutils import aperture_photometry

def iraf_style_photometry(
        phot_apertures,
        bg_apertures,
        data,
        error_array=None,
        bg_method='mode'):
    """Computes photometry with PhotUtils apertures, with IRAF formulae

    Parameters
    ----------
    phot_apertures : photutils PixelAperture object (or subclass)
        The PhotUtils apertures object to compute the photometry.
        i.e. the object returned via CirularAperture.
    bg_apertures : photutils PixelAperture object (or subclass)
        The phoutils aperture object to measure the background in.
        i.e. the object returned via CircularAnnulus.
    data : array
        The data for the image to be measured.
    error_array: array
        The array of pixelwise error of the data.  If none, the
        Poisson noise term in the error computation will just be the
        square root of the flux. If not none, the aperture_sum_err
        column output by aperture_photometry will be used as the
        Poisson noise term.
    bg_method:
        The statistic used to calculate the background.
        Valid options are mean, median, or mode (default).
        All measurements are sigma clipped.
        NOTE: From DAOPHOT, mode = 3 * median - 2 * mean.

    Returns
    -------
    final_tbl : astropy.table.Table
        An astropy Table with the colums X, Y, flux, flux_error, mag,
        and mag_err measurements for each of thesources

    """

    phot = aperture_photometry(data, phot_apertures, error=error_array)
    bg_phot = aperture_stats_tbl(data, bg_apertures, sigma_clip=True)

    ap_area = phot_apertures.area()
    bg_method_name = 'aperture_{}'.format(bg_method)

    flux = phot['aperture_sum'] - bg_phot[bg_method_name] * ap_area

    # Need to use variance of the sources
    # for Poisson noise term in error computation.
    #
    # This means error needs to be squared.
    # If no error_array error = flux ** .5
    if error_array:
        flux_error = compute_phot_error(phot['aperture_sum_err']**2.0,
                                        bg_phot, bg_method, ap_area)
    else:
        flux_error = compute_phot_error(flux, bg_phot,
                                        bg_method, ap_area)

    mag = -2.5 * np.log10(flux)
    mag_err = 1.0857 * flux_error / flux

    # Make the final table
    X, Y = phot_apertures.positions.T
    cols = [X, Y, flux, flux_error, mag, mag_err]
    for col in cols:
        print col.shape
    stacked = np.stack([X, Y, flux, flux_error, mag, mag_err], axis=1)
    print stacked.shape
    # return [X, Y, flux, flux_error, mag, mag_err]
    names = ['X', 'Y', 'flux', 'flux_error', 'mag', 'mag_error']

    final_tbl = Table(data=stacked, names=names)
    return final_tbl

def compute_phot_error(
        flux_variance,
        bg_phot,
        bg_method,
        ap_area,
        epadu=1.0):
    """Computes the flux errors using the DAOPHOT style computation"""
    bg_variance_terms = (ap_area * bg_phot['aperture_std'] ** 2. ) \
                        * (1. + ap_area/bg_phot['aperture_area'])
    variance = flux_variance / epadu + bg_variance_terms
    flux_error = variance ** .5
    return flux_error
