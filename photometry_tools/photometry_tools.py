"""Tools for aperture photometry with non native bg/error methods

These functions are used to aid photometric measurements.  Mostly,
they serve to support PhotUtils aperture photometry by giving
convenience functions for using the background stats implemented
in background_median, for both the flux and error measurements.

Authors
-------
    - Varun Bajaj, January 2018
Use
---
    from photometry_tools import run_photometry
    phot_aps = CircularAperture((sources['xcentroid'], sources['ycentroid']),r=10.)
    bg_aps = CircularAnnulus((sources['xcentroid'], sources['ycentroid']), r_in=13., r_out=15.)

    photometry_tbl = run_photometry(phot_aps, bg_aps, data, error_array, bg_method='mode')
"""

import numpy as np
from astropy.table import Table
from background_median import aperture_stats_tbl
from photutils import aperture_photometry

def run_photometry(phot_apertures, bg_apertures, data, error_array, bg_method='mode'):
    phot = aperture_photometry(data, phot_apertures, error=error_array)
    bg_phot = aperture_stats_tbl(data, bg_apertures, sigma_clip=True)

    ap_area = phot_apertures.area()
    bg_method_name = 'aperture_{}'.format(bg_method)

    flux = phot['aperture_sum'] - bg_phot[bg_method_name] * ap_area
    flux_error = compute_phot_error(flux, bg_phot, bg_method, ap_area)

    mag = -2.5 * np.log10(flux)
    mag_err = 1.0857 * flux_error / flux

    stacked = np.hstack([phot_apertures.positions, flux, flux_error, mag, mag_err])
    names = ['X', 'Y', 'flux', 'flux_error', 'mag', 'mag_error']

def compute_phot_error(flux, bg_phot, bg_method, ap_area, epadu=1.0):
    bg_var_terms = (ap_area * bg_phot['aperture_std'] ** 2. ) * (1. + ap_area/bg_phot['aperture_area'])
    variance = flux / epadu + bg_var_terms
    flux_error = variance ** .5
    return flux_error
