import math
import numpy as np

from functools import partial
from itertools import product
from multiprocessing import cpu_count, Pool
from time import perf_counter

from astropy.io import fits
from astropy.table import Table
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.psf import FittableImageModel
from photutils.psf.models import GriddedPSFModel
from scipy.ndimage import convolve, maximum_filter
from scipy.optimize import curve_fit

from astropy.modeling import Fittable2DModel, Parameter

from photometry_tools import aperture_stats_tbl
from .CatalogUtils import make_sky_coord_cat
from .PSFPhot import check_images, get_standard_psf, validate_file
from .PSFUtils import SlowGriddedFocusPSFModel, make_models

def run_python_psf_fitting(input_images, psf_model_file=None, hmin=5, fmin=1E3,
                           pmax=7E4, qmax=.5, cmin=-1., cmax=.1,
                           ncpu=None):
    """
    Main function to run to do finding/PSF fitting of stars with python engine

    This is the top level function of the python PSF photometry interface.
    This gets the PSF and other information, loops over the images and calls
    the fitting functions.  The inputs control the star finding/rejection
    criteria.  The outputs photometric catalogs are saved as ascii tables.

    Note: Focus dependent PSFs not yet supported in the python engine (but
    are coming soon)

    Parameters
    ----------
    input_images : list
        List of image filenames (strings) to measure.  Must be fits files.
    psf_model_file : str, optional
        Name of fits file containing PSF model.  If None (default),
        finds/downloads the appropriate model from WFC3 website.
    hmin : int, optional
        Minimum separation (in pixels) between peaks for them to be measured.
        Default 5.
    fmin : float, optional
        Minimum flux (in image units) in the central 2x2 pixels for a source
        to be measured. Default 1000.
    pmax : float, optional
        Maximum flux (in image units) of the brightest pixel in a source for
        it to be measured. Default 70000 (saturation threshold).  Setting this
        slightly lower is likely advantageous, as the python interface cannot
        measure saturated UVIS stars well.
    qmax : float, optional
        Maximum fit quality for fit of a given star to be considered good.
        Fit quality is defined as:
        sum(abs((data - sky - fitted_model)/fitted_flux)))
    cmin : float, optional
        Minimum scaled residual of the central pixel for the measurement to be
        considered good.  Scaled residual is defined as:
        (peak pixel value - sky - fit peak pixel value)/fitted_flux.
        This helps reject extended/diffuse sources.  Default -1.
    cmax : float, optional
        Maximum scaled residual of the central pixel for the measurement to be
        considered good (see cmin).  Default 0.1.  Helps reject cosmic rays.
    ncpu : int, optional
        Number of CPUs to use to distribute fitting across.  If None, uses
        all cpus.  Default None.
    """
    filt = check_images(input_images)
    if psf_model_file is None:
        psf_model_file = get_standard_psf('./', filt)
    det = fits.getval(input_images[0], 'DETECTOR')
    exts = [1] if det == 'IR' else [1,2]

    mods = make_models(psf_model_file)
    for im in input_images:
        print(im)
        sub_flag = fits.getval(im, 'SUBARRAY')
        if sub_flag:
            raise ValueError('Subarray images are not yet supported \
                              in the python fitting engine.')

        for ext in exts:
            data = fits.getdata(im , extname='SCI', extver=ext)
            tbl = measure_stars(data, mods[ext-1], hmin, fmin, pmax,
                                qmax, cmin, cmax, ncpu)
            tbl['x'] += 1.
            tbl['y'] += 1.
            make_sky_coord_cat(tbl, im, sci_ext=ext)
            break

def measure_stars(data, mod, hmin=5, fmin=1E3, pmax=7E4,
                  qmax=.5, cmin=-1., cmax=.1, ncpu=None):
    """
    Finds and measures stars in data array using the PSF model.

    This is the handler function to do the finding/fitting for stars in
    a single science extension of an image, as well as rejection of poor
    fit sources.

    Parameters
    ----------
    data : `numpy.ndarray`
        Array containing the image data for 1 chip (1 science extension)
    mod : `photutils.psf.models.GriddedPSFModel`
        The model of the PSF across that chip.  See make_models().
    hmin : int, optional
        Minimum separation (in pixels) between peaks for them to be measured.
        Default 5.
    fmin : float, optional
        Minimum flux (in image units) in the central 2x2 pixels for a source
        to be measured. Default 1000.
    pmax : float, optional
        Maximum flux (in image units) of the brightest pixel in a source for
        it to be measured. Default 70000 (saturation threshold).  Setting this
        slightly lower is likely advantageous, as the python interface cannot
        measure saturated UVIS stars well.
    qmax : float, optional
        Maximum fit quality for fit of a given star to be considered good.
        Fit quality is defined as:
        sum(abs((data - sky - fitted_model)/fitted_flux)))
    cmin : float, optional
        Minimum scaled residual of the central pixel for the measurement to be
        considered good.  Scaled residual is defined as:
        (peak pixel value - sky - fit pixel value)/fit flux.
        This helps reject extended/diffuse sources.  Default -1.
    cmax : float, optional
        Maximum scaled residual of the central pixel for the measurement to be
        considered good (see cmin).  Default 0.1.  Helps reject cosmic rays.
    ncpu : int, optional
        Number of CPUs to use to distribute fitting across.  If None, uses
        all cpus.  Default None.

    Returns
    -------
    tbl : `astropy.table.Table`
        Table of fit source positions, magnitudes, sky values, fit qualities,
         and scaled residual of the central pixel.
    """

    # Get filtered images to get peak measurements
    filt_image, max_4sum = _filter_images(data, hmin)
    # Find source candidates
    xs, ys = _find_sources(data, filt_image, max_4sum, fmin, pmax)
    skies = estimate_all_backgrounds(xs, ys, 8.5, 13.5, data)

    # Give buffer of 10% for peak pixel tolerance
    max_peak_val = _max_peakiness(mod) + .1

    # Throw out sources that are more peaked than PSF allows
    # Throwing bad ones now improves performance
    mask = reject_sources(xs, ys, data, max_4sum, skies, max_peak_val)
    xs = xs[mask]
    ys = ys[mask]
    skies = skies[mask]

    # Measure the remaining candidates
    for i in range(10):
        tbl = do_stars_mp(xs, ys, skies, mod, data, ncpu)
        # Last rejection pass
        final_good_mask = (tbl['q']<qmax) & (tbl['cx']<cmax) & (tbl['cx']>cmin)
        output_tbl = tbl[final_good_mask]
        rej = np.sum(~final_good_mask)
        print('Rejected {} more sources after qmax and excess clip'.format(rej))
    return output_tbl
#---------------------------DETECTION--------------------------------

def _conv_origin(data, origin):
    """Helper function to shift around conv kernel for peak finding"""
    # should just use lambda for this, but maybe useful if called > 1 time
    return convolve(data, weights=np.ones((2,2)),
                    mode='constant', cval=0,
                    origin=origin)

def _filter_images(data, hmin):
    """Performs filtering/convolution on images for source finding"""
    #Laziest way to get a circle mask
    fp = CircularAperture((0,0), r=hmin).to_mask().data>.1
    fp = fp.astype(bool)

    # Apply maximum filter, flux filter
    filt_image = maximum_filter(data, footprint=fp,
                                mode='constant', cval=0)
    origins = product([0,-1], [0,-1])
    max_4sum = np.amax([_conv_origin(data, o) for o in origins], axis=0)
    return(filt_image, max_4sum)

def _find_sources(data, filt_image, max_4sum, fmin=1E3, pmax=7E4):
    yi, xi = np.where(data>=filt_image)
    mask1 = (max_4sum[yi, xi] > fmin) & (data[yi,xi] < pmax)
    mask2 = (xi>2) & (yi>2) & (xi<data.shape[1]-3) & (yi<data.shape[0]-3)
    mask = mask1 & mask2
    yi = yi[mask]
    xi = xi[mask]
    return(xi, yi)

def _max_peakiness(mod):
    yg, xg = np.mgrid[-2:3,-2:3]
    origins = product([0,-1], [0,-1])
    peak_ratios = []
    for xc, yc in mod.grid_xypos:
        ev = mod.evaluate(xg+xc, yg+yc, 1., xc, yc)
        peak_flux = np.amax(_conv_origin(ev,[0,0]))
        peak_ratios.append(np.amax(ev)/peak_flux)
    return np.amax(peak_ratios)

def reject_sources(xs, ys, data, max_4sum, skies, max_peak_val):
    peak_vals = data[ys, xs] - skies
    box_fluxes = max_4sum[ys, xs] - skies*4.
    peak_ratios = peak_vals/box_fluxes
    good_peak_mask = peak_ratios < max_peak_val

    ntot = len(good_peak_mask)
    nrej = ntot-sum(good_peak_mask)
    print('Rejected {} of {} sources for being too peaked'.format(nrej, ntot))
    return good_peak_mask



#-----------------------------FITTING------------------------------



def estimate_all_backgrounds(xs, ys, r_in, r_out, data, stat='aperture_mode'):
    """
    Compute sky values around (xs, ys) in data with various parameters

    See photometry_tools.aperture_stats_tbl for more details.
    """
    ans = CircularAnnulus(positions=zip(xs, ys), r_in=r_in, r_out=r_out)
    bg_ests = aperture_stats_tbl(apertures=ans, data=data, sigma_clip=True)
    return np.array(bg_ests[stat])


class FlattenedModel:
    """This is a wrapper class to return the evaluated psf as a 1d array"""
    # Could probably make this a subclass of FittableImageModel instead.
    def __init__(self, psf_fittable_model):
        self.mod = psf_fittable_model

    def evaluate(self, x_y, flux, x_0, y_0):
        """Evaluate the model, and flatten the output"""
        x, y = x_y
        return np.ravel(self.mod.evaluate(x, y, flux=flux,
                                          x_0=x_0, y_0=y_0))


def fit_star(xi, yi, bg_est, model, im_data):
    """
    Fit object at some (x,y) in data array with PSF model

    This is the function that fits each object with the PSF.  It cuts out
    a 5x5 pixel box around the peak of the object, and fits the PSF to that
    cutout.  If it fails, all return values are nans.  Uses curve_fit in
    scipy.optimize for fitting.  The input pixels are given sigmas of
    sqrt(abs(im_data)) for weighting the fit.

    Parameters
    ----------
    xi : int or float
        x position of objects peak, in pixel coordinates.  Can just be the
        integer pixel position.
    yi : int or float
        y position of objects peak, in pixel coordinates.  Can just be the
        integer pixel position.
    bg_est : float
        Sky background level around the object to be subtracted before fitting
    model : `FlattenedModel`
        The GriddedPSFModel, with an evaluate method that returns a flattened
        version of the output (instead of shape being (n,m), get (n*m,))
    im_data : `numpy.ndarray`
        The full image (single chip) data array from which the object should
        be cut out.

    Returns
    -------
    f : float
        The fitted flux of the model
    x : float
        The fitted x position of the model
    y : float
        The fitted y position of the model
    q : float
        The calculated fit quality
    cx : float
        The scaled residual of the central pixel

    """
    # Should try making guesses for x,y to see if performance improves
    yg, xg = np.mgrid[-2:3,-2:3]
    yf, xf = yg+int(yi+.5), xg+int(xi+.5)

    cutout = im_data[yf, xf]
    f_guess = np.sum(cutout-bg_est)

    p0 = [f_guess, xi+.5, yi+.5]
    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    try:
        popt, pcov = curve_fit(model.evaluate, (xf,yf),
                               np.ravel(cutout)-bg_est, p0=p0,
                               sigma=np.ravel(np.sqrt(np.abs(cutout))))

        resid = cutout - bg_est - model.evaluate((xf,yf), *popt).reshape(5,5)
        q = np.sum(np.abs(resid))/popt[0]
        cx = resid[2,2]/popt[0]
    except (RuntimeError, ValueError):
        popt = [np.nan, np.nan, np.nan, np.nan]
        q = np.nan
        cx = np.nan
    f, x, y = popt
    return f, x, y, q, cx


# def do_stars(xs, ys, skies, mod, data):
#     # mod_func = set_mod(mod)
#     flat_model = FlattenedModel(mod)
#     xfit, yfit, ffit, qfit, cxs = [], [], [], [], []
#     for x,y,sky in zip(xs, ys, skies):
#         ff, xf, yf, qf, cxf = fit_star(float(x), float(y), sky,
#                               flat_model, data)
#         ffit.append(ff)
#         xfit.append(xf)
#         yfit.append(yf)
#         qfit.append(qf)
#         cxs.append(cxf)
#     ffit = np.array(ffit)
#     m = -2.5 * np.log10(ffit)
#
#     tbl = Table([xfit, yfit, list(m), qfit, skies, cxs],
#                 names=['x', 'y', 'm', 'q', 's', 'cx'])
#     return tbl

def do_stars_mp(xs, ys, skies, mod, data, ncpu):

    flat_model = FlattenedModel(mod)
    fit_func = partial(fit_star, model=flat_model, im_data=data)
    xy_sky = zip(xs, ys, skies)
    # start = perf_counter()
    if ncpu is None:
        p = Pool(cpu_count())
    else:
        p = Pool(ncpu)

    result = p.starmap(fit_func, xy_sky)
    p.close()
    p.join()
    # end = perf_counter()
    # print(end-start)

    result = np.array(result)
    tbl = Table(result, names=['m', 'x', 'y', 'q', 'cx'])
    tbl['s'] = skies
    tbl['m'] = -2.5 * np.log10(tbl['m'])
    tbl = tbl['x', 'y', 'm', 'q', 's', 'cx']
    return tbl
