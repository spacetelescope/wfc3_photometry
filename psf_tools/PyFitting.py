import math
import numpy as np

from functools import partial
from itertools import product
from multiprocessing import cpu_count, Pool
from time import perf_counter

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.psf import FittableImageModel
from photutils.psf.models import GriddedPSFModel
from scipy.ndimage import convolve, maximum_filter
from scipy.optimize import curve_fit

from photometry_tools import aperture_stats_tbl
from .CatalogUtils import make_sky_coord_cat
from .PSFPhot import check_images, get_standard_psf, validate_file

def run_python_psf_fitting(ims, psf_model_file=None, hmin=5, fmin=1E3,
                           pmax=7E4, qmax=.5, cmin=-1., cmax=.1,
                           ncpu=None):

    filt = check_images(ims)
    if psf_model_file is None:
        psf_model_file = get_standard_psf('./', filt)
    det = fits.getval(ims[0], 'DETECTOR')
    exts = [1] if det == 'IR' else [1,2]

    mods = make_models(psf_model_file)
    for im in ims:
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

def measure_stars(data, mod, hmin, fmin=1E3, pmax=7E4,
                  qmax=.5, cmin=-1., cmax=.1, ncpu=None):
    filt_image, max_4sum = filter_images(data, hmin)
    xs, ys = find_sources(data, filt_image, max_4sum, fmin, pmax)
    skies = estimate_all_backgrounds(xs, ys, 8.5, 13.5, data)

    # Give buffer of 10% for peak pixel tolerance
    max_peak_val = max_peakiness(mod) + .1

    mask = reject_sources(xs, ys, data, max_4sum, skies, max_peak_val)
    xs = xs[mask]
    ys = ys[mask]
    skies = skies[mask]

    # tbl = do_stars(xs, ys, skies, mod, data)
    tbl = do_stars_mp(xs, ys, skies, mod, data, ncpu)
    tbl = tbl[tbl['q']<qmax]
    tbl = tbl[tbl['cx']<cmax]
    tbl = tbl[tbl['cx']>cmin]
    old_len = len(xs)
    new_len = len(tbl)
    rej = old_len - new_len
    print('Rejected {} more sources after qmax and excess clip'.format(rej))
    return tbl
#---------------------------DETECTION--------------------------------

def conv_origin(data, origin):
    '''Helper function to shift around conv kernel for peak finding'''
    return convolve(data, weights=np.ones((2,2)),
                    mode='constant', cval=0,
                    origin=origin)

def filter_images(data, hmin):

    #Laziest way to get a circle mask
    fp = CircularAperture((0,0), r=hmin).to_mask()[0].data>.1
    fp = fp.astype(bool)

    # Apply maximum filter, flux filter
    filt_image = maximum_filter(data, footprint=fp,
                                mode='constant', cval=0)
    origins = product([0,-1], [0,-1])
    max_4sum = np.amax([conv_origin(data, o) for o in origins], axis=0)
    return(filt_image, max_4sum)

def find_sources(data, filt_image, max_4sum, fmin=1E3, pmax=7E4):
    yi, xi = np.where(data>=filt_image)
    mask1 = (max_4sum[yi, xi] > fmin) & (data[yi,xi] < pmax)
    mask2 = (xi>2) & (yi>2) & (xi<data.shape[1]-3) & (yi<data.shape[0]-3)
    mask = mask1 & mask2
    yi = yi[mask]
    xi = xi[mask]
    return(xi, yi)

def max_peakiness(mod):
    yg, xg = np.mgrid[-2:3,-2:3]
    origins = product([0,-1], [0,-1])
    peak_ratios = []
    for xc, yc in mod.grid_xypos:
        ev = mod.evaluate(xg+xc, yg+yc, 1., xc, yc)
        peak_flux = np.amax(conv_origin(ev,[0,0]))
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

def make_models(psf_file):
    hdu = fits.open(psf_file)
    psf_data = hdu[0].data
    hdr = hdu[0].header

    xlocs = [hdr['IPSFX'+str(i).zfill(2)] for i in range(1,11)]
    xlocs = np.array([xloc for xloc in xlocs if xloc != 9999]) -1

    ylocs = [hdr['JPSFY'+str(i).zfill(2)] for i in range(1,11)]
    ylocs = np.array([yloc for yloc in ylocs if yloc != 9999]) -1

    if len(ylocs >4):   # 2 chips
        ylocs1 = ylocs[:4]
        ylocs2 = ylocs[4:]-2048

        g_xypos1 = [p[::-1] for p in product(ylocs1, xlocs)]
        g_xypos2 = [p[::-1] for p in product(ylocs2, xlocs)]

        ndd1 = NDData(data=psf_data[:28],
                      meta={'grid_xypos':g_xypos1, 'oversampling':4})
        mod1 = GriddedPSFModel(ndd1)

        ndd2 = NDData(data=psf_data[28:],
                      meta={'grid_xypos':g_xypos2, 'oversampling':4})
        mod2 = GriddedPSFModel(ndd2)

        return(mod1, mod2)

    else:
        g_xypos = [p[::-1] for p in product(ylocs, xlocs)]
        ndd1 = NDData(data=psf_data,
                      meta={'grid_xypos':g_xypos, 'oversampling':4})
        mod1 = GriddedPSFModel(ndd1)
        return(mod1, None)

def estimate_all_backgrounds(xs, ys, r_in, r_out, data, stat='aperture_mode'):
    '''Compute sky values around (xs, ys) in data with various parameters'''
    ans = CircularAnnulus(positions=(xs, ys), r_in=r_in, r_out=r_out)
    bg_ests = aperture_stats_tbl(apertures=ans, data=data, sigma_clip=True)
    return np.array(bg_ests[stat])

# def estimate_background(xi, yi, r_in, r_out, im_data):
#     an = CircularAnnulus(positions=(xi, yi), r_in=r_in, r_out=r_out)
#     bg_est = aperture_stats_tbl(apertures=an, data=im_data, sigma_clip=True)['aperture_mode'][0]
#     return bg_est

class FlattenedModel:
    '''This is a wrapper class to return the evaluated psf as a 1d array'''
    def __init__(self, psf_fittable_model):
        self.mod = psf_fittable_model

    def evaluate(self, x_y, flux, x_0, y_0):
        '''Evaluate the model, and flatten the output'''
        x, y = x_y
        return np.ravel(self.mod.evaluate(x, y, flux=flux,
                                          x_0=x_0, y_0=y_0))


def fit_star(xi, yi, bg_est, model, im_data):
    yg, xg = np.mgrid[-2:3,-2:3]
    yf, xf = yg+int(yi+.5), xg+int(xi+.5)

    cutout = im_data[yf, xf]
    f_guess = np.sum(cutout-bg_est)

    p0 = [f_guess, xi+.5, yi+.5]
    try:
        popt, pcov = curve_fit(model.evaluate, (xf,yf),
                               np.ravel(cutout)-bg_est, p0=p0,
                               sigma=np.ravel(np.sqrt(np.abs(cutout))))

        resid = cutout - bg_est - model.evaluate((xf,yf), *popt).reshape(5,5)
        q = np.sum(np.abs(resid))/popt[0]
        cx = resid[2,2]/popt[0]
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan]
        q = np.nan
        cx = np.nan
    f, x, y = popt
    return f, x, y, q, cx


def do_stars(xs, ys, skies, mod, data):
    # mod_func = set_mod(mod)
    flat_model = FlattenedModel(mod)
    xfit, yfit, ffit, qfit, cxs = [], [], [], [], []
    for x,y,sky in zip(xs, ys, skies):
        ff, xf, yf, qf, cxf = fit_star(float(x), float(y), sky,
                              flat_model, data)
        ffit.append(ff)
        xfit.append(xf)
        yfit.append(yf)
        qfit.append(qf)
        cxs.append(cxf)
    ffit = np.array(ffit)
    m = -2.5 * np.log10(ffit)

    tbl = Table([xfit, yfit, list(m), qfit, skies, cxs],
                names=['x', 'y', 'm', 'q', 'sky', 'cx'])
    return tbl

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
    tbl['sky'] = skies
    tbl['m'] = -2.5 * np.log10(tbl['m'])
    tbl = tbl['x', 'y', 'm', 'q', 'sky', 'cx']
    return tbl

#---------------------------SUBTRACTION--------------------------------

def compute_cutout(x, y, flux, mod, shape):
    """Gets cutout of evaluated model, and indices to place cutout"""

    # Get integer pixel positions of centers, with appropriate
    # edge convention, i.e. edges of pixels are integers, 0 indexed
    x_cen = int(x-.5)
    y_cen = int(y-.5)

    # Handle cases where model spills over edge of data array
    x1 = max(0, x_cen - 10)
    y1 = max(0, y_cen - 10)

    # upper bound is exclusive, so add 1 more
    x2 = min(x_cen + 11, shape[1])
    y2 = min(y_cen + 11, shape[0])

    y_grid, x_grid = np.mgrid[y1:y2, x1:x2]
    cutout = mod.evaluate(x_grid, y_grid, flux, x-1., y-1.)
    return cutout, x_grid, y_grid

def subtract_psfs(data, cat, mod):
    """Subtracts the fitted PSF from the positions in catalog"""

    # Initialize the array to be subtracted
    subtrahend = np.zeros(data.shape, dtype=float)
    dimensions = data.shape

    fluxes = np.power(10, cat['m']/-2.5) # Convert from mags to fluxes

    # Evaluate the PSF at each x, y, flux, and place it in subtrhend
    for i, row in enumerate(cat):
        flux = fluxes[i]
        if flux == np.nan:
            continue
        cutout, x_grid, y_grid = compute_cutout(row['x'], row['y'],
                                                   flux, mod,
                                                   dimensions)
        # Important: use += to account for overlapping cutouts
        subtrahend[y_grid, x_grid] += cutout

    # Subtact the image!
    difference = data - subtrahend
    return difference
