import astropy.units as u
import glob
import numpy as np
import os
import subprocess

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.table import Table
from astropy.units import Quantity
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from drizzlepac.wcs_functions import make_perfect_cd
from scipy.stats import sigmaclip
from skimage.draw import polygon
from stwcs.wcsutil.hstwcs import HSTWCS
from stwcs.distortion import utils

from photometry_tools import photometry

def get_apcorr(data, cat):
    """
    Takes data array and matching PSF phot catalog and computes
    aperture correction.

    The PSF magnitudes are not measurements of all the light in the
    full extent of the psf, but roughly the innermost 5 pixels.
    This function performs 10 pixel aperture photometry and computes
    a clipped median offset between the aperture magnitudes and the
    PSF magnitudes to get a correction to the calibrated 10 pixel
    aperture, while maintaining the precision of PSF measurements.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing image data (SCI extension)
    cat : str
        Filename of psf catalog for the image data. Necessary for
        getting positions of stars to measure,

    Returns
    -------
    ap_corr : float
        Aperture correction from PSF mag to 10 pixel aperture mag.
    """
    t = Table.read(cat, format='ascii.commented_header')
    ap_t = photometry(data, coords=np.array([t['x'], t['y']]), salgorithm='median',
                      radius=10., annulus=10., dannulus=3., origin=1., )
    delta = t['m'] - ap_t['mag']
    # print(np.nanmedian(delta))
    nonzero_q = (t['q'] > 0)
    q_perc = np.nanpercentile(t['q'][nonzero_q], 20)
    qmask = nonzero_q & (t['q']<q_perc)
    ap_merr_perc = np.nanpercentile(ap_t['mag_error'][nonzero_q], 15)
    ap_mask = ap_t['mag_error'] < ap_merr_perc
    mask = qmask & ap_mask
    clip = delta[mask]
    ap_corr = np.nanmedian(clip, )
    n = len(clip)
    print('Computed aperture correction of {}\
           using {} stars for {}'.format(ap_corr, n, cat))
    return ap_corr

def get_ext_wcs(image_name, sci_ext=None):
    """
    Gets WCS for image extension sci_ext

    Parameters
    ----------
    image_name : str
        Name of the image from which to get the WCS.
    sci_ext : int
        Which science extension to pull WCS from (either 1 or 2)
    """

    hdu = fits.open(image_name)
    ext_names = [ext.name for ext in hdu]
    n_sci = ext_names.count('SCI')

    if n_sci == 1 and sci_ext is None:
        sci_ext = 1
        wcs_i = WCS(hdu['SCI', sci_ext].header, hdu)
    elif n_sci != 1 and sci_ext is None:
        hdu.close()

        raise ValueError('More than one sci ext detected \
                         so sci_ext cannot be None, must be \
                         directly specified in call to get_ext_wcs')
    else:
        wcs_i = WCS(hdu['SCI', sci_ext].header, hdu)
    hdu.close()
    return wcs_i

def make_chip_catalogs(input_catalogs):
    """
    Breaks up hst1pass catalogs and saves per chip catalogs.

    hst1pass outputs catalogs where the measurements from both chips
    are merged together.  This splits that catalog into 1 catalog for
    each chip, and resets the y coordinate for chip one (in merged
    catalog, chip 1 starts at y=2048, in separate catalogs it starts
    back at 0).  Also computes/adds RA/Dec columns into catalogs

    Parameters
    ----------
    input_catalogs : list
        List of catalogs output from hst1pass.  Typically should be
        the image filename, with '.fits' replaced with '.xympqks'.
    """

    for cat_name in input_catalogs:
        print(cat_name)
        image_root = cat_name.split('.')[0]
        image_name = image_root + '.fits'

        tmp = read_one_pass_tbl(cat_name)
        chip = tmp['k'].data.astype(int)
        inds = chip == 1 # the chip 1 indices
        tmp.remove_column('k')
        for i in set(chip):
            if i == 2: # represents sci 2, python 0 indexed
                inds = ~inds
            tbl = tmp[inds]
            tbl['y'] = tmp['y'][inds] - 2048.0 * float(i-1) # subtract chip height
            wcs_i = get_ext_wcs(image_name, i)
            make_sky_coord_cat(tbl, image_name, i, wcs_i)


def make_sky_coord_cat(tbl, image_name, sci_ext, wcs_i=None):
    """
    Adds RA/Dec column into table containing x/y columns via WCS.

    Inputs a table <tbl> with x/y values, reads a WCS from image
    <image_name> and transforms the x/y values to RA/DEC values via
    the wcs <wcs_i>.  If wcs_i not given, uses <image_name> and
    <sci_ext> to read in the WCS.  Saves the output table as
    <image_rootname>_fl[ct]_sci<sci_ext>_xyrd.cat.

    Parameters
    ----------
    tbl : astropy.table.Table
        Table containing x and y columns to be transformed
    image_name : str
        Image from which to get the WCS. Should be image that
        tbl was derived from.
    sci_ext : int
        Science extension from which to get the WCS/name the output.
        For UVIS this is either 1 or 2, for IR it is always 1.
    wcs_i : astropy.wcs.WCS or subclass, optional
        WCS object to use for transformation.  If None, the WCS is
        read in from the image_name and sci_ext given.


    """

    if wcs_i is None:

        wcs_i = get_ext_wcs(image_name, sci_ext)

    xi = tbl['x']
    yi = tbl['y']
    xyi = np.array([xi,yi]).T
    rdi = wcs_i.all_pix2world(xyi, 1)
    pos_tbl = Table([xi, yi])
    pos_tbl['r'] = rdi[:,0]
    pos_tbl['d'] = rdi[:,1]

    pcols = pos_tbl.colnames
    other_cols = [col for col in tbl.colnames if col not in pcols]

    for col in other_cols:
        pos_tbl[col] = tbl[col]

    image_root = image_name.split('.')[0]
    pos_tbl.write(image_root + '_sci{}_xyrd.cat'.format(sci_ext),
                format='ascii.commented_header', overwrite=True)

def make_tweakreg_catfile(input_images, update=False):
    """
    Makes the list of catalogs associated with each image for TweakReg

    Makes file containing the appropriate entries for TweakReg to use
    the PSF photometry catalog files.  See the TweakReg docs section
    regarding the 'catfile' parameter for further information.
    """

    catfile = open('tweakreg_catlist.txt', 'w')
    for f in input_images:
        cat_wildcard = f.replace('.fits', '_sci?_xyrd.cat')
        cats = sorted(glob.glob(cat_wildcard))
        cats_str = '\t'.join(cats)
        catfile.write('{}\t{}\n'.format(f, cats_str))
    catfile.close()

def read_one_pass_tbl(input_catalog):
    """
    Reads the hst1pass file into astropy table.

    Parameters
    ----------
    input_catalog : str
        Catalog output from hst1pass (.xympqks file)

    Returns
    -------
    tbl : astropy.table.Table
        Table containing the columns with appropriate names from
        hst1pass catalog.
    """

    penultimate_line = open(input_catalog).readlines()[-2]
    cols = penultimate_line.strip('#').replace('.', '').split()
    derp = np.loadtxt(input_catalog)
    tbl = Table(derp, names=cols)
    return tbl

def rd_to_refpix(cat, ref_wcs):
    """Convert sky to int pixel positions in the reference frame"""

    x, y, r, d, m, q = np.loadtxt(cat, unpack=True)[:6]
    refx, refy = ref_wcs.all_world2pix(np.array([r,d]).T, 1).T -.5
    return np.array([refx, refy]).astype(int).T

def update_catalogs(input_images):
    """
    Recomputes image catalogs' RA/Dec vals. Useful after aligning.

    Aligning images changes the WCS, so the transformation from
    x/y to RA/Dec must be recomputed.  This is a helper function
    to do that for multiple images and the corresponding catalogs.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).  Must be fits files.
    """

    for im in input_images:
        cat_wildcard = im.replace('.fits', '_sci?_xyrd.cat')
        cats = sorted(glob.glob(cat_wildcard))
        for i, cat in enumerate(cats):
            cat_tbl = Table.read(cat, format='ascii.commented_header')
            make_sky_coord_cat(cat_tbl, im, sci_ext=i+1)

#------------------Other utilities-------------------------------------

def create_coverage_map(input_wcss, ref_wcs):
    """
    Creates coverage map of input images in reference frame

    Parameters
    ----------
    input_wcss : list
        List of WCS's of science extensions of data.  Should
        include distortion solutions etc.
    ref_wcs : astropy.wcs.WCS
        WCS object for the final reference frame.

    Returns
    -------
    coverage_map : numpy.ndarray
        Array with same dimensions as ref_wcs._naxis, with number
        of images per pixel.
    """

    print('Computing image coverage map.')

    coverage_image = np.zeros(ref_wcs._naxis[::-1], dtype=int)

    for hwcs in input_wcss:
        vx, vy = ref_wcs.all_world2pix(hwcs.calc_footprint(), 0).T - .5
        poly_xs, poly_ys = polygon(vx, vy, coverage_image.shape[::-1])
        coverage_image[poly_ys, poly_xs] += 1
    return coverage_image

def create_output_wcs(input_images, make_coverage_map=False):
    # No Longer needed due to storing of WCS in metadata dict
    """
    Calculates a WCS for the final reference frame

    This function is used to calculate the WCS for the final reference
    frame for the photometric catalogs, peak map and coverage map. The
    WCS has the NAXIS keywords set large enough to encompass all input
    images completely.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).

    Returns
    -------
    output_wcs : HSTWCS
        WCS object for the final reference frame. Like an Astropy WCS.
    """

    hst_wcs_list = []
    for f in input_images:
        hdu = fits.open(f)
        for i, ext in enumerate(hdu):
            if 'SCI' in ext.name:
                hst_wcs_list.append(HSTWCS(hdu, ext=i))

    output_wcs = utils.output_wcs(hst_wcs_list, undistort=True)
    output_wcs.wcs.cd = make_perfect_cd(output_wcs)

    print('The output WCS is the following: ')
    print(output_wcs)
    return output_wcs

def get_gaia_cat(input_images, cat_name='gaia'):
    """
    Get the Gaia catalog for the area covered by input images.

    This function queries Gaia for a table of sources. It determines
    the dimensions to use for the query by finding the sky positions
    of the corners of each of the images.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).  Must be fits files.
    cat_name : str, optional
        What to store saved catalog as, automatically appends '.cat'.
        Default = 'gaia'

    Returns
    -------
    cat_filename: str
        The name of the saved gaia catalog.
    """


    print('Calculating coordinate ranges for Gaia query:')
    footprint_list = list(map(get_footprints, input_images))


    merged = []
    for im in footprint_list:
        for ext in im:
            merged.append(ext)
    merged = np.vstack(merged)
    ras = merged[:,0]
    decs = merged[:,1]

    ra_midpt = (max(ras)+min(ras))/2.
    dec_midpt = (max(decs)+min(decs))/2.

    ra_width = (np.amax(ras)-np.amin(ras))
    dec_height = (np.amax(decs)-np.amin(decs))

    coord = SkyCoord(ra=ra_midpt, dec=dec_midpt, unit=(u.degree, u.degree), frame='icrs')
    width = Quantity(ra_width, u.deg) * 1.1
    height = Quantity(dec_height, u.deg) * 1.1
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    assert len(r) > 0, 'No sources found in Gaia query\n'
    print('Sources returned: {}'.format(len(r)))


    cat = r['ra', 'dec']
    cat_filename = '{}.cat'.format(cat_name)
    cat.write(cat_filename, format='ascii.commented_header',
                              overwrite=True)
    return cat_filename



def get_footprints(image):
    """
    Get footprints of images while handling images with >1 chip

    Calculates sky footprint of images using the WCS's in the image.
    If the image has multiple sci extensions (and therefore multiple
    WCS's), the footprints of all extensions are calculated.

    Parameters
    ----------
    image : str
        Filename of fits image for which to calculate sky footprint(s)

    Returns
    -------
    footprints : list
        List of footprints for each science extension of image.  Each
        footprint is a list of the RA/Dec coordinates of the 4 corners
        of a single science extension.
    """

    footprints = []
    hdu = fits.open(image)
    flt_flag = 'flt.fits' in image or 'flc.fits' in image
    for ext in hdu:
        if 'SCI' in ext.name:
            hdr = ext.header
            wcs = WCS(hdr, hdu)
            footprint = wcs.calc_footprint(hdr, undistort=flt_flag)
            footprints.append(footprint)
    return footprints


def pixel_area_correction(catalog, detchip, mag_colname='m'):
    """
    Performs pixel area correction on magnitude measurements in catalog.

    The PSF magnitudes are measured in the image frame, which does not
    account for the pixel area.  This function stores the pixel area map
    as a set of 2D polynomials, and evaluates the values of those
    polynomials at each x,y in the catalogs.  Finally it applies those
    evaluated values to the magnitude measurements in the catalog.

    Note: Since the table is passed by reference, the corrected column
    does not need to be returned to updated the catalog (it's updated
    in place).

    Parameters
    ----------
    catalog : astropy.table.Table
        Table containing x, y and magnitude measurements
    detchip : str
        Detector/chip string.  I.E chip 1 of wfc3/uvis is 'uvis1'.  If
        detector only has 1 chip, then no number should be added (i.e just
        'ir' for wfc3/ir).
    mag_colname : str
        Name of the column containing the magnitudes to be corrected.
        Default is 'm'.
    """

    pam_func = get_pam_func(detchip)

    intx = np.array(catalog['x']).astype(int) - 1
    inty = np.array(catalog['y']).astype(int) - 1
    corrections = -2.5 * np.log10(pam_func(intx, inty))

    catalog['m'] += corrections

def get_pam_func(detchip):
    # TODO: Implement ACS Pixel Area Correction
    # Requires WCS of image
    degrees = {'ir':2, 'uvis1':3, 'uvis2':3,
               'wfc1':3, 'wfc2':3}

    # Store polynomial coefficient values for each chip
    coeff_dict = {}
    coeff_dict['ir'] = [9.53791038e-01, -3.68634734e-07, -3.14690506e-10,
                        8.27064384e-05, 1.48205135e-09, 2.12429722e-10]
    coeff_dict['uvis1'] = [9.83045965e-01, 8.41184852e-06, 1.59378242e-11,
                           -2.02027686e-20, -8.69187898e-06, 1.65696133e-11,
                           1.31974097e-17, -3.86520105e-11, -5.92106744e-17,
                           -9.87825173e-17]
    coeff_dict['uvis2'] = [1.00082580e+00, 8.66150267e-06, 1.61942281e-11,
                           -1.01349112e-20, -9.07898503e-06, 1.70183371e-11,
                           4.10618989e-18, -8.02421371e-11, 3.54901127e-16,
                           -1.01400681e-16]

    coeffs = coeff_dict[detchip.lower()]
    pam_func = models.Polynomial2D(degree=degrees[detchip.lower()])

    pam_func.parameters = coeffs
    return pam_func
