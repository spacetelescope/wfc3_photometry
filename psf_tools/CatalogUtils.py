import astropy.units as u
import glob
import numpy as np
import os
import subprocess

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from drizzlepac.wcs_functions import make_perfect_cd
from stwcs.wcsutil.hstwcs import HSTWCS
from stwcs.distortion import utils

def make_chip_catalogs(input_catalogs):
    """Breaks up hst1pass catalogs and saves per chip catalogs."""
    # Can probably clean up this function for better DRYness.

    for f in input_catalogs:
        print f
        image_root = f.split('.')[0]
        image_name = image_root + '.fits'
        hdu = fits.open(image_name)
        tmp = read_one_pass_tbl(f)
        x = tmp['x']
        y = tmp['y']
        m = tmp['m']
        q = tmp['q']
        chip = tmp['k'].data
        sci_1_inds = chip == 1

        x1 = x[sci_1_inds]
        y1 = y[sci_1_inds]
        m1 = m[sci_1_inds]
        q1 = q[sci_1_inds]
        wcs_sci_1 = WCS(fits.getheader(image_name, 1), hdu)
        xy1 = np.array([x1,y1]).T
        rd1 = wcs_sci_1.all_pix2world(xy1, 1)

        x2 = x[~sci_1_inds]
        y2 = y[~sci_1_inds] - 2048
        m2 = m[~sci_1_inds]
        q2 = q[~sci_1_inds]
        wcs_sci_2 = WCS(fits.getheader(image_name, 4), hdu)
        rd2 = wcs_sci_2.all_pix2world(np.array([x2,y2]).T, 1)

        hdu.close()

        xyrd1 = np.vstack([x1, y1, rd1[:,0], rd1[:,1], m1, q1]).T
        xyrd2 = np.vstack([x2, y2, rd2[:,0], rd2[:,1], m2, q2]).T
        table1 = Table(xyrd1, names=['x', 'y', 'r','d','m', 'q'])
        table2 = Table(xyrd2, names=['x', 'y', 'r','d','m', 'q'])
        table1.write(image_root + '_sci1_xyrd.cat', format='ascii.commented_header')
        table2.write(image_root + '_sci2_xyrd.cat', format='ascii.commented_header')
        # np.savetxt(image_root + '_sci1_xyrd.cat', xyrd1)
        # np.savetxt(image_root + '_sci2_xyrd.cat', xyrd2)

def make_tweakreg_catfile(input_images):
    """Makes the list of catalogs associated with each image for TweakReg"""

    catfile = open('tweakreg_catlist.txt', 'w')
    for f in input_images:
        cat_wildcard = f.replace('.fits', '_sci?_xyrd.cat')
        cats = sorted(glob.glob(cat_wildcard))
        cats_str = '\t'.join(cats)
        catfile.write('{}\t{}\n'.format(f, cats_str))
    catfile.close()

def read_one_pass_tbl(input_catalog):
    """Reads the hst1pass files into astropy table."""

    penultimate_line = open(input_catalog).readlines()[-2]
    cols = penultimate_line.strip('#').replace('.', '').split()
    derp = np.loadtxt(input_catalog)
    tbl = Table(derp, names=cols)
    return tbl

def rd_to_refpix(cat, ref_wcs):
    """Convert sky to int pixel positions in the reference frame"""

    x, y, r, d, m, q = np.loadtxt(cat, unpack=True)
    refx, refy = ref_wcs.all_world2pix(np.array([r,d]).T, 1).T -.5
    return np.array([refx, refy]).astype(int).T

#------------------Other utilities-------------------------------------

def create_output_wcs(input_images):
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
        hw1 = HSTWCS(f, ext=1)
        hw2 = HSTWCS(f, ext=4)
        hst_wcs_list.append(hw1)
        hst_wcs_list.append(hw2)
    output_wcs = utils.output_wcs(hst_wcs_list, undistort=True)
    output_wcs.wcs.cd = make_perfect_cd(output_wcs)

    print 'The output WCS is the following: '
    print output_wcs
    return output_wcs

def get_gaia_cat(input_images, cat_name='gaia'):
    """
    Get the Gaia catalog for the area of input images.

    This function queries Gaia for a table of sources. It determines
    the dimensions to use for the query by finding the sky positions
    of the corners of each of the images.

    """


    print('Calculating coordinate ranges for Gaia query:')
    footprint_list = map(get_footprints, input_images)


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
    cat.write('{}.cat'.format(cat_name), format='ascii.commented_header')
    return '{}.cat'.format(cat_name)



def get_footprints(im):
    """Get footprints of images while handling images with >1 chip"""
    fps = []
    hdu = fits.open(im)
    flt_flag = 'flt.fits' in im or 'flc.fits' in im
    for ext in hdu:
        if 'SCI' in ext.name:
            hdr = ext.header
            wcs = WCS(hdr, hdu)
            fp = wcs.calc_footprint(hdr, undistort=flt_flag)
            fps.append(fp)
    return fps
