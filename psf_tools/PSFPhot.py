import glob
import matplotlib.pyplot as  plt
import numpy as np
import os
import subprocess

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from bisect import bisect_left
from skimage.draw import polygon

from CatalogUtils import create_output_wcs, make_chip_catalogs, make_tweakreg_catfile, rd_to_refpix
from MatchUtils import get_match_indices, make_id_list

def align_images(input_images, reference_catalog=None,
                 searchrad=None, cat_extension='.xympqk'):
    """
    Run TweakReg on the images, using the catalogs from hst1pass.

    Astrometrically align the input images via TweakReg.  Uses the
    catalogs from hstpass as the input positions of the sources.  Can
    optionally align images to a reference frame by giving a reference
    catalog such as Gaia.  For more information, see the TweakReg
    documentation.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).
    reference_catalog : str, optional
        File name of the reference catalog.  If None images are aligned
        to first image in input_images.
    searchrad : float, optional
        Search radius in arcseconds, if not given, uses tweakreg
        default of 1 arcsec.
    cat_extension : str, optional
        Extension of the hst1pass catalog filenames.  Only changes if
        different outputs were requested in hst1pass step.
    """

    from drizzlepac import tweakreg
    input_catalogs = [
                      im.replace('.fits', cat_extension)
                      for im in input_images]
    make_chip_catalogs(input_catalogs)
    make_tweakreg_catfile(input_images)
    tweakreg.TweakReg(input_images, catfile='tweakreg_catlist.txt',
                      refcat=reference_catalog, searchrad=searchrad,
                      interactive=False, updatehdr=True, shiftfile=True)

def collate(match_arr, tbls):
    """
    Averages measurements taken from each input image for matched stars

    This function calculates the average and standard deviation of the
    magnitude, RA, Dec, Q parameter, and reference frame X and Y positions
    for each star in the peakmap (that was detected at least the minimum number of
    times).  It also outputs the number of times that star was detected.
    Future enhancements include: making a mbar/mstd column for each filter,
    and adding rejection into computation.

    Parameters
    ----------
    match_arr : numpy.ndarray
        Array with one row for each star, 1 column for each input catalog
        The elements should be the index of the star's row in each input table
        or -1 if the star is not in that input table.
    tbls : list
        List of Astropy tables.  One table for each input catalog.

    Returns
    -------
    final_tbl : astropy.table.Table
        Final averaged photometric catalog.  See the docstring to see
        description of the columns.
    """
    big_shape = match_arr.T.shape
    mags, rs, ds, qs, xs, ys = np.zeros((6, big_shape[0], big_shape[1]),
                                        dtype=np.float64)
    ns = np.zeros(len(match_arr.T), dtype=int)
    n_images = len(tbls)

    arrays = [mags, rs, ds, qs, xs, ys]

    for j, row in enumerate(match_arr.T):
        for i, element in enumerate(row):
            if element == -1:
                for arr in arrays:
                    arr[j,i] = np.nan
            elif element != -1:
                mags[j,i] = tbls[i]['m'][element]
                rs[j,i] = tbls[i]['r'][element]
                ds[j,i] = tbls[i]['d'][element]
                qs[j,i] = tbls[i]['q'][element]
                xs[j,i] = tbls[i]['rx'][element]
                ys[j,i] = tbls[i]['ry'][element]
        ns[j] = sum(~np.isnan(mags[j]))


    final_tbl = Table()
    final_tbl['mbar'] = np.nanmean(mags, axis=1)
    final_tbl['rbar'] = np.nanmean(rs, axis=1)
    final_tbl['dbar'] = np.nanmean(ds, axis=1)
    final_tbl['qbar'] = np.nanmean(qs, axis=1)
    final_tbl['xbar'] = np.nanmean(xs, axis=1)
    final_tbl['ybar'] = np.nanmean(ys, axis=1)
    final_tbl['mstd'] = np.nanstd(mags, axis=1)
    final_tbl['rstd'] = np.nanstd(rs, axis=1)
    final_tbl['dstd'] = np.nanstd(ds, axis=1)
    final_tbl['qstd'] = np.nanstd(qs, axis=1)
    final_tbl['xstd'] = np.nanstd(xs, axis=1)
    final_tbl['ystd'] = np.nanmean(ys, axis=1)
    final_tbl['n'] = ns

    return final_tbl

def make_coverage_map(input_images, ref_wcs):
    """
    Creates coverage map of input images in reference frame

    This will soon be expanded to be used with the peak map for
    source selection for the final averaging.  Dividing the peak map
    by the coverage map gives the fractional detection percentage.
    """

    coverage_image = np.zeros(output_wcs._naxis[::-1], dtype=int)

    for hw in hst_wcs_list:
        vx, vy = ref.all_world2pix(hw.calc_footprint(), 0).T
        poly_xs, poly_ys = polygon(vx, vy)
        coverage_image[poly_ys, poly_xs] += 1
    return coverage_image

def make_final_table(input_images, save_peakmap=True, min_detections=3):
    """
    Wrapper for final photometric matching and averaging.

    This function wraps around several utility functions to match,
    organize, and average the hst1pass photometric sources to create
    a final catalog.  It does so by:
    1. projecting all sources detected in the input images
    into a final reference frame creating a map of the detections
    2. Matching sources from the catalogs corresponding to the input
    images
    3. Calculates the statistics of photometric quantities from the
    measurements in the catalogs for each source detected at least
    the number of times specified by the min_detections parameter.
    If saved, the peakmap shows how many times a source was detected
    at certain pixel of the final reference frame in the catalogs.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).
    save_peakmap : bool, optional
        Flag to save the map of detections (peakmap).  Default True
    min_detections : int, optional
        The minimum number of images a source must be detected in
        to be included in the final catalog

    Returns
    -------
    final_catalog : astropy.table.Table
        Final averaged photometric catalog.  See documentation of
        collate() for more information.
    """
    outwcs = create_output_wcs(input_images)
    peakmap, all_int_coords = make_peakmap(input_images,
                                           outwcs,
                                           save_peakmap=save_peakmap)

    input_catalogs = []
    filters = []

    for f in input_images:
        filt = fits.getval(f, 'FILTER')
        cat_wildcard = f.replace('.fits', '_sci?_xyrd.cat')
        input_catalogs += sorted(glob.glob(cat_wildcard))
        filters += [filt] * len(input_catalogs)
    final_catalog = process_peaks(peakmap, all_int_coords,
                                  input_catalogs, outwcs,
                                  filters,
                                  min_detections=min_detections)
    return final_catalog

def make_peakmap(input_images, ref_wcs, save_peakmap=True):
    """
    Creates reate the peak map in the reference frame.

    Creates the peak map, that is the number of times a source
    is detected across the catalogs from the input images. The
    image size and orientation is derived from the ref_wcs parameter.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).
    ref_wcs : astropy.wcs.WCS
        WCS object to of the reference frame convert sky to pixel
        positions, and get the dimensions of the final reference frame
    save_peakmap : bool, optional
        Flag to save the map of detections (peakmap).  Default True

    Returns
    -------
    peakmap : numpy.ndarray
        The map of source detections
    all_int_coords : list
        List of numpy arrays containing integer coordinates of pixel
        positions in the reference frame.
    """
    all_int_coords, input_catalogs = [], []
    for f in input_images:
        filt = fits.getval(f, 'FILTER')
        cat_wildcard = f.replace('.fits', '_sci?_xyrd.cat')
        input_catalogs += sorted(glob.glob(cat_wildcard))
        for cat in input_catalogs:
            tmp = rd_to_refpix(cat, ref_wcs)
            all_int_coords.append(tmp)
        all_int_coords = np.array(all_int_coords)

    peakmap = np.zeros((ref_wcs._naxis2, ref_wcs._naxis1), dtype=int)
    for coord_list in all_int_coords:
        peakmap[coord_list[:,1], coord_list[:,0]] += 1

    if save_map:
        pri_hdu = fits.PrimaryHDU()
        im_hdu = fits.hdu.ImageHDU(data=peakmap, header=ref_wcs.to_header())
        hdul = fits.HDUList([pri_hdu, im_hdu])
        hdul.writeto('python_pkmap.fits', overwrite=True)

    return peakmap, all_int_coords

def process_peaks(peakmap, all_int_coords, input_cats,
                  ref_wcs, filter_list, min_detections=3):
    """
    Analyzes peaks in the peak map, matches peaks with catalogs, and averages

    Workhorse function for matching and averaging sources from the input
    catalogs.  This function should only be used if having slightly more
    control over the reference frame is desired.  Otherwise, use the
    make_final_table function.

    Parameters
    ----------
    peakmap : numpy.ndarray
        Peak map as described in make_peakmap() returned value
    all_int_coords : list
        List of ndarrays as described in make_peakmap() returned value
    input_cats : list
        List of filenames of catalogs to match/collate.  Order must be
        the same as ordering of all_int_coords
    ref_wcs : astropy.wcs.WCS
        WCS object to of the reference frame convert sky to pixel
        positions, and get the dimensions of the final reference frame
    filter_list : list
        List of filter corresponding to each input catalog
    min_detections : int, optional
        The minimum number of images a source must be detected in
        to be included in the final catalog

    Returns
    -------
    final_catalog : astropy.table.Table
        Final averaged photometric catalog.  See documentation of
        collate() for more information.
    """


    print('\nMatching stars from input images with peaks in peakmap')
    match_ints = np.where(peakmap.T>=min_detections)
    match_ids = make_id_list(match_ints)

    res = []
    for coord_block in all_int_coords:
        tmp_input_ids = make_id_list(coord_block.T, 4, 4)
        tmp_matches = get_match_indices(match_ids, tmp_input_ids)
        res.append(tmp_matches)

    res = np.array(res)

    tbls = []
    colnames = ['x', 'y', 'r', 'd', 'm', 'q']
    for i, cat in enumerate(input_cats):
#         print cat
        tmp_tbl = Table.read(cat, names=colnames, format='ascii.no_header')
        tmp_tbl.meta['filter'] = filter_list[i]
        rx, ry = ref_wcs.all_world2pix(np.array([tmp_tbl['r'],tmp_tbl['d']]).T, 1).T
        tmp_tbl['rx'] = rx
        tmp_tbl['ry'] = ry
        tbls.append(tmp_tbl)

    print('\nFinal step: collating properties of matched stars')
    final_tbl = collate(res, tbls)
    return final_tbl


def run_hst1pass(input_images, hmin=5, fmin=1000, pmax=99999, out='xympqk', executable_path=None):
    """Run hst1pass on set of images"""
    if not executable_path:
        executable_path = '.'
    if not executable_path.endswith('hst1pass.e'):
        executable_path = os.path.join(executable_path, 'hst1pass.e')

    if type(images) != str:
        try:
            images = ' '.join(images)
        except:
            raise ValueError('Could not interpret inputs.  First argument must either be a string or list of images')


    cmd = '{} HMIN={} FMIN={} PMAX={} OUT={} {}'.format(executable_path, hmin, fmin, pmax, out, images)
    print cmd
    os.system(cmd)
