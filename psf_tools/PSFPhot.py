import glob
import inspect
import numpy as np
import os
import shutil
import subprocess
import sys
import urllib

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.wcs import WCS
from bisect import bisect_left
from drizzlepac import astrodrizzle, tweakreg
from drizzlepac.wcs_functions import make_perfect_cd
from skimage.draw import polygon
from stwcs.distortion import utils
from stwcs.wcsutil.hstwcs import HSTWCS


from .CatalogUtils import create_output_wcs, make_chip_catalogs, \
    make_tweakreg_catfile, rd_to_refpix, get_gaia_cat, create_coverage_map, \
    pixel_area_correction, update_catalogs, get_apcorr
from .MatchUtils import get_match_indices, make_id_list


def align_images(input_images, reference_catalog=None,
                 searchrad=1.0, gaia=False, **kwargs):
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
        List of image filenames to align using xyrd catalogs (strings).
        Must be fits files.
    reference_catalog : str, optional
        File name of the reference catalog.  If None images are aligned
        to first image in input_images.
    searchrad : float, optional
        Search radius in arcseconds, if not given, uses tweakreg
        default of 1 arcsec.
    gaia : bool
        Align images to Gaia?  If True, queries Gaia in region
        encompassing input images, and saves catalog of Gaia source
        positions, and uses this catalog in TweakReg
    **kwargs : keyword arguments, optional
        Other keyword arguments to be passed to TweakReg.  Examples
        could include minflux, maxflux, fluxunits.  More information
        available at
        https://drizzlepac.readthedocs.io/en/latest/tweakreg.html
    """



    make_tweakreg_catfile(input_images)
    if gaia:
        if reference_catalog:
            print('Gaia set to true, overriding reference catalog \
                    with Gaia catalog')
        reference_catalog = get_gaia_cat(input_images)
    tweakreg.TweakReg(input_images, catfile='tweakreg_catlist.txt',
                      refcat=reference_catalog, searchrad=searchrad,
                      interactive=False, updatehdr=True, shiftfile=True,
                      reusename=True, clean=True, **kwargs)

    print('Updating Catalogs with new RA/Dec')
    update_catalogs(input_images)

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
    exptimes = np.array([tbl.meta['exptime'] for tbl in tbls])
    apcorrs = np.array([tbl.meta['apcorr'] for tbl in tbls])
    photflams = np.array([tbl.meta['photflam'] for tbl in tbls])
    for tbl in tbls:
        pixel_area_correction(tbl, tbl.meta['detchip'])
    print('Pixel Area Map correction complete')

    arrays = [mags, rs, ds, qs, xs, ys]
    print(match_arr.shape)

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

    mags += 2.5 * np.log10(exptimes)[None, :] - 21.1 - 2.5*np.log10(photflams)[None, :]
    if 'ir' in tbls[0].meta['detchip'].lower():
        mags -= 2.5 * np.log10(exptimes)[None, :]
    mags -= apcorrs[None, :]
    print('Median aperture_correction {}'.format(np.nanmedian(apcorrs)))
    np.savetxt('mags.txt', mags)
    np.savetxt('qs.txt', qs)

    print('Clipping the fit quality')
    clipped_q = sigma_clip(qs, sigma=2.5, axis=1, copy=True)
    clip_mask = clipped_q.mask

    orig_nans = np.sum(np.isnan(mags))

    for array in arrays:
        array[clip_mask] = np.nan

    total_nans = np.sum(np.isnan(mags))
    print('Rejected {} measurements'.format(total_nans-orig_nans))

    print('Performing zeropoint normalization')
    # Get the top 10% of unsaturated stars
    qbar = np.nanmean(qs, axis=1)
    mbar = np.nanmean(mags, axis=1)
    good_qs = qbar > 0.
    upper_mag = np.percentile(mbar[good_qs], 10.)
    mag_mask = np.logical_and(good_qs, mbar < upper_mag)

    # Find the offsetof each image from the mean mag for each star
    offsets = mags - mbar[:,None]
    meds = np.nanmedian(offsets[mag_mask], axis = 0)
    # print(meds)

    #Subtract the median
    mags -= meds[None, :]

    final_tbl = Table()
    final_tbl['mbar'] = np.nanmean(mags, axis=1)
    final_tbl['rbar'] = np.nanmean(rs, axis=1)
    final_tbl['dbar'] = np.nanmean(ds, axis=1)
    final_tbl['qbar'] = qbar
    final_tbl['xbar'] = np.nanmean(xs, axis=1)
    final_tbl['ybar'] = np.nanmean(ys, axis=1)
    final_tbl['mstd'] = np.nanstd(mags, axis=1)
    final_tbl['rstd'] = np.nanstd(rs, axis=1)
    final_tbl['dstd'] = np.nanstd(ds, axis=1)
    final_tbl['qstd'] = np.nanstd(qs, axis=1)
    final_tbl['xstd'] = np.nanstd(xs, axis=1)
    final_tbl['ystd'] = np.nanstd(ys, axis=1)
    final_tbl['n'] = np.sum(~np.isnan(mags), axis=1)

    return final_tbl


def make_final_table(input_images, save_peakmap=True, min_detections=3,
                     drizzle=False):
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
        List of image filenames (strings).  Must be fits files.
    save_peakmap : bool, optional
        Flag to save the map of detections (peakmap).  Default True
    min_detections : int or float, optional
        If int: minimum number of images a source must be detected in
        to be included in the final catalog.
        If float: minimum fraction of images a source must be detected
        in to be included in the final catalog.
        Default is 3.
    drizzle : bool, optional
        Make a drizzled image on the same grid as the reference frame?
        Default False.  The image output from this is not optimized so
        it should only be treated as a simple preview.

    Returns
    -------
    final_catalog : astropy.table.Table
        Final averaged photometric catalog.  See documentation of
        collate() for more information.
    """
    # Restructure this to read in tables, get meta info, as well as
    # HSTWCS objects to flexibly create output WCS, cov map

    input_catalogs = []
    metas = {'filters' : [], 'exptimes' : [], 'detchips' : [],
             'wcss' : [], 'apcorrs' : [], 'photflams' : []}
    for f in input_images:
        hdr = fits.getheader(f)
        if hdr['INSTRUME'] == 'ACS':
            photflam = fits.getval(f, 'PHOTFLAM', ext=1)
            filt = hdr['FILTER1']
            if filt == 'CLEAR1L' or filt == 'CLEAR1S':
                filt = hdr['FILTER2']
        else:
            filt = hdr['FILTER']
            photflam = hdr['PHOTFLAM']
        cat_wildcard = f.replace('.fits', '_sci?_xyrd.cat')
        im_cats = sorted(glob.glob(cat_wildcard))
        exptime = hdr['EXPTIME']


        input_catalogs += im_cats
        metas['filters'] += [filt] * len(im_cats)
        metas['exptimes'] += [exptime] * len(im_cats)
        metas['photflams'] += [photflam] * len(im_cats)
        for i, cat in enumerate(im_cats):
            im_data = fits.getdata(f, ('sci', i+1))
            metas['apcorrs'].append(get_apcorr(im_data, cat))

        det = [hdr['DETECTOR']]
        det0 = det[0]
        if det0 != 'IR':
            # Determine which chip for UVIS or WFC (needed for subarray)
            det = [det0 + str(fits.getval(f, 'CCDCHIP', 1))]
            wcslist = [HSTWCS(f, ext=('sci', 1))]
            if len(im_cats) == 2:
                det += [det0 + str(fits.getval(f, 'CCDCHIP', ('sci', 2)))]
                wcslist += [HSTWCS(f, ext=('sci', 2))]
        else:
            wcslist = [HSTWCS(f, ext=('sci', 1))]
        metas['detchips'] += det
        metas['wcss'] += wcslist


    outwcs = utils.output_wcs(metas['wcss'], undistort=True)
    outwcs.wcs.cd = make_perfect_cd(outwcs)

    peakmap, all_int_coords = make_peakmap(input_images,
                                           outwcs,
                                           save_peakmap=save_peakmap)


    cov_map = create_coverage_map(metas['wcss'], outwcs)


    pri_hdu = fits.PrimaryHDU()
    im_hdu = fits.hdu.ImageHDU(data=cov_map, header=outwcs.to_header())
    hdul = fits.HDUList([pri_hdu, im_hdu])
    hdul.writeto('python_coverage_map.fits', overwrite=True)




    final_catalog = process_peaks(peakmap, all_int_coords,
                                  input_catalogs, outwcs,
                                  metas, cov_map,
                                  min_detections=min_detections)

    final_catalog.write('{}_final_cat.txt'.format(filt),
                        format='ascii.commented_header',
                        overwrite=True)


    if drizzle:
        simple_drizzle(input_images, cov_map,
                       'python_coverage_map.fits', filt)

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
        List of image filenames (strings).  Must be fits files.
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
        # Note sure why this is in here....fix later if needed
        # if fits.getval(f, 'INSTRUME') == 'ACS':
        #     hdr = fits.getheader(f)
        #     filt = hdr['FILTER1']
        #     if filt == 'CLEAR1L' or filt == 'CLEAR1S':
        #         filt = hdr['FILTER2']
        # else:
        #     filt = fits.getval(f, 'FILTER')
        cat_wildcard = f.replace('.fits', '_sci?_xyrd.cat')
        input_catalogs = sorted(glob.glob(cat_wildcard))
        for cat in input_catalogs:
            tmp = rd_to_refpix(cat, ref_wcs)
            all_int_coords.append(tmp)

    all_int_coords = np.array(all_int_coords)

    peakmap = np.zeros((ref_wcs._naxis[::-1]), dtype=int)
    for coord_list in all_int_coords:
        peakmap[coord_list[:,1], coord_list[:,0]] += 1

    if save_peakmap:
        pri_hdu = fits.PrimaryHDU()
        im_hdu = fits.hdu.ImageHDU(data=peakmap, header=ref_wcs.to_header())
        hdul = fits.HDUList([pri_hdu, im_hdu])
        hdul.writeto('python_pkmap.fits', overwrite=True)

    return peakmap, all_int_coords

def process_peaks(peakmap, all_int_coords, input_cats,
                  ref_wcs, metas, coverage_map, min_detections=3):
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
    metas : dict
        Dictionary containing the meta information for each input
        catalog.  Each key is a type of information (such as 'filter'
        or exposure time), and the corresponding value is a list with
        one entry for each of the input catalogs.  Each entry is the
        value of the keyword for the corresponding input catalog.
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
    if min_detections < 1.:
        ratio = peakmap.astype(float)/coverage_map.astype(float)
        match_ints = np.where(ratio.T>=min_detections)

    else:
        match_ints = np.where(peakmap.T>=min_detections)
    x_digits = len(str(peakmap.shape[1]))
    y_digits = len(str(peakmap.shape[0]))
    match_ids = make_id_list(match_ints, x_digits=x_digits, y_digits=y_digits)

    res = []
    for coord_block in all_int_coords:
        tmp_input_ids = make_id_list(coord_block.T, x_digits, y_digits)
        tmp_matches = get_match_indices(match_ids, tmp_input_ids)
        res.append(tmp_matches)

    res = np.array(res)


    tbls = []
    colnames = ['x', 'y', 'r', 'd', 'm', 'q']
    for i, cat in enumerate(input_cats):
#         print cat
        tmp_tbl = Table.read(cat, format='ascii.commented_header')
        tmp_tbl.meta['filter'] = metas['filters'][i]
        tmp_tbl.meta['exptime'] = metas['exptimes'][i]
        tmp_tbl.meta['detchip'] = metas['detchips'][i]
        tmp_tbl.meta['apcorr'] = metas['apcorrs'][i]
        tmp_tbl.meta['photflam'] = metas['photflams'][i]
        tmp_tbl.meta['wcs'] = metas['wcss'][i]
        rx, ry = ref_wcs.all_world2pix(np.array([tmp_tbl['r'],tmp_tbl['d']]).T, 1).T
        tmp_tbl['rx'] = rx
        tmp_tbl['ry'] = ry
        tbls.append(tmp_tbl)

    print('\nFinal step: collating properties of matched stars')
    final_tbl = collate(res, tbls)
    final_tbl['n_expected'] = coverage_map.T[match_ints]
    return final_tbl

def simple_drizzle(images, coverage_map, wcs_file, filt):
    """
    Naively drizzle the images onto the final wcs grid

    Uses astrodrizzle to combine input images into single stacked
    image.  Uses the coverage map created in create_coverage_map to
    determine the "median depth" of the image (how many images per
    output image pixel)- useful for choosing best algorithm for
    median combine.  Output image is on same pixel grid as the
    image specified by wcs_file parameter.  The coverage map filename
    is the typical input.  This image is not an optimized drizzled
    image, and is meant to give a basic stack.  The drizzled image
    is stored as <FILT>_dr[cz].fits (drc or drz based on input
    images).

    Parameters
    ----------
    images : list
        List of image filenames to drizzle.  Must be fits files.
    coverage_map : numpy.ndarray
        File created via create_coverage_map, i.e. total number of
        input images per output pixel
    wcs_file : string
        File to read WCS from to get the output grid of the drizzled
        image.  Use the filename of the coverage map for simplicity.
    filt : string
        The filter the images were taken with.  Used for naming
        output image

    """

    # This calculates how many of the input images cover each
    # pixel of the output image grid.  Used to determine best
    # median algorithm for drizzle
    median_coverage = np.median(coverage_map[coverage_map>0])
    out_name = '{}_final'.format(filt)
    if median_coverage>4:
        med_alg = 'median'
        combine_nhigh = 1
    else:
        med_alg = 'minmed'
        combine_nhigh = 0

    # Due to bug in astrodrizzle, this can't yet be optimized
    # to use the mdriztab

    astrodrizzle.AstroDrizzle(images, clean=True, build=True,
                              context=False, preserve=False,
                              combine_type=med_alg,
                              combine_nhigh=combine_nhigh,
                              in_memory=False, final_wcs=True,
                              final_refimage=wcs_file,
                              output=out_name)


def run_hst1pass(input_images, hmin=5, fmin=1000, pmax=99999,
                 out='xympqks', executable_path=None, **kwargs):
    """
    Run hst1pass.e Fortran code on images to produce initial catalogs.

    This function runs the Fortran routine hst1pass.e, on a set of
    input images.  This is Jay Anderson's PSF fitting single pass
    photometry code.  The fortran code must first be compiled for
    this code to run.  Running this code outputs one catalog for
    each input image, and saves those files to disk.  More parameters
    from the hst1pass.e code may be added to this interface in the
    future.  The original Fortran executable can also be called from
    the command line if desired.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).  Must be fits files.
    hmin : int
        Minimum separation between stars. Default 5
    fmin : int, optional
        The minimum flux (in image units) a source must have to be
        included in the output catalogs. Default 1000.
    pmax : int, optional
        The maximum flux (in image units) a source can have in the
        peak pixel to be included in the output catalogs.  Used to
        filter out saturated sources.  Default 99999.
    out : str, optional
        The measurments to be recorded in the output catalogs.  Each
        character corresponds to one output column.  The default is
        'xympqks' which gives: x coordinate, y coordinate, instrumental
        magnitude, peak pixel value, quality of fit (0 is perfect
        fit), the chip the star was detected on (chip 1 or chip 2
        for UVIS), and the sky value.  If being used with other functions
        in this package x, y, m, q, k, and s must be included in the
        outputs.  More measurements will be supported at a later date.
    executable_path : str, optional
        The path to the hst1pass.e compiled executable.  If None (default),
        the code is assumed to be in the psf_tools directory.

    Returns
    -------
    expected_output_list : list
        List of the output catalogs from running hst1pass
    """

    if not executable_path:
        executable_path = _get_exec_path()
    if os.path.isdir(executable_path):
        exec_name = _get_exec_name()
        executable_path = os.path.join(executable_path, exec_name)
    file_dir = os.path.split(executable_path)[0]

    if type(hmin) != int:
        try:
            hmin = int(hmin)
        except:
            raise ValueError('Could not convert hmin to int, hmin\
            must be integer.')

    upper_dict = {k.upper():v for (k,v) in kwargs.items()}



    all_psf_filts = ['F105W', 'F110W', 'F125W', 'F127M', 'F139M',
                    'F140W', 'F160W', 'F225W', 'F275W', 'F336W',
                    'F390W', 'F438W', 'F467M', 'F555W', 'F606W',
                    'F775W', 'F814W', 'F850LP'] + ['F435W', 'F625W']
    focus_filts = ['F275W', 'F336W', 'F410M', 'F438W', 'F467M',
                    'F606W', 'F814W']

    filt = check_images(input_images)
    det = fits.getval(input_images[0], 'detector')

    if 'GDC' not in upper_dict.keys() or upper_dict['GDC'].upper() == 'NONE':
        upper_dict['GDC']='NONE' # Set to NONE or capitalize to NONE
    elif upper_dict['GDC'].upper() == 'AUTO':
        gdc = get_standard_gdc(file_dir, filt, det)
        upper_dict['GDC'] = gdc
        validate_file(upper_dict['GDC'])
    else:
        gdc = upper_dict['GDC']
        shutil.copy(gdc, '.')
        upper_dict['GDC'] = os.path.split(gdc)[-1]
        validate_file(upper_dict['GDC'])

    keyword_str = ' '.join(['{}={}'.format(key, val) for \
                            key, val in upper_dict.items()])


    if 'FOCUS' in keyword_str and 'PSF=' not in keyword_str:
        if filt not in focus_filts:
            raise('{} does not have a focus dependent PSF \
                    model file, cannot find focus'.format(filt))
        psf_file = get_focus_dependent_psf(file_dir, filt)
        keyword_str = '{} PSF={}'.format(keyword_str, psf_file)
    elif 'PSF=' not in keyword_str and filt in all_psf_filts:
        psf_file = get_standard_psf(file_dir, filt, det)
        keyword_str = '{} PSF={}'.format(keyword_str, psf_file)
    elif 'PSF=' in keyword_str:
        psf_file = upper_dict['PSF']

    # Check to see if the PSF file isn't broken
    validate_file(psf_file)

    if type(input_images) != str:
        try:
            input_images = ' '.join(input_images)
        except:
            raise TypeError('Could not interpret inputs. \
            First argument must either be a string or list of images')




    expected_outputs = input_images.replace('.fits', '.{}'.format(out))
    expected_output_list = expected_outputs.split()

    cmd = '{} HMIN={} FMIN={} PMAX={} OUT={} {} {}'.format(
           executable_path, hmin, fmin, pmax, out,
           keyword_str, input_images)
    print(cmd)
    run_and_print_output(cmd)
    make_chip_catalogs(expected_output_list)
    return expected_output_list

def get_standard_gdc(path, filt, det):
    """
    Checks if GDC file exists and if not downloads from WFC3 page

    NOTE:  The GDC file is likely not necessary in most cases.  These
    files should only be downloaded when deemed explicitly necessary.
    This is only the case if the you want other outputs from hst1pass
    such as pixel area correction/sky coordinate computation etc.  In
    general, this is unnecessary as python functions already handle
    those things.  Furthermore the GDC files are quite large (~300MB),
    so downloading them slows things down.

    Parameters
    ----------
    path : str
        Path where to to look for/store GDC files.
    filt : str
        Filter for which to get the GDC file.

    Returns
    -------
    gdc_path : str
        Path GDC file was downloaded to/found at.
    """

    if det == 'IR':
        detector = 'WFC3IR'
        gdc_filename = 'STDGDC_WFC3IR.fits'
    elif det == 'UVIS':
        # must do [:5] to handle LP filters
        detector = 'WFC3UV'
        gdc_filename = 'STDGDC_WFC3UV_{}.fits'.format(filt[:5])
    elif det == 'WFC':
        detector = 'ACSWFC'
        gdc_filename = 'STDGDC_ACSWFC_{}.fits'.format(filt[:5])
    elif det == 'HRC':
        detector = 'ACSHRC'
        gdc_filename = 'STDGDC_ACSHRC_{}.fits'.format(filt[:5])

    gdc_path = '{}/{}'.format(path, gdc_filename)

    if not os.path.exists(gdc_path):
        print('Downloading GDC File.  Size ~= 300MB, this may take a while.')
        url = 'http://www.stsci.edu/~jayander/STDGDCs/{}/{}'.format(
            detector, gdc_filename)
        urllib.request.urlretrieve(url, gdc_path)
        print('Saving GDC file to {}'.format(gdc_path))


    if not os.path.exists(gdc_filename):
        print('Copying GDC file to current directory')
        shutil.copy(gdc_path, '.')

    print('Using GDC file {}'.format(gdc_filename))
    return gdc_filename

def get_focus_dependent_psf(path, filt):
    """
    Checks if PSF file exists and if not downloads from WFC3 page.

    Parameters
    ----------
    path : str
        Path where to to look for/store PSF files.
    filt : str
        Filter which to get the focus dependent PSF for.

    Returns
    -------
    psf_path : str
        Path focus dependent PSF file was downloaded to/found at.
    """

    filt = filt[:5]
    psf_filename = 'STDPBF_WFC3UV_{}.fits'.format(filt)
    psf_path = '{}/{}'.format(path, psf_filename)

    if not os.path.exists(psf_path):
        print('Downloading PSF')
        psf_filename = 'STDPBF_WFC3UV_{}.fits'.format(filt)

        url = 'http://www.stsci.edu/~jayander/STDPBFs/WFC3UV/{}'.format(
            psf_filename)
        urllib.request.urlretrieve(url, psf_path)
        print('Saving PSF file to {}'.format(psf_path))

    if not os.path.exists(psf_filename):
        print('Copying PSF file to current directory')
        shutil.copy(psf_path, '.')
    print('Using PSF file {}'.format(psf_filename))
    return psf_filename

def get_standard_psf(path, filt, det):
    """
    Checks if PSF file exists and if not downloads from WFC3 page.

    Parameters
    ----------
    path : str
        Path where to to look for/store PSF files.
    filt : str
        Filter which to get the PSF for.

    Returns
    -------
    psf_path : str
        Path PSF file was downloaded to/found at.
    """
    filt = filt[:5]
    psf_dets = {'ir':'WFC3IR', 'uvis':'WFC3UV', 'wfc':'ACSWFC'}
    detector = psf_dets[det.lower()]
    psf_filename = 'PSFSTD_{}_{}.fits'.format(detector, filt)
    psf_path = '{}/{}'.format(path, psf_filename)



    if not os.path.exists(psf_path):
        print('Downloading PSF')


        url = 'http://www.stsci.edu/~jayander/STDPSFs/{}/{}'.format(
            detector, psf_filename)
        urllib.request.urlretrieve(url, psf_path)
        print('Saving PSF file to {}'.format(psf_path))

    if not os.path.exists(psf_filename):
        print('Copying PSF file to current directory')
        shutil.copy(psf_path, '.')
    print('Using PSF file {}'.format(psf_filename))
    return psf_filename

def validate_file(input_file):
    """
    Makes sure input_file is a valid fits file.

    Tries to open the image to ensure the fits file is readable.
    Good for ensuring the download didn't fail/download garbage

    Parameters
    ----------
    input_file : string
        Filename of fits file to validate
    """
    try:
        hdu = fits.open(input_file, ignore_missing_end=True)
        hdu.close()
    except IOError:
        raise IOError('Could not read input file {}, ensure it exists \
                and is not corrupted'.format(input_file))
    except Exception as e:
        raise Exception(e)


def check_images(input_images):
    """
    Checks images to make sure they are wfc3 and one filter.

    Software currently only has full support for WFC3.  Furthermore,
    only a single filter should be passed into most of the functions,
    as the measurements are filter specific.  This function raises an
    exception if those conditions are not met.

    Parameters
    ----------
    input_images : list
        List of image filenames (strings).  Must be fits files.

    Returns
    -------
    filter : str
        Filter name.

    """

    filter_list = []
    for im in input_images:
        hdr = fits.getheader(im)
        inst = hdr['INSTRUME']
        if inst == 'ACS':
            print('IS ACS')
            filt1 = hdr['FILTER1']
            filt2 = hdr['FILTER2']
            filt = filt1
            if filt1 == 'CLEAR1L' or filt1 == 'CLEAR1S':
                filt = filt2
            filter_list.append(filt)
        elif inst not in ['WFC3', 'ACS']:
            raise ValueError('Image {} is not a WFC3 image'.format(im))
        else:
            filter_list.append(hdr['FILTER'])

    if len(set(filter_list)) != 1:
        for i, im in enumerate(input_images):
            print('{}: {}'.format(im,filter_list[i]))
        raise RuntimeError('Multiple filters detected in inputs, \
        only run on one filter at a time.')

    return filter_list[0]

def check_focus(input_catalogs):
    """
    Read the measured focus level from the input catalogs

    If hst1pass was run using focus based PSFs, the focus level for
    each image is stored in the catalog output from hst1pass (the
    .xympqks file).

    Parameters
    ----------
    input_catalogs : list
        List of catalog filenames (strings).

    Returns
    -------
    focus_dict : dict
        Dictionary where dict[input_catalog] = focus level.
    """
    focus_dict = {}
    for cat in input_catalogs:
        lines = open(cat).readlines()
        for line in lines:
            if 'FOC_LEVu' in line:
                focus = line.split()[-1]
                break
        focus_dict[cat] = float(focus)
    return focus_dict


def _get_exec_name():
    """Checks the current OS to select correct executable"""
    platform = sys.platform
    if platform == 'darwin':
        exec_name = 'hst1pass_darwin.e'
    elif platform.lower().startswith('linux'):
        exec_name = 'hst1pass_linux.e'
    else:
        raise OSError('Exectuable not compiled for this OS, only for \
                        Mac OSX and linux.  Use run_python_psf_fitting()')
    return exec_name

def _get_exec_path():
    exec_name = _get_exec_name()
    direc = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(direc, exec_name)

def run_and_print_output(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True,
                               close_fds=True)
    while True:
        output = process.stdout.readline()
        if len(output) == 0 and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc
