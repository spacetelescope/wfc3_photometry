from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS, Sip
from itertools import product

import numpy as np

def make_wcs(row):
    """Create a WCS for an FLT/FLC sci extension from
    values in row of a DRZ/DRC HDRTAB

    This function takes in a row in a HDRTAB extension
    from a drizzled image and reconstructs the WCS for the
    extension of the exposure corresponding to that row.  See
    make_wcs_list for further information.

    """
    hdr = fits.Header()
    dts = row.dtype
    for i, cn in enumerate(row.colnames):
        if '>f' in dts[i].str:
            if np.isnan(row[cn]):
                continue
        hdr[cn] = row[cn]
    return WCS(hdr)

def make_wcs_list(hdrtab):
    """
    Create list of WCS objects of input exposures/extensions of
    a drizzled image, by reading parameters from DRZ HDRTAB.

    This function regenerates the WCS of input exposures, by
    reading the WCS parameters from the corresponding row in
    the HDRTAB.  This allows recovery of exposure astromertic solutions
    without needing the files of the exposures themselves.
    Furthermore, this ensures consistency of the exposure WCS and the
    drizzled image, as the HDRTAB is created at drizzle runtime.


    NOTE: This does NOT include the look-up table distortions,
    so positions from this WCS may differ from the real WCS by
    a few hundredths of a pixel.

    Parameters
    ----------
    hdrtab : astropy Table
        The HDRTab from the drizzled image file.

    Returns
    -------
    wcs_list : list
        List of WCS objects for each sci extension that
        created drizzled image.
    """
    wcs_list = []
    for row in hdrtab:
        wcs_list.append(make_wcs(row))
    return wcs_list

def compute_coverage(drz, cat):
    """
    This function computes how many exposures and seconds cover a
    catalog of positions in a drizzled frame.

    For every position in a catalog corresponding to a drizzled frame,
    this function returns how many exposures covered that point, and
    how many seconds of exposure time covered that point.  It does this
    by reconstructing the WCS of every input exposure/extension, and
    transforming the positions in the catalog to the exposure/extension
    frame, and seeing if the transformed positions land on the array.
    It then adds up the exposure time of each exposure, accounting for
    each exposures coverage of the different positions.  The WCS's are
    reconstructed from the HDRTAB, see make_wcs_list() for details.

    For this to work properly, the drizzled image must be multi extension,
    with a single SCI extension and HDRTAB extension (this is default).
    Furthermore, the catalog must contain columns named 'X' and 'Y',
    which are 1-indexed pixel positions in the drizzled image frame.

    NOTE: As the lookup table distortions are not stored in the HDRTAB,
    the transformed positions are slightly off from the real positions,
    but the errors are small, generally within a few hundredths of a pixel.
    NOTE: This does NOT account for effects like saturation, or guide star
    tracking issues.

    Parameters
    ----------
    drz : str
        Path to drizzled image
    cat : astropy Table
        Catalog containing X and Y positions in drizzled frame

    Returns
    -------
    wcs_list : list
        List of WCS objects for each sci extension that
        created drizzled image.
    """

    """
    ref_wcs = WCS(fits.getheader(drz,'SCI'))
    hdrtab = Table.read(drz,'HDRTAB')
    wcs_list = make_wcs_list(hdrtab)

    xys = np.array([cat['X'], cat['Y']]).T
    rds = ref_wcs.all_pix2world(xys, 1)

    arr = np.zeros((len(hdrtab), len(cat)))
    for i, wcs in enumerate(wcs_list):
        arr[i] = _transform_points(rds, wcs)
    n_exp = np.sum(arr, axis=0)
    etime = np.sum(arr*np.array(hdrtab['EXPTIME'])[:,None], axis=0)

    return n_exp, etime

def _prefilter_coordinates(input_skycoords, xmin, xmax, ymin, ymax, flt_wcs):
    # this probably will explode if the image contains a celestial pole?
    inp_ra, inp_dec = np.array(input_skycoords).T

    corners = [p for p in product([xmin, xmax], [ymin, ymax])]
    corner_ra, corner_dec = flt_wcs.all_pix2world(corners, 1).T

    mask_ra = (inp_ra>np.amin(corner_ra)) & (inp_ra < np.max(corner_ra))
    mask_dec = (inp_dec>np.amin(corner_dec)) & (inp_dec<np.amax(corner_dec))
    mask = mask_ra & mask_dec
    return mask
    # return input_skycoords[mask]


def _transform_points(input_skycoords, flt_wcs, padding=0):

    xmin = ymin = -1 * padding
    xmax, ymax = np.array(flt_wcs._naxis) + padding

    bboxmask = _prefilter_coordinates(input_skycoords, xmin, xmax,
                                             ymin, ymax, flt_wcs)
    inds = np.arange(len(input_skycoords)).astype(int)
    filtered_coords = input_skycoords[bboxmask]
    bbinds = inds[bboxmask]
    xc, yc = flt_wcs.all_world2pix(filtered_coords, 1).T
    mask = (xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax)

    good_inds = bbinds[mask]
    on_frame = np.zeros(inds.shape)
    on_frame[good_inds] = 1.

    return on_frame
