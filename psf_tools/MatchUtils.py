import astropy.units as u
import numpy as np

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from bisect import bisect_left

def make_id_list(coord_ints, x_digits=None, y_digits=None):
    """
    Turns pixel coordinates into IDs for each star.

    Each ID is just the X and Y integer pixel postions concatenated.
    For example, (3894, 215) becomes 38940215.  These IDs are then
    used in the matching process.

    Parameters
    ----------
    coord_ints : ndarray
        Nx2 array of coordinates.  Must be integers
    x_digits : int, optional
        Number of digits in the x coordinates.  If none, will
        calculate from largest value.  Necessary for correct
        padding of of ID.
    y_digits : int, optional
        Number of digits in the y coordinates.  If none, will
        calculate from largest value.  Necessary for correct
        padding of of ID.

    Returns
    -------
    ids: list
        List of ids for the source positions from the input coords.
    """
    xs = coord_ints[0]
    ys = coord_ints[1]
    if not x_digits:
        x_digits = len(str(max(xs)))
    if not y_digits:
        y_digits = len(str(max(ys)))
    ids = []
    for i in range(len(xs)):
        coord_id = str(xs[i]).zfill(x_digits) + str(ys[i]).zfill(y_digits)
        ids.append(coord_id)
    return ids


def get_match_indices(master_ids, input_ids):
    """Matches ID from master list with input ids from input catalog"""

    matched_indices = []
    input_sorted_inds = np.argsort(input_ids)
    input_ids = sorted(input_ids)
    for master_id in master_ids:
        ind = binary_search_index(input_ids, master_id)

        if ind >= 0:
            matched_indices.append(input_sorted_inds[ind])
        else:
            matched_indices.append(-1)
    print 'N matched: {}'.format(len(matched_indices)-matched_indices.count(-1))
    return matched_indices

def binary_search_index(a, x):
    """Binary searches array a for element x and returns index"""
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def match_final_catalogs(cat1, cat2, max_distance=.05):
    """
    Matches two final catalogs so each row of the returned tables
    correspond to the same star.

    This function takes in two final catalogs and matches the sources
    listed in them.  This is done by finding closest point (in sky
    space) of in cat2 for every point in cat1.  Points closer than
    max_distance are considered to be the same source. The
    returned tables only contain the sources that were matched.

    Parameters
    ----------
    cat1 : astropy.table.Table or str
        The first table (or filename) to be matched
    cat2 : astropy.table.Table or str
        The second table (or filename) to be matched
    max_distance : float, optional
        The threshold (in arcsec) which distances must be in for
        sources to be considered a match.
    """

    if type(cat1) == str:
        cat1 = Table.read(cat1, format='ascii.commented_header')
    if type(cat2) == str:
        cat2 = Table.read(cat2, format='ascii.commented_header')

    cat1_skycoord = SkyCoord(cat1['rbar']*u.deg, cat1['dbar']*u.deg)
    cat2_skycoord = SkyCoord(cat2['rbar']*u.deg, cat2['dbar']*u.deg)

    idx, ang, wat = cat1_skycoord.match_to_catalog_sky(cat2_skycoord)
    distance_mask = ang.arcsec  < max_distance

    matched_cat1 = cat1[distance_mask]
    matched_cat2 = cat2[idx][distance_mask]
    return matched_cat1, matched_cat2

def match_to_master_catalog(master_cat, sci_cat, max_distance=.05):

    if type(cat1) == str:
        master = Table.read(master_cat, format='ascii.commented_header')
    if type(cat2) == str:
        sci = Table.read(sci_cat, format='ascii.commented_header')
        
